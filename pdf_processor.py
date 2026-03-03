"""
pdf_processor_v2.py - Production PDF Processor (THE PARAGRAPH KILLER)
=====================================================================
✅ Paragraph Killer (Stops extracting when it hits paragraph text)
✅ Layout Engine (Perfect spatial representation)
✅ Column-Cropping (Prevents Table 1 & 2 from merging)
"""

import os
import re
import uuid
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import fitz  # PyMuPDF
import pdfplumber

from models import ProcessedDocument, ProcessedEquation, ProcessedTable, ProcessedFigure

logger = logging.getLogger(__name__)

class StrictEquationDetector:
    STRONG_OPERATORS = ['=', '≈', '≤', '≥', '∫', '∑', '∏', '∝', '→', '∈']
    
    @classmethod
    def is_equation(cls, text: str, bbox: Tuple[float, float, float, float] = None) -> bool:
        text = text.strip()
        if len(text) < 8 or len(text) > 300: return False
        if re.search(r'(https?://|www\.|doi:|arxiv:|github\.com|\.pdf)', text, re.IGNORECASE): return False
        if re.match(r'^\[\d+\]', text): return False
        if re.match(r'^\s*(Figure|Fig|Table|Appendix|Algorithm|Listing)\b', text, re.IGNORECASE): return False
        if 'id=' in text.lower() or 'id =' in text.lower(): return False
        if text.startswith(')'): return False
        if not any(op in text for op in cls.STRONG_OPERATORS): return False

        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        if len(words) > 6: return False 
        return True

    @classmethod
    def extract_latex_from_text(cls, text: str) -> Optional[str]:
        if '\\' in text: return text
        latex = text.replace('\n', ' ')
        greek_map = {'α': r'\alpha ', 'β': r'\beta ', 'γ': r'\gamma ', 'δ': r'\delta ', 'η': r'\eta ', 'θ': r'\theta ', 'λ': r'\lambda ', 'μ': r'\mu '}
        for greek, tex in greek_map.items(): latex = latex.replace(greek, tex)
        
        latex = re.sub(r'≈\s*X', r'\\approx \\sum', latex)
        latex = re.sub(r'=\s*X', r'= \\sum', latex)
        latex = latex.replace('∑', r'\sum ').replace('∏', r'\prod ')
        
        latex = latex.replace('\ufffd', '').replace('\x01', '').replace('\x00', '')
        latex = latex.replace('exp  d(z)', 'exp(d(z))').replace('exp d(z)', 'exp(d(z))')
        return latex.strip()


class TableDetector:
    @staticmethod
    def extract_table_markdown(plumber_page, fitz_page) -> List[Dict[str, Any]]:
        tables = []
        try:
            # 1. CAPTION DETECTION
            table_captions = []
            blocks = fitz_page.get_text("blocks")
            for b in blocks:
                if b[6] == 0:  
                    text = b[4].strip()
                    if re.match(r'^Table\s+\d+[:\.]', text, re.IGNORECASE) and len(text.split()) < 40:
                        table_captions.append({"text": text.replace('\n', ' '), "bbox": b[:4]})
            
            if not table_captions: 
                return tables

            page_width = fitz_page.rect.width
            page_height = fitz_page.rect.height
            
            # 2. ISOLATE AND EXTRACT "TEXT PHOTOGRAPH"
            for cap in table_captions:
                cx0, cy0, cx1, cy1 = cap["bbox"]
                
                is_left = cx1 < (page_width / 2) + 20
                is_right = cx0 > (page_width / 2) - 20
                
                crop_x0 = 0 if is_left else (page_width / 2)
                crop_x1 = (page_width / 2) if is_left and not is_right else page_width
                if is_left and is_right:  
                    crop_x0, crop_x1 = 0, page_width
                    
                crop_box = (crop_x0, max(0, cy1 - 2), crop_x1, min(page_height, cy1 + 250))
                cropped_plumber = plumber_page.within_bbox(crop_box)
                
                raw_text = cropped_plumber.extract_text(layout=True)
                
                if not raw_text or len(raw_text.strip()) < 15: 
                    continue
                
                cleaned_lines = []
                empty_streak = 0
                
                # 3. 🚨 THE PARAGRAPH KILLER ENGINE 🚨
                for line in raw_text.split('\n'):
                    stripped = line.strip()
                    
                    if not stripped:
                        empty_streak += 1
                        # If we see 2 blank lines, the table is finished.
                        if empty_streak >= 2 and cleaned_lines: break
                        continue
                        
                    empty_streak = 0
                    
                    # Kill Switch 1: Pdfplumber squished paragraph (long string, no spaces)
                    if len(stripped) > 30 and ' ' not in stripped:
                        if cleaned_lines: break # Exit table!
                        
                    # Kill Switch 2: Section Headers (e.g., "4.2 Abstractive...")
                    if re.match(r'^\d+\.\d+\s+[A-Z]', stripped):
                        if cleaned_lines: break # Exit table!
                        
                    # Kill Switch 3: Standard prose (long sentence, mostly letters)
                    alpha_ratio = sum(c.isalpha() for c in stripped) / len(stripped) if stripped else 0
                    if len(stripped) > 50 and alpha_ratio > 0.85 and not re.search(r'\s{3,}', line):
                        if cleaned_lines: break # Exit table!
                        
                    cleaned_lines.append(line)
                    
                if len(cleaned_lines) < 2: 
                    continue
                
                md_text = "```text\n" + "\n".join(cleaned_lines) + "\n```"
                
                tables.append({
                    "markdown": md_text,
                    "bbox": crop_box,
                    "caption": cap["text"],
                    "text": " ".join(cleaned_lines)
                })
                
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        return tables


class SectionDetector:
    SECTION_PATTERNS = [r'^\d+\.?\s+[A-Z]', r'^[IVX]+\.?\s+[A-Z]', r'^(Abstract|Introduction|Methods|Results|Discussion|Conclusion|References)']
    @classmethod
    def detect_section(cls, text: str) -> Optional[str]:
        text = text.strip()
        if len(text) > 100: return None
        for p in cls.SECTION_PATTERNS:
            if re.match(p, text, re.IGNORECASE): return text
        return None

@dataclass
class ExtractedEquation:
    text: str; latex: str; page_num: int; bbox: Tuple[float, float, float, float]; global_number: int; section: str = ""
@dataclass
class ExtractedTable:
    text: str; markdown: str; page_num: int; bbox: Tuple[float, float, float, float]; global_number: int; caption: str = ""; section: str = ""
@dataclass
class ExtractedFigure:
    caption: str; page_num: int; bbox: Tuple[float, float, float, float]; global_number: int; image_path: Optional[str] = None; section: str = ""

class PDFProcessorV2:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.eq_det = StrictEquationDetector()
        self.tbl_det = TableDetector()
        self.sec_det = SectionDetector()
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        self.eq_counter, self.tbl_counter, self.fig_counter, self.current_section = 0, 0, 0, ""
        doc = fitz.open(pdf_path)
        plumber_doc = pdfplumber.open(pdf_path)
        equations, tables, figures, page_texts = [], [], [], []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            plumber_page = plumber_doc.pages[page_num]
            
            page_tables = self.tbl_det.extract_table_markdown(plumber_page, page)
            for pt in page_tables:
                self.tbl_counter += 1
                tbl = ExtractedTable(str(pt["text"]), pt["markdown"], page_num, pt.get("bbox", (0,0,0,0)), self.tbl_counter, pt.get("caption", ""), self.current_section)
                tables.append(tbl)
            
            blocks = page.get_text("blocks")
            page_text_parts = []
            
            for b in blocks:
                x0, y0, x1, y1, text, block_no, block_type = b
                text = text.strip()
                bbox = (x0, y0, x1, y1)
                
                if not text and block_type == 0: continue
                
                if block_type == 1:
                    self.fig_counter += 1
                    fig = ExtractedFigure(f"Image from page {page_num + 1}", page_num, bbox, self.fig_counter, self.current_section)
                    figures.append(fig)
                    page_text_parts.append(f"[Figure {self.fig_counter}]")
                    continue
                    
                if block_type == 0:
                    if re.match(r'^\s*(Figure|Fig\.)\s*\d+', text, re.IGNORECASE):
                        self.fig_counter += 1
                        caption = text.replace('\n', ' ')
                        fig = ExtractedFigure(caption, page_num, bbox, self.fig_counter, self.current_section)
                        figures.append(fig)
                        page_text_parts.append(f"[{caption}]")
                        continue

                    section_title = self.sec_det.detect_section(text)
                    if section_title:
                        self.current_section = section_title
                        page_text_parts.append(text)
                        continue
                    
                    if self.eq_det.is_equation(text, bbox):
                        self.eq_counter += 1
                        latex = self.eq_det.extract_latex_from_text(text)
                        eq = ExtractedEquation(text.replace('\n', ' '), latex, page_num, bbox, self.eq_counter, self.current_section)
                        equations.append(eq)
                        page_text_parts.append(f"[Equation {self.eq_counter}]")
                        continue
                    
                    page_text_parts.append(text)
            
            page_texts.append("\n\n".join(page_text_parts))
        
        num_pages = len(doc)
        doc.close()
        plumber_doc.close()
        
        return {'equations': equations, 'tables': tables, 'figures': figures, 'page_texts': page_texts, 'num_pages': num_pages}

class EnhancedPDFProcessor:
    def __init__(self, config: Dict[str, Any] = None):
        self.processor = PDFProcessorV2(config or {})
        
    def process_pdf(self, pdf_path: str) -> ProcessedDocument:
        raw = self.processor.process_pdf(pdf_path)
        
        equations = [ProcessedEquation(f"eq_{uuid.uuid4().hex[:8]}", e.global_number, e.text, e.latex, e.page_num, e.bbox, e.section) for e in raw.get('equations', [])]
        tables = [ProcessedTable(f"tb_{uuid.uuid4().hex[:8]}", t.global_number, t.caption, t.page_num, t.markdown, t.section, bbox=t.bbox) for t in raw.get('tables', [])]
        figures = [ProcessedFigure(f"fig_{uuid.uuid4().hex[:8]}", f.global_number, f.caption, f.page_num, f.image_path, f.bbox, f.section) for f in raw.get('figures', [])]
            
        return ProcessedDocument(
            doc_id=f"doc_{uuid.uuid4().hex[:8]}", filename=os.path.basename(pdf_path), num_pages=raw.get('num_pages', 0),
            page_texts=raw.get('page_texts', []), enriched_page_texts=raw.get('page_texts', []), sections=[], 
            equations=equations, tables=tables, figures=figures, title=os.path.basename(pdf_path)
        )
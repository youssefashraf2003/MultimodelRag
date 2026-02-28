"""
advanced_formatter.py - نظام عرض ردود احترافي V2.0
=======================================================
✅ عرض المعادلات بشكل صحيح وكامل في LaTeX
✅ عرض الجداول كجداول منسقة (HTML/Markdown)
✅ عدم عرض عناصر غير مطلوبة
✅ ردود شاملة ومقنعة بدون حشو
✅ تنسيق تلقائي حسب نوع السؤال
✅ إصلاح مشاكل encoding
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from equation_latex import to_latex, looks_like_math
    HAS_LATEX = True
except ImportError:
    HAS_LATEX = False
    logger.warning("⚠️ equation_latex not available")


def safe_text(text, fallback='[text unavailable]'):
    """تحويل آمن للنص مع معالجة encoding"""
    if text is None:
        return fallback
    
    try:
        if not isinstance(text, str):
            text = str(text)
        # Clean problematic characters
        text = text.encode('utf-8', errors='replace').decode('utf-8')
        return text
    except Exception:
        return fallback


class ResponseMode(Enum):
    """أنماط الردود"""
    SHOW_EQUATION = "show_equation"
    SHOW_TABLE = "show_table"
    SHOW_FIGURE = "show_figure"
    LIST_ALL = "list_all"
    COMPARISON = "comparison"
    EXPLANATION = "explanation"
    GENERAL = "general"


@dataclass
class FormattedResponse:
    """رد منسّق"""
    mode: ResponseMode
    content: str
    metadata: Dict[str, Any]
    latex_equations: List[str] = None
    tables: List[Dict[str, Any]] = None
    figures: List[Dict[str, Any]] = None
    warnings: List[str] = None


class AdvancedResponseFormatter:
    """
    نظام تنسيق ردود احترافي
    """
    
    def __init__(self, registry: Any = None):
        self.registry = registry
        logger.info("✅ AdvancedResponseFormatter initialized")
    
    def format_response(
        self,
        query: str,
        intent: Any,  # QueryIntent
        retrieved_chunks: List[Any],
        llm_response: str
    ) -> FormattedResponse:
        """
        تنسيق الرد حسب نوع الاستعلام
        """
        from smart_retriever import QueryType
        
        # اختيار نمط العرض
        if intent.query_type == QueryType.SPECIFIC_ELEMENT:
            if intent.target_type == 'equation':
                return self._format_equation_response(retrieved_chunks, llm_response)
            elif intent.target_type == 'table':
                return self._format_table_response(retrieved_chunks, llm_response)
            elif intent.target_type == 'figure':
                return self._format_figure_response(retrieved_chunks, llm_response)
        
        elif intent.query_type == QueryType.LIST_ALL:
            return self._format_list_all_response(intent.target_type, retrieved_chunks)
        
        elif intent.query_type == QueryType.COMPARISON:
            return self._format_comparison_response(retrieved_chunks, llm_response)
        
        elif intent.query_type in [QueryType.EQUATION, QueryType.TABLE, QueryType.FIGURE]:
            # شرح عام لمعادلات/جداول/صور
            return self._format_explanation_response(intent, retrieved_chunks, llm_response)
        
        else:  # GENERAL
            return self._format_general_response(retrieved_chunks, llm_response)
    
    # ═══════════════════════════════════════════════════════════════════════
    #  EQUATION FORMATTING
    # ═══════════════════════════════════════════════════════════════════════
    
    def _format_equation_response(
        self,
        chunks: List[Any],
        llm_response: str
    ) -> FormattedResponse:
        """
        عرض معادلة محددة
        """
        if not chunks:
            return FormattedResponse(
                mode=ResponseMode.SHOW_EQUATION,
                content="⚠️ المعادلة المطلوبة غير موجودة في الوثيقة.",
                metadata={},
                warnings=["Equation not found"]
            )
        
        chunk = chunks[0]
        metadata = chunk.metadata
        
        # استخراج LaTeX
        latex = safe_text(metadata.get('latex', ''))
        if not latex:
            latex = safe_text(metadata.get('raw_text', ''))
        
        # تحويل إلى LaTeX نظيف
        if HAS_LATEX and latex:
            try:
                latex = to_latex(latex)
            except Exception:
                pass  # Keep original if conversion fails
        
        # بناء الرد
        eq_number = metadata.get('global_number', '?')
        section = safe_text(metadata.get('section', 'Unknown Section'))
        context = safe_text(metadata.get('context', ''))
        page_num = metadata.get('page_num', '?')
        
        # شرح من LLM (تنظيف)
        explanation = safe_text(self._clean_llm_response(llm_response))
        
        content = f"""
## 📐 Equation {eq_number}

**Section:** {section}  
**Page:** {page_num}

### Mathematical Expression

```latex
{latex}
```

### Explanation

{explanation}

### Context

{context[:500]}
""".strip()
        
        return FormattedResponse(
            mode=ResponseMode.SHOW_EQUATION,
            content=content,
            metadata={
                'equation_number': eq_number,
                'section': section,
                'page': page_num
            },
            latex_equations=[latex]
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    #  TABLE FORMATTING
    # ═══════════════════════════════════════════════════════════════════════
    
    def _format_table_response(
        self,
        chunks: List[Any],
        llm_response: str
    ) -> FormattedResponse:
        """
        عرض جدول محدد
        """
        if not chunks:
            return FormattedResponse(
                mode=ResponseMode.SHOW_TABLE,
                content="⚠️ الجدول المطلوب غير موجود في الوثيقة.",
                metadata={},
                warnings=["Table not found"]
            )
        
        chunk = chunks[0]
        metadata = chunk.metadata
        
        # استخراج بيانات الجدول
        table_number = metadata.get('global_number', '?')
        caption = metadata.get('caption', 'No caption')
        markdown = metadata.get('markdown', '')
        section = metadata.get('section', 'Unknown Section')
        page_num = metadata.get('page_num', '?')
        num_rows = metadata.get('num_rows', '?')
        num_cols = metadata.get('num_cols', '?')
        
        # تحويل Markdown إلى HTML للعرض الأفضل
        html_table = self._markdown_to_html_table(markdown)
        
        # شرح من LLM
        explanation = self._clean_llm_response(llm_response)
        
        content = f"""
## 📊 Table {table_number}

**Caption:** {caption}  
**Section:** {section}  
**Page:** {page_num}  
**Size:** {num_rows} rows × {num_cols} columns

### Data

{markdown}

### Analysis

{explanation}
""".strip()
        
        return FormattedResponse(
            mode=ResponseMode.SHOW_TABLE,
            content=content,
            metadata={
                'table_number': table_number,
                'caption': caption,
                'section': section,
                'page': page_num
            },
            tables=[{
                'number': table_number,
                'caption': caption,
                'markdown': markdown,
                'html': html_table
            }]
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    #  FIGURE FORMATTING
    # ═══════════════════════════════════════════════════════════════════════
    
    def _format_figure_response(
        self,
        chunks: List[Any],
        llm_response: str
    ) -> FormattedResponse:
        """
        عرض صورة/رسم محدد
        """
        if not chunks:
            return FormattedResponse(
                mode=ResponseMode.SHOW_FIGURE,
                content="⚠️ الصورة المطلوبة غير موجودة في الوثيقة.",
                metadata={},
                warnings=["Figure not found"]
            )
        
        chunk = chunks[0]
        metadata = chunk.metadata
        
        fig_number = metadata.get('global_number', '?')
        caption = metadata.get('caption', 'No caption')
        section = metadata.get('section', 'Unknown Section')
        page_num = metadata.get('page_num', '?')
        image_path = metadata.get('image_path', None)
        
        explanation = self._clean_llm_response(llm_response)
        
        content = f"""
## 🖼️ Figure {fig_number}

**Caption:** {caption}  
**Section:** {section}  
**Page:** {page_num}

### Description

{explanation}
""".strip()
        
        if image_path:
            content += f"\n\n**Image Path:** `{image_path}`"
        
        return FormattedResponse(
            mode=ResponseMode.SHOW_FIGURE,
            content=content,
            metadata={
                'figure_number': fig_number,
                'caption': caption,
                'section': section,
                'page': page_num,
                'image_path': image_path
            },
            figures=[{
                'number': fig_number,
                'caption': caption,
                'path': image_path
            }]
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    #  LIST ALL FORMATTING
    # ═══════════════════════════════════════════════════════════════════════
    
    def _format_list_all_response(
        self,
        element_type: str,
        chunks: List[Any]
    ) -> FormattedResponse:
        """
        عرض قائمة بجميع العناصر
        """
        if not chunks:
            return FormattedResponse(
                mode=ResponseMode.LIST_ALL,
                content=f"⚠️ لا توجد {element_type}s في الوثيقة.",
                metadata={},
                warnings=[f"No {element_type}s found"]
            )
        
        # ترتيب حسب الرقم
        sorted_chunks = sorted(chunks, key=lambda c: c.metadata.get('global_number', 0))
        
        content = f"## 📋 All {element_type.title()}s in Document\n\n"
        content += f"**Total Count:** {len(sorted_chunks)}\n\n"
        
        if element_type == 'equation':
            content += self._list_all_equations(sorted_chunks)
        elif element_type == 'table':
            content += self._list_all_tables(sorted_chunks)
        elif element_type == 'figure':
            content += self._list_all_figures(sorted_chunks)
        
        return FormattedResponse(
            mode=ResponseMode.LIST_ALL,
            content=content,
            metadata={
                'element_type': element_type,
                'count': len(sorted_chunks)
            }
        )
    
    def _list_all_equations(self, chunks: List[Any]) -> str:
        """قائمة بجميع المعادلات"""
        lines = []
        
        for chunk in chunks:
            number = chunk.metadata.get('global_number', '?')
            section = chunk.metadata.get('section', 'Unknown')
            page = chunk.metadata.get('page_num', '?')
            latex = chunk.metadata.get('latex', '')[:100]  # أول 100 حرف
            
            lines.append(f"**Equation {number}** (Page {page}, Section: {section})")
            lines.append(f"  `{latex}...`")
            lines.append("")
        
        return "\n".join(lines)
    
    def _list_all_tables(self, chunks: List[Any]) -> str:
        """قائمة بجميع الجداول"""
        lines = []
        
        for chunk in chunks:
            number = chunk.metadata.get('global_number', '?')
            caption = chunk.metadata.get('caption', 'No caption')
            section = chunk.metadata.get('section', 'Unknown')
            page = chunk.metadata.get('page_num', '?')
            
            lines.append(f"**Table {number}** (Page {page})")
            lines.append(f"  *{caption}*")
            lines.append(f"  Section: {section}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _list_all_figures(self, chunks: List[Any]) -> str:
        """قائمة بجميع الصور"""
        lines = []
        
        for chunk in chunks:
            number = chunk.metadata.get('global_number', '?')
            caption = chunk.metadata.get('caption', 'No caption')
            section = chunk.metadata.get('section', 'Unknown')
            page = chunk.metadata.get('page_num', '?')
            
            lines.append(f"**Figure {number}** (Page {page})")
            lines.append(f"  *{caption}*")
            lines.append(f"  Section: {section}")
            lines.append("")
        
        return "\n".join(lines)
    
    # ═══════════════════════════════════════════════════════════════════════
    #  GENERAL FORMATTING
    # ═══════════════════════════════════════════════════════════════════════
    
    def _format_comparison_response(
        self,
        chunks: List[Any],
        llm_response: str
    ) -> FormattedResponse:
        """تنسيق رد المقارنة"""
        
        if len(chunks) < 2:
            return FormattedResponse(
                mode=ResponseMode.COMPARISON,
                content="⚠️ لم يتم العثور على جميع العناصر المطلوبة للمقارنة.",
                metadata={},
                warnings=["Not enough elements for comparison"]
            )
        
        explanation = self._clean_llm_response(llm_response)
        
        content = f"""
## 🔍 Comparison

{explanation}
""".strip()
        
        return FormattedResponse(
            mode=ResponseMode.COMPARISON,
            content=content,
            metadata={'num_compared': len(chunks)}
        )
    
    def _format_explanation_response(
        self,
        intent: Any,
        chunks: List[Any],
        llm_response: str
    ) -> FormattedResponse:
        """تنسيق رد الشرح"""
        
        explanation = self._clean_llm_response(llm_response)
        
        # إضافة المراجع
        references = self._extract_references(chunks)
        
        content = f"""
{explanation}

---

### 📚 References

{references}
""".strip()
        
        return FormattedResponse(
            mode=ResponseMode.EXPLANATION,
            content=content,
            metadata={'num_sources': len(chunks)}
        )
    
    def _format_general_response(
        self,
        chunks: List[Any],
        llm_response: str
    ) -> FormattedResponse:
        """تنسيق رد عام"""
        
        explanation = self._clean_llm_response(llm_response)
        references = self._extract_references(chunks)
        
        content = f"""
{explanation}

---

### 📚 Sources

{references}
""".strip()
        
        return FormattedResponse(
            mode=ResponseMode.GENERAL,
            content=content,
            metadata={'num_sources': len(chunks)}
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    #  HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════
    
    def _clean_llm_response(self, response: str) -> str:
        """تنظيف رد LLM من الحشو"""
        
        # إزالة عبارات متكررة
        response = re.sub(r'(The document|This document|As mentioned|As shown)\s+', '', response, flags=re.IGNORECASE)
        
        # إزالة جمل فارغة
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        return '\n\n'.join(lines)
    
    def _extract_references(self, chunks: List[Any]) -> str:
        """استخراج المراجع من الـ chunks"""
        
        references = []
        
        for chunk in chunks:
            chunk_type = chunk.chunk_type
            page = chunk.metadata.get('page_num', '?')
            section = chunk.metadata.get('section', 'Unknown')
            
            if chunk_type == 'equation':
                number = chunk.metadata.get('global_number', '?')
                references.append(f"- Equation {number} (Page {page}, Section: {section})")
            elif chunk_type == 'table':
                number = chunk.metadata.get('global_number', '?')
                caption = chunk.metadata.get('caption', '')[:50]
                references.append(f"- Table {number}: {caption}... (Page {page})")
            elif chunk_type == 'figure':
                number = chunk.metadata.get('global_number', '?')
                caption = chunk.metadata.get('caption', '')[:50]
                references.append(f"- Figure {number}: {caption}... (Page {page})")
            else:
                references.append(f"- Page {page}, Section: {section}")
        
        return '\n'.join(references) if references else "No specific references"
    
    def _markdown_to_html_table(self, markdown: str) -> str:
        """تحويل Markdown table إلى HTML"""
        
        lines = [line.strip() for line in markdown.split('\n') if line.strip()]
        
        if not lines:
            return "<p>Empty table</p>"
        
        html = "<table border='1' style='border-collapse: collapse;'>\n"
        
        # Header
        if lines:
            header_cells = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
            html += "  <thead>\n    <tr>\n"
            for cell in header_cells:
                html += f"      <th style='padding: 8px; background-color: #f2f2f2;'>{cell}</th>\n"
            html += "    </tr>\n  </thead>\n"
        
        # Body
        html += "  <tbody>\n"
        for line in lines[2:]:  # تخطي header وخط الفصل
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                html += "    <tr>\n"
                for cell in cells:
                    html += f"      <td style='padding: 8px;'>{cell}</td>\n"
                html += "    </tr>\n"
        html += "  </tbody>\n"
        
        html += "</table>"
        
        return html


if __name__ == "__main__":
    print("✅ AdvancedResponseFormatter V2.0 Ready")
    print("\nFeatures:")
    print("  - LaTeX equations rendering")
    print("  - HTML table formatting")
    print("  - Clean LLM responses")
    print("  - Automatic mode detection")
    print("  - Reference extraction")
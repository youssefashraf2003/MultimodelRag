"""
enhanced_extractors.py - محسّنات استخراج المعادلات والجداول
=================================================================
✅ استخراج محسّن للمعادلات من PDF
✅ كشف وتنظيف اللاتيك
✅ استخراج الجداول بدقة عالية
✅ تحسين جودة السياق
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EnhancedEquationExtractor:
    """
    مستخرج معادلات محسّن مع دعم لجميع أنماط المعادلات
    """
    
    # أنماط المعادلات
    EQUATION_PATTERNS = [
        # LaTeX inline: $...$
        re.compile(r'\$([^\$\n]{3,200}?)\$', re.MULTILINE),
        
        # LaTeX display: $$...$$
        re.compile(r'\$\$([^\$]{3,500}?)\$\$', re.MULTILINE | re.DOTALL),
        
        # LaTeX environments
        re.compile(
            r'\\begin\{(equation|align|gather|multline)\*?\}(.*?)\\end\{\1\*?\}',
            re.MULTILINE | re.DOTALL
        ),
        
        # معادلات مع أرقام: (1), (2a), etc.
        re.compile(
            r'([^.\n]{20,200}?[=∝→←↔≡≜≃≈≠≤≥])([^.\n]{0,200}?)\s*\(\s*\d{1,3}[a-z]?\s*\)',
            re.MULTILINE
        ),
        
        # معادلات بدون أرقام لكن مع رموز رياضية واضحة
        re.compile(
            r'(?:^|\n)\s*([^.\n]{10,150}?[∫∑∏√∂∇][^.\n]{5,150}?[=∝→]?[^.\n]{0,100}?)(?:\n|$)',
            re.MULTILINE
        ),
        
        # معادلات احتمالية: p(x|y), P(A|B), etc.
        re.compile(
            r'\b([pPqQ](?:_[a-z]+)?)\s*\(([^)]{1,80}?)\|([^)]{1,80}?)\)',
            re.MULTILINE
        ),
        
        # دوال خاصة: exp, log, max, argmax, etc.
        re.compile(
            r'\b(exp|log|argmax|argmin|max|min|softmax|sigmoid)\s*\([^)]{3,150}?\)',
            re.MULTILINE
        ),
    ]
    
    # رموز يونانية
    GREEK_LETTERS = 'αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ'
    
    # رموز رياضية
    MATH_SYMBOLS = '∫∑∏√±∂∇∞≈≠≤≥∝∈∉⊂⊆∪∩→←↔⊕⊗⊤‖⟨⟩·×'
    
    @classmethod
    def extract_equations(
        cls,
        text: str,
        page_num: int = 0,
        min_confidence: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        استخراج جميع المعادلات من النص
        
        Args:
            text: النص المراد استخراج المعادلات منه
            page_num: رقم الصفحة
            min_confidence: الحد الأدنى للثقة
            
        Returns:
            قائمة المعادلات المستخرجة
        """
        equations = []
        seen = set()  # لتجنب التكرار
        
        for pattern_idx, pattern in enumerate(cls.EQUATION_PATTERNS):
            matches = pattern.finditer(text)
            
            for match in matches:
                # استخراج النص
                if pattern_idx <= 2:  # LaTeX patterns
                    eq_text = match.group(1) if match.lastindex >= 1 else match.group(0)
                else:
                    eq_text = match.group(0)
                
                # تنظيف
                eq_text = eq_text.strip()
                
                # التحقق من الجودة
                if not cls._is_valid_equation(eq_text):
                    continue
                
                # تجنب التكرار
                eq_hash = hash(eq_text[:100])
                if eq_hash in seen:
                    continue
                seen.add(eq_hash)
                
                # حساب الثقة
                confidence = cls._calculate_confidence(eq_text)
                if confidence < min_confidence:
                    continue
                
                # استخراج السياق
                context = cls._extract_context(text, match.start(), match.end())
                
                equations.append({
                    'text': eq_text,
                    'latex': cls._clean_latex(eq_text),
                    'page_number': page_num,
                    'confidence': confidence,
                    'context': context,
                    'pattern_type': pattern_idx,
                    'position': match.start()
                })
        
        # ترتيب حسب الموقع في الصفحة
        equations.sort(key=lambda e: e['position'])
        
        # إضافة أرقام تسلسلية
        for idx, eq in enumerate(equations, 1):
            eq['global_number'] = idx
        
        logger.info(f"✅ استخرج {len(equations)} معادلة من الصفحة {page_num}")
        
        return equations
    
    @classmethod
    def _is_valid_equation(cls, text: str) -> bool:
        """التحقق من أن النص معادلة صالحة"""
        if not text or len(text) < 3:
            return False
        
        # يجب أن تحتوي على رموز رياضية أو عمليات
        has_math_symbol = any(c in cls.GREEK_LETTERS + cls.MATH_SYMBOLS for c in text)
        has_operator = any(op in text for op in ['=', '∝', '→', '←', '≡', '≈', '≤', '≥'])
        has_function = any(fn in text for fn in ['exp', 'log', 'max', 'min', 'arg', 'sum', 'prod'])
        has_latex = '\\' in text or '$' in text
        
        # رفض النصوص العادية
        if not (has_math_symbol or has_operator or has_function or has_latex):
            return False
        
        # رفض النصوص الطويلة جدًا (أكثر من 500 حرف)
        if len(text) > 500:
            return False
        
        # رفض النصوص التي تحتوي على كلمات كثيرة جدًا
        words = text.split()
        if len(words) > 50:
            return False
        
        return True
    
    @classmethod
    def _calculate_confidence(cls, text: str) -> float:
        """حساب مستوى الثقة في المعادلة"""
        score = 0.5  # نقطة البداية
        
        # وجود LaTeX
        if '\\' in text:
            score += 0.2
        if '$' in text:
            score += 0.1
        
        # وجود رموز يونانية
        greek_count = sum(1 for c in text if c in cls.GREEK_LETTERS)
        score += min(greek_count * 0.05, 0.2)
        
        # وجود رموز رياضية
        symbol_count = sum(1 for c in text if c in cls.MATH_SYMBOLS)
        score += min(symbol_count * 0.03, 0.15)
        
        # وجود عمليات
        if '=' in text:
            score += 0.1
        if any(op in text for op in ['∝', '→', '≡', '≈']):
            score += 0.05
        
        # وجود دوال
        if any(fn in text for fn in ['exp', 'log', 'argmax', 'softmax']):
            score += 0.1
        
        # وجود subscripts/superscripts
        if '_' in text or '^' in text:
            score += 0.1
        
        return min(score, 1.0)
    
    @classmethod
    def _extract_context(cls, full_text: str, start: int, end: int, window: int = 200) -> str:
        """استخراج السياق المحيط بالمعادلة"""
        # أخذ نص قبل وبعد المعادلة
        context_start = max(0, start - window)
        context_end = min(len(full_text), end + window)
        
        context = full_text[context_start:context_end]
        
        # تنظيف
        context = ' '.join(context.split())
        
        return context[:400]  # حد أقصى 400 حرف
    
    @classmethod
    def _clean_latex(cls, text: str) -> str:
        """تنظيف نص LaTeX"""
        # إزالة $ إذا كانت موجودة
        text = text.replace('$', '')
        
        # إزالة المسافات الزائدة
        text = re.sub(r'\s+', ' ', text)
        
        # تنظيف الأقواس
        text = text.replace('( ', '(').replace(' )', ')')
        text = text.replace('{ ', '{').replace(' }', '}')
        
        # إزالة أرقام المعادلات في النهاية
        text = re.sub(r'\(\s*\d{1,3}[a-z]?\s*\)\s*$', '', text)
        
        return text.strip()


class EnhancedTableExtractor:
    """
    مستخرج جداول محسّن مع كشف دقيق للهيكل
    """
    
    @classmethod
    def extract_tables(
        cls,
        text: str,
        page_num: int = 0,
        min_rows: int = 3,
        min_cols: int = 2
    ) -> List[Dict[str, Any]]:
        """
        استخراج الجداول من النص
        
        Args:
            text: النص المراد استخراجه
            page_num: رقم الصفحة
            min_rows: الحد الأدنى للصفوف
            min_cols: الحد الأدنى للأعمدة
            
        Returns:
            قائمة الجداول المستخرجة
        """
        tables = []
        
        # البحث عن جداول بتنسيق markdown/ASCII
        table_blocks = cls._find_table_blocks(text)
        
        for idx, (table_text, start_pos) in enumerate(table_blocks, 1):
            # تحليل الجدول
            parsed = cls._parse_table(table_text)
            
            if not parsed:
                continue
            
            rows, headers = parsed
            
            # التحقق من الحد الأدنى
            if len(rows) < min_rows or (headers and len(headers) < min_cols):
                continue
            
            # استخراج العنوان والسياق
            caption, context = cls._extract_table_metadata(text, start_pos)
            
            tables.append({
                'global_number': idx,
                'text': table_text,
                'rows': rows,
                'headers': headers,
                'caption': caption,
                'context': context,
                'page_number': page_num,
                'num_rows': len(rows),
                'num_cols': len(headers) if headers else (len(rows[0]) if rows else 0),
                'confidence': cls._calculate_table_confidence(rows, headers)
            })
        
        logger.info(f"✅ استخرج {len(tables)} جدول من الصفحة {page_num}")
        
        return tables
    
    @classmethod
    def _find_table_blocks(cls, text: str) -> List[Tuple[str, int]]:
        """العثور على كتل الجداول في النص"""
        blocks = []
        lines = text.split('\n')
        
        in_table = False
        table_lines = []
        table_start = 0
        
        for idx, line in enumerate(lines):
            # كشف بداية جدول
            if cls._looks_like_table_row(line):
                if not in_table:
                    in_table = True
                    table_start = text.find(line)
                    table_lines = []
                table_lines.append(line)
            else:
                # نهاية جدول
                if in_table and table_lines:
                    table_text = '\n'.join(table_lines)
                    blocks.append((table_text, table_start))
                    in_table = False
                    table_lines = []
        
        # جدول في النهاية
        if table_lines:
            table_text = '\n'.join(table_lines)
            blocks.append((table_text, table_start))
        
        return blocks
    
    @classmethod
    def _looks_like_table_row(cls, line: str) -> bool:
        """التحقق من أن السطر يشبه صف جدول"""
        # يحتوي على فواصل أعمدة
        separators = line.count('|') + line.count('\t')
        
        if separators >= 2:
            return True
        
        # يحتوي على مسافات متعددة منتظمة (ASCII table)
        multi_spaces = len(re.findall(r'\s{3,}', line))
        if multi_spaces >= 2:
            return True
        
        return False
    
    @classmethod
    def _parse_table(cls, table_text: str) -> Optional[Tuple[List[List[str]], List[str]]]:
        """تحليل نص الجدول إلى صفوف وعناوين"""
        if not table_text:
            return None
        
        lines = [l.strip() for l in table_text.split('\n') if l.strip()]
        
        if len(lines) < 2:
            return None
        
        # محاولة تحليل markdown/pipe table
        if '|' in lines[0]:
            return cls._parse_pipe_table(lines)
        
        # محاولة تحليل ASCII table (مسافات)
        return cls._parse_space_table(lines)
    
    @classmethod
    def _parse_pipe_table(cls, lines: List[str]) -> Optional[Tuple[List[List[str]], List[str]]]:
        """تحليل جدول markdown (|)"""
        rows = []
        
        for line in lines:
            # تخطي خطوط الفصل (|---|---|)
            if re.match(r'^\s*\|?\s*[-:]+\s*\|', line):
                continue
            
            # تقسيم بواسطة |
            cells = [c.strip() for c in line.split('|') if c.strip()]
            
            if cells:
                rows.append(cells)
        
        if not rows or len(rows) < 2:
            return None
        
        # أول صف هو العناوين
        headers = rows[0]
        data_rows = rows[1:]
        
        return data_rows, headers
    
    @classmethod
    def _parse_space_table(cls, lines: List[str]) -> Optional[Tuple[List[List[str]], List[str]]]:
        """تحليل جدول ASCII (مسافات)"""
        rows = []
        
        for line in lines:
            # تقسيم بواسطة مسافات متعددة
            cells = re.split(r'\s{2,}', line.strip())
            cells = [c.strip() for c in cells if c.strip()]
            
            if cells and len(cells) >= 2:
                rows.append(cells)
        
        if not rows or len(rows) < 2:
            return None
        
        # أول صف هو العناوين
        headers = rows[0]
        data_rows = rows[1:]
        
        return data_rows, headers
    
    @classmethod
    def _extract_table_metadata(cls, full_text: str, table_pos: int) -> Tuple[str, str]:
        """استخراج عنوان وسياق الجدول"""
        # البحث عن "Table X: ..." قبل الجدول
        search_window = full_text[max(0, table_pos - 500):table_pos]
        
        caption_match = re.search(
            r'Table\s+(\d+)\s*[:.]?\s*([^\n]{0,200})',
            search_window,
            re.IGNORECASE
        )
        
        caption = caption_match.group(2).strip() if caption_match else ""
        
        # استخراج السياق (فقرة قبل الجدول)
        context_window = full_text[max(0, table_pos - 300):table_pos]
        context_lines = [l.strip() for l in context_window.split('\n') if l.strip()]
        context = ' '.join(context_lines[-2:]) if context_lines else ""
        
        return caption, context[:300]
    
    @classmethod
    def _calculate_table_confidence(cls, rows: List[List[str]], headers: List[str]) -> float:
        """حساب مستوى الثقة في الجدول"""
        score = 0.5
        
        # عدد الصفوف
        if len(rows) >= 5:
            score += 0.2
        elif len(rows) >= 3:
            score += 0.1
        
        # عدد الأعمدة
        if headers:
            score += 0.1
            if len(headers) >= 3:
                score += 0.1
        
        # تناسق عدد الأعمدة
        if rows:
            col_counts = [len(r) for r in rows]
            if col_counts and len(set(col_counts)) == 1:
                score += 0.1
        
        # وجود أرقام (دليل على بيانات فعلية)
        num_cells_with_numbers = 0
        for row in rows:
            for cell in row:
                if any(c.isdigit() for c in cell):
                    num_cells_with_numbers += 1
        
        if num_cells_with_numbers > len(rows):
            score += 0.1
        
        return min(score, 1.0)


if __name__ == "__main__":
    # اختبار
    logging.basicConfig(level=logging.INFO)
    
    test_text = """
    The loss function is defined as:
    
    L = - ∑ log p(y|x)  (1)
    
    where p_θ(y|x) is the conditional probability.
    
    Table 1: Results
    
    Model | Accuracy | F1-Score
    BERT  | 0.92     | 0.89
    GPT   | 0.88     | 0.85
    """
    
    eq_extractor = EnhancedEquationExtractor()
    equations = eq_extractor.extract_equations(test_text)
    
    print(f"\n✅ Found {len(equations)} equations:")
    for eq in equations:
        print(f"  - {eq['text'][:80]}")
    
    tbl_extractor = EnhancedTableExtractor()
    tables = tbl_extractor.extract_tables(test_text)
    
    print(f"\n✅ Found {len(tables)} tables:")
    for tbl in tables:
        print(f"  - {tbl['num_rows']}x{tbl['num_cols']} table")
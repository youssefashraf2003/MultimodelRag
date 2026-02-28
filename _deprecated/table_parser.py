"""
table_parser.py - Table Text Parser
====================================
✅ Parse table text (from PDF extraction) into structured rows/columns
✅ Auto-detect headers
✅ Handle misaligned columns
✅ Convert to pandas DataFrame for display
"""

import re
from typing import List, Tuple, Optional
import pandas as pd


def split_row_to_cols(row: str, min_spaces: int = 2) -> List[str]:
    """
    Split a table row into columns based on multiple spaces.
    
    Args:
        row: Single row of table text
        min_spaces: Minimum number of spaces to consider as column separator
        
    Returns:
        List of column values
    """
    # Split on 2+ spaces (typical PDF table column separator)
    cols = re.split(rf'\s{{{min_spaces},}}', row.strip())
    return [c.strip() for c in cols if c.strip()]


def remove_caption_lines(lines: List[str]) -> List[str]:
    """
    Remove table caption lines (e.g., "Table 2: Title").
    
    Args:
        lines: List of text lines
        
    Returns:
        Filtered lines without captions
    """
    filtered = []
    for line in lines:
        # Skip lines that look like table captions
        if re.search(r'\bTable\s+\d+\s*[:.]\s*', line, re.IGNORECASE):
            continue
        # Skip lines that are just numbers (page numbers, etc.)
        if re.match(r'^\s*\d+\s*$', line):
            continue
        filtered.append(line)
    return filtered


def parse_table_text(table_text: str, min_col_width: int = 2) -> List[List[str]]:
    """
    Parse table text into structured rows and columns.
    
    Args:
        table_text: Raw table text from PDF
        min_col_width: Minimum column width to keep
        
    Returns:
        List of rows, where each row is a list of column values
    """
    if not table_text:
        return []
    
    # Split into lines
    lines = [ln.rstrip() for ln in table_text.split('\n') if ln.strip()]
    
    # Remove caption lines
    lines = remove_caption_lines(lines)
    
    # Filter out very short lines (likely noise)
    lines = [ln for ln in lines if len(ln.strip()) > min_col_width]
    
    if not lines:
        return []
    
    # Convert each line to columns
    rows = []
    for line in lines:
        cols = split_row_to_cols(line)
        if cols:  # Only add non-empty rows
            rows.append(cols)
    
    return rows


def detect_header_row(rows: List[List[str]]) -> Tuple[Optional[List[str]], List[List[str]]]:
    """
    Auto-detect which row is the header.
    
    Strategy: 
    1. Find row with maximum columns
    2. Prefer row where next row has more numbers (indicating it's data)
    3. Allow headers with numbers/symbols (like "Top-1", "F1", "EM")
    
    Args:
        rows: List of parsed rows
        
    Returns:
        Tuple of (header_row, body_rows)
    """
    if not rows:
        return None, []
    
    # Find the row with maximum columns
    max_cols = max(len(r) for r in rows)
    
    # Find first row with max_cols as header candidate
    for i, row in enumerate(rows):
        if len(row) == max_cols:
            # Check quality of this header
            header_text = ' '.join(row)
            
            # Skip if it looks like garbage or has too many special chars
            if any(marker in header_text for marker in ['(cid:', '†', '‡', '§', '@']):
                continue
            
            # Skip if it's mostly email/URL
            if '@' in header_text or 'http' in header_text.lower():
                continue
            
            # Good signs for a header:
            # 1. Short cells (Model, Accuracy, F1, etc.)
            avg_cell_len = sum(len(c) for c in row) / max(len(row), 1)
            
            # 2. Next row (if exists) has more numbers = this is header
            is_followed_by_data = False
            if i + 1 < len(rows):
                next_row_text = ' '.join(rows[i + 1])
                num_digits = sum(c.isdigit() or c == '.' for c in next_row_text)
                is_followed_by_data = num_digits > len(next_row_text) * 0.2
            
            # Accept if:
            # - Average cell length is reasonable (< 30 chars)
            # - OR it's followed by numeric data
            if avg_cell_len < 30 or is_followed_by_data:
                return row, rows[i+1:]
    
    # Fallback: first row is header
    return rows[0], rows[1:]


def normalize_row_length(rows: List[List[str]], target_len: int) -> List[List[str]]:
    """
    Normalize all rows to the same length (for DataFrame compatibility).
    
    Args:
        rows: List of rows with varying lengths
        target_len: Target number of columns
        
    Returns:
        Normalized rows
    """
    normalized = []
    for row in rows:
        if len(row) < target_len:
            # Pad with empty strings
            row = row + [''] * (target_len - len(row))
        elif len(row) > target_len:
            # Merge excess columns into the last column
            row = row[:target_len-1] + [' '.join(row[target_len-1:])]
        normalized.append(row)
    return normalized


def table_text_to_dataframe(table_text: str) -> pd.DataFrame:
    """
    Convert raw table text to pandas DataFrame.
    
    Args:
        table_text: Raw table text from PDF
        
    Returns:
        pandas DataFrame
    """
    # Parse into rows
    rows = parse_table_text(table_text)
    
    if not rows:
        return pd.DataFrame()
    
    # Detect header
    header, body = detect_header_row(rows)
    
    if not header:
        return pd.DataFrame()
    
    # Normalize row lengths
    n_cols = len(header)
    body = normalize_row_length(body, n_cols)
    
    # Filter out completely empty rows
    body = [r for r in body if any(c.strip() for c in r)]
    
    if not body:
        return pd.DataFrame(columns=header)
    
    try:
        df = pd.DataFrame(body, columns=header)
        return df
    except Exception as e:
        # Fallback: return empty DataFrame with error message
        print(f"Error creating DataFrame: {e}")
        return pd.DataFrame()


def is_valid_table(table_text: str, min_rows: int = 3, min_cols: int = 2) -> bool:
    """
    Check if table text is valid (has enough rows and columns).
    ULTRA-STRICT validation to completely avoid author lists and garbage.
    
    Args:
        table_text: Raw table text
        min_rows: Minimum number of rows
        min_cols: Minimum number of columns
        
    Returns:
        True if valid table
    """
    if not table_text or len(table_text.strip()) < 20:
        return False
    
    rows = parse_table_text(table_text)
    
    if len(rows) < min_rows:
        return False
    
    # Check if any row has enough columns
    has_enough_cols = any(len(r) >= min_cols for r in rows)
    if not has_enough_cols:
        return False
    
    # CRITICAL 1: Check for paragraph indicators (NOT tables)
    text_lower = table_text.lower()
    paragraph_indicators = [
        'abstract', 'introduction', 'references', 'conclusion',
        'we present', 'we propose', 'this paper', 'our approach',
        'in this work', 'we show', 'we demonstrate', 'we introduce'
    ]
    if any(indicator in text_lower for indicator in paragraph_indicators):
        return False
    
    # CRITICAL 2: Check for email addresses (author affiliations, not tables)
    if '@' in table_text and table_text.count('@') > 1:
        return False
    
    # CRITICAL 3: Reject tables with too many special symbols (author markers)
    # Count special symbols like †, ‡, §, (cid:XX), ∗, ††, ‡‡
    special_symbols = ['†', '‡', '§', '∗', '(cid:', '††', '‡‡', '§§']
    symbol_count = sum(table_text.count(sym) for sym in special_symbols)
    
    # If more than 5 special symbols, likely author affiliations
    if symbol_count > 5:
        return False
    
    # CRITICAL 4: Reject if contains typical author patterns
    author_patterns = [
        r'[A-Z][a-z]+\s+[A-Z][a-z]+\s*[†‡§∗]',  # Name + symbol
        r'University|Institute|College|Research',  # Affiliations
        r'Department of|School of',
        r'[A-Z][a-z]+\s*,\s*[A-Z][a-z]+\s*,',  # Name, Name, pattern
    ]
    
    author_pattern_matches = sum(
        1 for pattern in author_patterns 
        if re.search(pattern, table_text)
    )
    
    if author_pattern_matches >= 2:
        return False
    
    # CRITICAL 5: Check column consistency (real tables have consistent column counts)
    col_counts = [len(r) for r in rows if len(r) > 0]
    if not col_counts:
        return False
    
    most_common_cols = max(set(col_counts), key=col_counts.count)
    consistency_ratio = col_counts.count(most_common_cols) / len(col_counts)
    
    # At least 70% of rows should have the same column count (stricter)
    if consistency_ratio < 0.7:
        return False
    
    # CRITICAL 6: Check average row length (tables have shorter rows than paragraphs)
    avg_row_len = sum(len(' '.join(r)) for r in rows) / max(len(rows), 1)
    if avg_row_len > 150:  # Stricter (was 200)
        return False
    
    # CRITICAL 7: Real data tables should have NUMERIC content
    all_text = ' '.join(' '.join(r) for r in rows)
    has_numbers = any(c.isdigit() for c in all_text)
    
    # Count ratio of digits
    digit_ratio = sum(1 for c in all_text if c.isdigit()) / max(len(all_text), 1)
    
    # Exception: if it's clearly a header-only table with structured keywords
    table_keywords = [
        'model', 'method', 'dataset', 'metric', 'score', 'accuracy', 
        'precision', 'recall', 'f1', 'performance', 'results', 'test',
        'train', 'validation', 'baseline', 'proposed'
    ]
    
    keyword_count = sum(
        1 for word in table_keywords
        if word in text_lower
    )
    
    # Must have numbers OR high keyword density
    if not has_numbers and keyword_count < 2:
        return False
    
    # If has numbers, check density (at least 2% digits)
    if has_numbers and digit_ratio < 0.02:
        return False
    
    # CRITICAL 8: Reject if ALL rows are names (author list pattern)
    name_pattern = re.compile(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+')
    rows_that_look_like_names = sum(
        1 for row in rows
        if any(name_pattern.match(cell) for cell in row)
    )
    
    # If more than 60% rows look like names, reject
    if rows_that_look_like_names / max(len(rows), 1) > 0.6:
        return False
    
    return True


# ─── Alternative: Column-based parsing (if space-based fails) ──────────────────

def parse_table_by_columns(table_text: str, col_positions: Optional[List[int]] = None) -> List[List[str]]:
    """
    Parse table by fixed column positions (if alignment is very strict).
    
    Args:
        table_text: Raw table text
        col_positions: List of character positions where columns start
        
    Returns:
        List of rows
    """
    if not table_text:
        return []
    
    lines = [ln for ln in table_text.split('\n') if ln.strip()]
    
    if not col_positions:
        # Auto-detect column positions by finding vertical alignment
        col_positions = detect_column_positions(lines)
    
    rows = []
    for line in lines:
        cols = []
        for i in range(len(col_positions)):
            start = col_positions[i]
            end = col_positions[i+1] if i+1 < len(col_positions) else len(line)
            col_text = line[start:end].strip()
            cols.append(col_text)
        if any(c for c in cols):  # At least one non-empty cell
            rows.append(cols)
    
    return rows


def detect_column_positions(lines: List[str], min_align: int = 3) -> List[int]:
    """
    Detect column positions by finding vertical alignment of whitespace.
    
    Args:
        lines: List of table lines
        min_align: Minimum number of lines that must align
        
    Returns:
        List of column start positions
    """
    if not lines:
        return [0]
    
    max_len = max(len(ln) for ln in lines)
    
    # Count spaces at each position across all lines
    space_counts = [0] * max_len
    for line in lines:
        for i, char in enumerate(line):
            if char == ' ':
                space_counts[i] += 1
    
    # Find positions where many lines have spaces (likely column boundaries)
    positions = [0]  # Start at position 0
    for i in range(1, max_len):
        if space_counts[i] >= min_align and space_counts[i-1] < min_align:
            positions.append(i)
    
    return positions


if __name__ == "__main__":
    # Test case
    test_table = """
    Table 2: Test Scores
    
    Model          Accuracy  F1-Score  Latency
    BERT           0.92      0.89      120ms
    GPT-2          0.88      0.85      95ms
    RoBERTa        0.94      0.91      130ms
    """
    
    print("Testing table_parser.py:")
    print("=" * 60)
    
    df = table_text_to_dataframe(test_table)
    print("\nParsed DataFrame:")
    print(df)
    print(f"\nShape: {df.shape}")
    print(f"Valid table: {is_valid_table(test_table)}")
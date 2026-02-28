"""
smart_retriever.py - Enhanced Smart Retrieval System V3.0 (FIXED IMPORTS)
=========================================================================
✅ All required exports for enhanced_rag_system.py
✅ Fixed import structure
✅ Compatible with existing code
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  RAG TOKEN HELPERS
# ═══════════════════════════════════════════════════════════════════════════

RAG_TOKEN_TERMS = [
    "rag token", "rag-token", "p_rag-token", "p rag-token",
    "p_rag_token", "ragtoken", "top-k", "top k", "p_eta", "p_theta",
    "pη", "pθ"
]


def _contains_rag_token_query(q: str) -> bool:
    """Check if query is about RAG token"""
    q = (q or "").lower()
    return (
        ("rag token" in q)
        or ("rag-token" in q)
        or ("p_rag-token" in q)
        or ("ragtoken" in q)
    )


def _keyword_boost_score(text: str) -> float:
    """Boost score for RAG-token-related terms"""
    if not text:
        return 0.0
    t = text.lower()
    score = 0.0
    for term in RAG_TOKEN_TERMS:
        if term in t:
            score += 1.0
    # Bigger boost for explicit RAG token
    if ("p_rag-token" in t) or ("rag-token(y|x" in t) or ("rag-token (y|x" in t):
        score += 5.0
    return score


# ═══════════════════════════════════════════════════════════════════════════
#  QUERY TYPES
# ═══════════════════════════════════════════════════════════════════════════

class QueryType(Enum):
    """Query types for classification"""
    EQUATION = "equation"
    TABLE = "table"
    FIGURE = "figure"
    GENERAL = "general"
    SPECIFIC_ELEMENT = "specific_element"
    LIST_ALL = "list_all"
    COMPARISON = "comparison"
    CROSS_REFERENCE = "cross_reference"


@dataclass
class QueryIntent:
    """Structured query intent"""
    query_type: QueryType
    target_type: Optional[str] = None  # equation/table/figure
    target_number: Optional[int] = None
    target_numbers: Optional[List[int]] = None
    keywords: List[str] = field(default_factory=list)
    requires_context: bool = True
    confidence: float = 1.0
    search_strategy: str = "hybrid"
    
    def __str__(self):
        return f"QueryIntent(type={self.query_type.value}, target={self.target_type}#{self.target_number})"


# ═══════════════════════════════════════════════════════════════════════════
#  QUERY CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

class QueryClassifier:
    """High-accuracy query classifier"""
    
    # Detection patterns
    EQUATION_PATTERNS = [
        r'\bequation\s+(\d+)\b',
        r'\beq\.?\s+(\d+)\b',
        r'\bformula\s+(\d+)\b',
        r'explain.*equation\s+(\d+)',
        r'show.*equation\s+(\d+)',
        r'what.*equation\s+(\d+)',
    ]
    
    TABLE_PATTERNS = [
        r'\btable\s+(\d+)\b',
        r'\btbl\.?\s+(\d+)\b',
        r'show.*table\s+(\d+)',
        r'display.*table\s+(\d+)',
    ]
    
    FIGURE_PATTERNS = [
        r'\bfigure\s+(\d+)\b',
        r'\bfig\.?\s+(\d+)\b',
        r'show.*figure\s+(\d+)',
    ]
    
    LIST_ALL_PATTERNS = [
        r'(?:show|list|display)\s+all\s+(equation|table|figure)s?',
        r'how many\s+(equation|table|figure)s?',
        r'all\s+(?:the\s+)?(equation|table|figure)s?',
    ]
    
    COMPARISON_PATTERNS = [
        r'compare\s+(equation|table|figure)s?\s+(\d+)\s+(?:and|with)\s+(\d+)',
        r'difference\s+between\s+(equation|table|figure)s?\s+(\d+)\s+and\s+(\d+)',
    ]
    
    EQUATION_KEYWORDS = [
        'equation', 'formula', 'mathematical', 'calculation', 'expression',
        'variable', 'function', 'compute', 'solve'
    ]
    
    TABLE_KEYWORDS = [
        'table', 'data', 'results', 'statistics', 'values', 'columns',
        'rows', 'dataset', 'comparison'
    ]
    
    FIGURE_KEYWORDS = [
        'figure', 'image', 'diagram', 'graph', 'plot', 'chart',
        'illustration', 'visualization', 'picture'
    ]
    
    def classify(self, query: str) -> QueryIntent:
        """
        Classify query intent
        
        Args:
            query: User query string
            
        Returns:
            QueryIntent object
        """
        if not query:
            return QueryIntent(
                query_type=QueryType.GENERAL,
                confidence=0.5,
                search_strategy="hybrid"
            )
        
        query_lower = query.lower().strip()
        
        # 1. Check for cross-reference queries
        cross_ref_patterns = [
            r'relation.*between',
            r'how.*relate',
            r'connection.*between',
            r'link.*between'
        ]
        
        if any(re.search(p, query_lower) for p in cross_ref_patterns):
            return QueryIntent(
                query_type=QueryType.CROSS_REFERENCE,
                keywords=query.split(),
                confidence=0.9,
                search_strategy="hybrid"
            )
        
        # 2. Check for comparison queries
        for pattern in self.COMPARISON_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                groups = match.groups()
                return QueryIntent(
                    query_type=QueryType.COMPARISON,
                    target_type=groups[0],
                    target_numbers=[int(groups[1]), int(groups[2])],
                    keywords=query.split(),
                    confidence=0.95,
                    search_strategy="hybrid"
                )
        
        # 3. Check for "show all" queries
        for pattern in self.LIST_ALL_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                element_type = match.group(1).rstrip('s')
                return QueryIntent(
                    query_type=QueryType.LIST_ALL,
                    target_type=element_type,
                    requires_context=False,
                    confidence=1.0,
                    search_strategy="dense"
                )
        
        # 4. Check for specific element queries
        # Equations
        for pattern in self.EQUATION_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                number = int(match.group(1))
                return QueryIntent(
                    query_type=QueryType.SPECIFIC_ELEMENT,
                    target_type='equation',
                    target_number=number,
                    keywords=query.split(),
                    confidence=1.0,
                    search_strategy="dense"
                )
        
        # Tables
        for pattern in self.TABLE_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                number = int(match.group(1))
                return QueryIntent(
                    query_type=QueryType.SPECIFIC_ELEMENT,
                    target_type='table',
                    target_number=number,
                    keywords=query.split(),
                    confidence=1.0,
                    search_strategy="dense"
                )
        
        # Figures
        for pattern in self.FIGURE_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                number = int(match.group(1))
                return QueryIntent(
                    query_type=QueryType.SPECIFIC_ELEMENT,
                    target_type='figure',
                    target_number=number,
                    keywords=query.split(),
                    confidence=1.0,
                    search_strategy="dense"
                )
        
        # 5. Classify by keywords
        return self._classify_by_keywords(query_lower)
    
    def _classify_by_keywords(self, query: str) -> QueryIntent:
        """Classify by keyword matching"""
        
        equation_score = sum(1 for kw in self.EQUATION_KEYWORDS if kw in query)
        table_score = sum(1 for kw in self.TABLE_KEYWORDS if kw in query)
        figure_score = sum(1 for kw in self.FIGURE_KEYWORDS if kw in query)
        
        max_score = max(equation_score, table_score, figure_score)
        
        if max_score == 0:
            return QueryIntent(
                query_type=QueryType.GENERAL,
                keywords=query.split(),
                requires_context=True,
                confidence=0.7,
                search_strategy="hybrid"
            )
        
        if equation_score == max_score:
            return QueryIntent(
                query_type=QueryType.EQUATION,
                target_type='equation',
                keywords=query.split(),
                requires_context=True,
                confidence=0.8,
                search_strategy="hybrid"
            )
        elif table_score == max_score:
            return QueryIntent(
                query_type=QueryType.TABLE,
                target_type='table',
                keywords=query.split(),
                requires_context=True,
                confidence=0.8,
                search_strategy="hybrid"
            )
        else:
            return QueryIntent(
                query_type=QueryType.FIGURE,
                target_type='figure',
                keywords=query.split(),
                requires_context=True,
                confidence=0.8,
                search_strategy="hybrid"
            )


# ═══════════════════════════════════════════════════════════════════════════
#  SMART RETRIEVER
# ═══════════════════════════════════════════════════════════════════════════

class SmartRetriever:
    """
    Enhanced smart retrieval system
    """
    
    def __init__(self, vector_store: Any):
        """
        Initialize SmartRetriever
        
        Args:
            vector_store: Vector store instance
        """
        self.vector_store = vector_store
        self.classifier = QueryClassifier()
        
        logger.info("✅ SmartRetriever initialized")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        enable_self_rag: bool = True,
        use_hybrid: bool = True,
    ) -> Dict[str, Any]:
        """
        Smart retrieval with classification
        
        Args:
            query: User query
            top_k: Number of results
            enable_self_rag: Enable validation (unused, for compatibility)
            use_hybrid: Use hybrid search
            
        Returns:
            Dictionary with chunks, scores, and metadata
        """
        
        # 1. Classify query
        intent = self.classifier.classify(query)
        logger.info(f"Query classified: {intent}")
        
        # 2. Determine search strategy
        search_type = "hybrid" if use_hybrid and intent.search_strategy == "hybrid" else "dense"
        
        # 3. Handle specific element queries (O(1) registry lookup)
        if intent.query_type == QueryType.SPECIFIC_ELEMENT and intent.target_number:
            chunks = self._retrieve_specific_element(
                intent.target_type,
                intent.target_number,
                top_k
            )
            
            if chunks:
                # Convert to expected format
                results = []
                for chunk in chunks:
                    results.append({
                        'chunk': chunk,
                        'score': 1.0,  # Exact match
                        'rank': len(results)
                    })
                
                return {
                    'chunks': [r['chunk'] for r in results],
                    'scores': [r['score'] for r in results],
                    'intent': intent,
                    'strategy': 'registry_lookup',
                    'success': True
                }
        
        # 4. Vector search
        try:
            # Try to use the search method
            if hasattr(self.vector_store, 'search'):
                results = self.vector_store.search(
                    query=query,
                    top_k=top_k,
                    search_type=search_type
                )
            else:
                # Fallback for older vector store API
                results = self.vector_store.similarity_search(query, k=top_k)
            
            # 5. Apply type filtering if needed
            if intent.target_type:
                results = self._filter_by_type(results, intent.target_type)
            
            # 6. Apply RAG-token boosting if needed
            if _contains_rag_token_query(query):
                results = self._apply_rag_token_boost(results)
            
            # Extract chunks and scores
            chunks = []
            scores = []
            
            for r in results:
                if isinstance(r, dict):
                    chunks.append(r.get('chunk'))
                    scores.append(r.get('score', 0.0))
                elif hasattr(r, 'chunk'):
                    chunks.append(r.chunk)
                    scores.append(r.similarity_score if hasattr(r, 'similarity_score') else 0.0)
                else:
                    chunks.append(r)
                    scores.append(0.5)
            
            return {
                'chunks': chunks,
                'scores': scores,
                'intent': intent,
                'strategy': search_type,
                'success': len(chunks) > 0
            }
        
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return {
                'chunks': [],
                'scores': [],
                'intent': intent,
                'strategy': search_type,
                'success': False,
                'error': str(e)
            }
    
    def _retrieve_specific_element(
        self,
        element_type: str,
        element_number: int,
        top_k: int
    ) -> List[Any]:
        """Retrieve specific element by registry lookup"""
        
        # Try registry lookup if available
        if hasattr(self.vector_store, 'registry'):
            chunk_id = self.vector_store.registry.lookup(
                element_type=element_type,
                number=element_number
            )
            
            if chunk_id and hasattr(self.vector_store, 'get_chunk_by_id'):
                chunk = self.vector_store.get_chunk_by_id(chunk_id)
                if chunk:
                    return [chunk]
        
        # Fallback: search with specific query
        query = f"{element_type} {element_number}"
        try:
            results = self.vector_store.search(query, top_k=top_k, search_type="dense")
            
            # Filter to exact match
            for r in results:
                chunk = r.chunk if hasattr(r, 'chunk') else r
                if (chunk.chunk_type == element_type and 
                    chunk.metadata.get('global_number') == element_number):
                    return [chunk]
        except:
            pass
        
        return []
    
    def _filter_by_type(self, results: List[Any], target_type: str) -> List[Any]:
        """Filter results by chunk type"""
        
        filtered = []
        for r in results:
            chunk = r.chunk if hasattr(r, 'chunk') else r
            if chunk.chunk_type == target_type:
                filtered.append(r)
        
        return filtered
    
    def _apply_rag_token_boost(self, results: List[Any]) -> List[Any]:
        """Apply semantic boosting for RAG-token queries"""
        
        boosted = []
        
        for r in results:
            chunk = r.chunk if hasattr(r, 'chunk') else r
            score = r.similarity_score if hasattr(r, 'similarity_score') else 0.5
            
            # Calculate boost
            boost = _keyword_boost_score(chunk.text) * 0.1
            new_score = min(1.0, score + boost)
            
            if hasattr(r, 'similarity_score'):
                r.similarity_score = new_score
            
            boosted.append(r)
        
        # Re-sort by score
        boosted.sort(key=lambda x: x.similarity_score if hasattr(x, 'similarity_score') else 0.5, reverse=True)
        
        return boosted


# ═══════════════════════════════════════════════════════════════════════════
#  EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    'SmartRetriever',
    'QueryClassifier',
    'QueryType',
    'QueryIntent',
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test classification
    classifier = QueryClassifier()
    
    test_queries = [
        "Show me equation 3",
        "What is the RAG token equation?",
        "List all tables",
        "Compare equation 1 and equation 2",
    ]
    
    for query in test_queries:
        intent = classifier.classify(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {intent}")
    
    print("\n✅ smart_retriever.py ready")
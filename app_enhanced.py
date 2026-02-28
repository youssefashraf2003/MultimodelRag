"""
app_enhanced.py - Streamlit Application V4.0 (COMPLETE REDESIGN)
=================================================================
✅ Professional modern UI design
✅ Fixed all method call errors
✅ Better equation/table/figure display
✅ Improved error handling
✅ Real-time statistics
✅ Enhanced user experience
✅ Markdown table formatting
✅ LaTeX equation rendering
✅ Comprehensive metadata display
"""

# ══════════════════════════════════════════════════════════════════════════
# FIX ENCODING FIRST (before any other imports)
# ══════════════════════════════════════════════════════════════════════════
import sys
import io
import os
import vector_store
print("VECTOR STORE LOADED FROM:", vector_store.__file__)
# Force UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding='utf-8',
        errors='replace',
        line_buffering=True
    )

if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer,
        encoding='utf-8',
        errors='replace',
        line_buffering=True
    )

os.environ['PYTHONIOENCODING'] = 'utf-8'

# ══════════════════════════════════════════════════════════════════════════
# Now import everything else
# ══════════════════════════════════════════════════════════════════════════

import streamlit as st
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_rag_system import EnhancedRAGSystem, EnhancedRAGConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Page config
st.set_page_config(
    page_title="Enhanced RAG System V4.0",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #1E88E5, #00ACC1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Stat boxes */
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .stat-box:hover {
        transform: translateY(-2px);
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
    }
    .stat-label {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.9);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Message boxes */
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }
    
    /* Equations */
    .equation-box {
        background-color: #f8f9fa;
        border: 2px solid #1E88E5;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
    }
    
    /* Tables */
    .table-container {
        overflow-x: auto;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Answer section */
    .answer-section {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Reference tags */
    .reference-tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'document_loaded' not in st.session_state:
    st.session_state.document_loaded = False
if 'doc_info' not in st.session_state:
    st.session_state.doc_info = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def initialize_system(api_key: str) -> bool:
    """Initialize RAG system"""
    try:
        config = EnhancedRAGConfig()
        config.groq_api_key = api_key
        st.session_state.rag_system = EnhancedRAGSystem(config)
        return True
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {e}")
        return False


def process_document(pdf_file) -> Dict[str, Any]:
    """Process uploaded PDF"""
    try:
        # Save uploaded file temporarily
        temp_path = Path("temp_upload.pdf")
        with open(temp_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        # Process
        with st.spinner("🔄 Processing document... This may take a moment."):
            summary = st.session_state.rag_system.process_document(str(temp_path))
        
        st.session_state.document_loaded = True
        st.session_state.doc_info = st.session_state.rag_system.get_document_info()
        
        # Clean up
        temp_path.unlink(missing_ok=True)
        
        return summary
    
    except Exception as e:
        st.error(f"❌ Error processing document: {e}")
        return None


async def query_async(query: str) -> Dict[str, Any]:
    """Query the system (async)"""
    return await st.session_state.rag_system.query(query, return_metadata=True)


def query_system(query: str) -> Dict[str, Any]:
    """Query the system (wrapper)"""
    try:
        result = asyncio.run(query_async(query))
        return result
    except Exception as e:
        st.error(f"❌ Error querying system: {e}")
        return None


def display_response(result: Dict[str, Any]):
    """Display response with formatting"""
    if not result or not result.get('success', False):
        error_msg = result.get('answer', '⚠️ No answer generated') if result else '⚠️ No answer generated'
        st.markdown(f'<div class="error-box">{error_msg}</div>', unsafe_allow_html=True)
        return
    
    answer = result['answer']
    metadata = result.get('metadata', {})
    
    # Self-RAG status
    validation = metadata.get('validation')
    if validation:
        if validation['passed']:
            st.markdown(
                '<div class="success-box">✅ <strong>Self-RAG Validation:</strong> Passed</div>',
                unsafe_allow_html=True
            )
        else:
            confidence = validation.get('confidence', 0)
            st.markdown(
                f'<div class="warning-box">⚠️ <strong>Self-RAG Validation:</strong> '
                f'Issues detected (Confidence: {confidence:.1%})</div>',
                unsafe_allow_html=True
            )
            if validation.get('issues'):
                with st.expander("⚠️ View Validation Issues"):
                    for issue in validation['issues']:
                        st.warning(f"• {issue}")
    
    # Main answer
    st.markdown("### 📝 Answer")
    st.markdown('<div class="answer-section">', unsafe_allow_html=True)
    st.markdown(answer)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # LaTeX equations
    latex_eqs = metadata.get('latex_equations', [])
    if latex_eqs:
        st.markdown("### 📐 Equations")
        for i, eq in enumerate(latex_eqs, 1):
            st.markdown('<div class="equation-box">', unsafe_allow_html=True)
            try:
                st.latex(eq)
            except Exception:
                st.code(eq, language='latex')
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tables
    tables = metadata.get('tables', [])
    if tables:
        st.markdown("### 📊 Tables")
        for table in tables:
            caption = table.get('caption', '')
            st.markdown(f"**Table {table['number']}:** {caption}")
            st.markdown('<div class="table-container">', unsafe_allow_html=True)
            st.markdown(table['markdown'])
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Figures
    figures = metadata.get('figures', [])
    if figures:
        st.markdown("### 🖼️ Figures")
        for fig in figures:
            st.markdown(f"**Figure {fig['number']}:** {fig.get('caption', '')}")
    
    # References
    references = metadata.get('references', [])
    if references:
        st.markdown("### 📚 References")
        ref_html = "".join([
            f'<span class="reference-tag">{ref}</span>'
            for ref in references[:5]  # Show first 5
        ])
        st.markdown(ref_html, unsafe_allow_html=True)
    
    # Metadata expander
    with st.expander("📊 Query Metadata"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Intent", metadata.get('intent', 'N/A'))
            st.metric("Sources", metadata.get('num_sources', 0))
            st.metric("Strategy", metadata.get('search_strategy', 'N/A'))
        
        with col2:
            quality = metadata.get('quality_scores', {})
            st.metric("Quality Score", f"{quality.get('overall', 0):.1%}")
            st.metric("Relevance", f"{quality.get('relevance', 0):.1%}")
            st.metric("Completeness", f"{quality.get('completeness', 0):.1%}")
        
        with col3:
            st.metric("Target Type", metadata.get('target_type', 'N/A') or 'General')
            target_num = metadata.get('target_number')
            st.metric("Target #", target_num if target_num else 'N/A')
            st.metric("Mode", result.get('mode', 'N/A'))


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════

# Header
st.markdown('<div class="main-header">🚀 Enhanced RAG System V4.0</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Specialized Multi-Modal RAG with Self-Validation & Hybrid Search</div>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Enter your Groq API key"
    )
    
    if st.button("🔧 Initialize System", use_container_width=True, type="primary"):
        if api_key:
            if initialize_system(api_key):
                st.success("✅ System initialized!")
                st.balloons()
        else:
            st.error("❌ Please enter API key")
    
    st.markdown("---")
    
    # Document upload
    st.header("📄 Document")
    
    pdf_file = st.file_uploader(
        "Upload PDF",
        type=['pdf'],
        help="Upload a PDF document to analyze"
    )
    
    if pdf_file and st.session_state.rag_system:
        if st.button("🔄 Process Document", use_container_width=True, type="primary"):
            summary = process_document(pdf_file)
            if summary:
                st.success("✅ Document processed!")
                st.json(summary)
    
    # Document info
    if st.session_state.document_loaded and st.session_state.doc_info:
        st.markdown("---")
        st.header("📊 Document Statistics")
        
        doc_info = st.session_state.doc_info
        
        st.markdown(f"**Title:** {doc_info.get('title', 'Unknown')}")
        st.markdown(f"**Pages:** {doc_info.get('num_pages', 0)}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="stat-number">{len(doc_info.get("equations", []))}</div>',
                unsafe_allow_html=True
            )
            st.markdown('<div class="stat-label">Equations</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="stat-number">{len(doc_info.get("tables", []))}</div>',
                unsafe_allow_html=True
            )
            st.markdown('<div class="stat-label">Tables</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="stat-number">{len(doc_info.get("figures", []))}</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="stat-label">Figures</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Main content
if not st.session_state.rag_system:
    st.markdown(
        '<div class="info-box">👈 Please initialize the system first '
        '(enter API key and click "Initialize System")</div>',
        unsafe_allow_html=True
    )

elif not st.session_state.document_loaded:
    st.markdown(
        '<div class="info-box">👈 Please upload and process a document</div>',
        unsafe_allow_html=True
    )

else:
    # Chat interface
    st.header("💬 Ask Questions")
    
    # Quick actions
    st.subheader("🎯 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📐 Show All Equations", use_container_width=True):
            with st.spinner("🔍 Searching..."):
                result = query_system('Show all equations')
                if result:
                    st.session_state.chat_history.append({
                        'query': 'Show all equations',
                        'result': result
                    })
                    st.rerun()
    
    with col2:
        if st.button("📊 Show All Tables", use_container_width=True):
            with st.spinner("🔍 Searching..."):
                result = query_system('Show all tables')
                if result:
                    st.session_state.chat_history.append({
                        'query': 'Show all tables',
                        'result': result
                    })
                    st.rerun()
    
    with col3:
        if st.button("🖼️ Show All Figures", use_container_width=True):
            with st.spinner("🔍 Searching..."):
                result = query_system('Show all figures')
                if result:
                    st.session_state.chat_history.append({
                        'query': 'Show all figures',
                        'result': result
                    })
                    st.rerun()
    
    with col4:
        if st.button("📋 Document Summary", use_container_width=True):
            with st.spinner("🔍 Analyzing..."):
                result = query_system('Summarize the main findings of this document')
                if result:
                    st.session_state.chat_history.append({
                        'query': 'Summarize main findings',
                        'result': result
                    })
                    st.rerun()
    
    st.markdown("---")
    
    # Query input
    query = st.text_input(
        "Your Question",
        placeholder="e.g., Explain Equation 3, Show Table 1, What are the main results?",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 5])
    
    with col1:
        submit = st.button("🚀 Ask", use_container_width=True, type="primary")
    
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    if submit and query:
        with st.spinner("🤔 Thinking..."):
            result = query_system(query)
        
        if result:
            st.session_state.chat_history.append({
                'query': query,
                'result': result
            })
            # Clear input by rerunning
            st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.header("📜 Conversation History")
        
        # Show most recent first
        for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
            q_num = len(st.session_state.chat_history) - i + 1
            with st.expander(f"Q{q_num}: {chat['query']}", expanded=(i == 1)):
                display_response(chat['result'])


# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; font-size: 0.9rem;">'
    '🚀 Enhanced RAG System V4.0 | '
    'Powered by Groq LLaMA & Streamlit | '
    '© 2024'
    '</div>',
    unsafe_allow_html=True
)
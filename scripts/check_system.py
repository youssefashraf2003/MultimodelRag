"""
check_system.py - System Diagnostic Script v2
==============================================
Checks all required system components correctly.
Updated to match actual project structure.
"""

import sys
import os


def check_python():
    print("=" * 60)
    print("Python Version Check")
    print("=" * 60)
    print(f"Version: {sys.version}")
    print(f"Path: {sys.executable}")

    if sys.version_info >= (3, 10):
        print("✅ Python version OK (3.10+)")
        return True
    else:
        print("❌ Python version too old. Need 3.10+")
        return False


def check_files():
    print("\n" + "=" * 60)
    print("File Check")
    print("=" * 60)

    required_files = {
        'multimodal_agentic_rag.py': 'Main RAG system',
        'app_multimodal_rag.py':     'Streamlit app',
        'config.py':      'Configuration',
        'pdf_processor.py':          'PDF processing',
        'vector_store.py':           'Vector store',
        'hybrid_retrieval.py':       'Hybrid retrieval',
        'response_formatter.py':     'Response formatter',
        'web_search.py':             'Web search',
        'chat_history.py':           'Chat history',
        'rate_limiter.py':           'Rate limiter',
        'document_chunker.py':       'Document chunker',
        'requirements.txt':          'Requirements',
    }

    all_good = True
    for file, description in required_files.items():
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file:35s} ({size:,} bytes) - {description}")
        else:
            print(f"❌ {file:35s} - MISSING! - {description}")
            all_good = False

    return all_good


def check_imports():
    print("\n" + "=" * 60)
    print("Package Import Check")
    print("=" * 60)

    # Core required
    required = [
        ('groq',                   'Groq API client'),
        ('streamlit',              'Streamlit UI'),
        ('sentence_transformers',  'Sentence Transformers'),
        ('fitz',                   'PyMuPDF — PDF processing'),
        ('PIL',                    'Pillow — image handling'),
        ('dotenv',                 'python-dotenv'),
    ]

    # Important but can degrade gracefully
    optional = [
        ('faiss',           'FAISS vector store'),
        ('pdfplumber',      'pdfplumber — better table extraction'),
        ('rank_bm25',       'BM25 sparse retrieval'),
        ('wikipedia',       'Wikipedia search'),
        ('arxiv',           'ArXiv search'),
        ('duckduckgo_search','DuckDuckGo search'),
        ('langchain',       'LangChain'),
        ('langchain_community', 'LangChain Community'),
    ]

    all_required_ok = True
    for package, description in required:
        try:
            __import__(package)
            print(f"✅ {package:30s} - {description}")
        except ImportError:
            print(f"❌ {package:30s} - MISSING (required) - {description}")
            all_required_ok = False

    print("\n--- Optional (degrade gracefully) ---")
    for package, description in optional:
        try:
            __import__(package)
            print(f"✅ {package:30s} - {description}")
        except ImportError:
            print(f"⚠️  {package:30s} - not installed (optional) - {description}")

    return all_required_ok


def check_config():
    print("\n" + "=" * 60)
    print("Configuration Check")
    print("=" * 60)

    try:
        # Import config as module (not as a class)
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", "config.py")
        if spec is None:
            print("❌ config.py not found")
            return False

        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        api_key = getattr(config_module, 'GROQ_API_KEY', '') or ''
        if api_key and len(api_key) > 10 and 'your-' not in api_key.lower():
            print(f"✅ GROQ_API_KEY configured: {api_key[:10]}...{api_key[-4:]}")
            key_ok = True
        else:
            print("⚠️  GROQ_API_KEY not configured — set GROQ_API_KEY env var or edit config")
            key_ok = False

        models = getattr(config_module, 'GROQ_MODELS', {})
        if models:
            print(f"✅ Models configured: {list(models.keys())}")
        else:
            print("⚠️  No GROQ_MODELS found in config")

        return key_ok

    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False


def check_main_import():
    print("\n" + "=" * 60)
    print("Main System Import Check")
    print("=" * 60)

    if not os.path.exists('multimodal_agentic_rag.py'):
        print("❌ multimodal_agentic_rag.py not found")
        return False

    try:
        sys.path.insert(0, os.getcwd())
        import multimodal_agentic_rag
        print("✅ multimodal_agentic_rag module imported successfully")

        # Check key classes exist
        classes_to_check = ['MultiModalAgenticRAG', 'MultiModalRAGConfig']
        for cls_name in classes_to_check:
            if hasattr(multimodal_agentic_rag, cls_name):
                print(f"✅ {cls_name} found")
            else:
                print(f"⚠️  {cls_name} not found in module")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_groq_api():
    print("\n" + "=" * 60)
    print("Groq API Connection Check")
    print("=" * 60)

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", "config.py")
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        api_key = getattr(config_module, 'GROQ_API_KEY', '') or os.getenv('GROQ_API_KEY', '')
    except Exception:
        api_key = os.getenv('GROQ_API_KEY', '')

    if not api_key or len(api_key) < 10:
        print("⚠️  No API key — skipping connection test")
        return True  # Not a failure, just skip

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        # Minimal test call
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        print("✅ Groq API connection successful")
        return True
    except Exception as e:
        err_str = str(e)
        if "401" in err_str or "invalid_api_key" in err_str.lower():
            print(f"❌ Invalid API key: {e}")
            return False
        elif "rate" in err_str.lower():
            print(f"⚠️  Rate limit hit (API key is valid): {e}")
            return True
        else:
            print(f"⚠️  API check failed (may still work): {e}")
            return True  # Network errors aren't config failures


def main():
    print("\n" + "🔍 Multi-Modal RAG System Diagnostic v2".center(60))
    print("\n")

    results = {
        'Python Version':    check_python(),
        'Required Files':    check_files(),
        'Package Imports':   check_imports(),
        'Configuration':     check_config(),
        'Main System Import':check_main_import(),
        'Groq API':          check_groq_api(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED! System ready to run.")
        print("\nNext steps:")
        print("1. Run: streamlit run app_multimodal_rag.py")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"⚠️  {len(failed)} check(s) failed: {', '.join(failed)}")
        print("\nTroubleshooting:")
        print("1. Install packages: pip install -r requirements.txt")
        print("2. Set API key: export GROQ_API_KEY=gsk_...")
        print("3. Check Python version: python --version (need 3.10+)")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
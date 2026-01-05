#!/usr/bin/env python3
"""
Build Temporal GraphRAG knowledge graph from documents.

This script:
1. Loads documents from various sources (ECT-QA corpus, text files, or directories)
2. Creates TemporalGraphRAG from config.yaml (uses tgrag.create_temporal_graphrag_from_config)
3. Builds the temporal knowledge graph
4. Saves everything to the output directory

Usage:
    # Set API keys (provider-specific)
    export OPENAI_API_KEY="your-key-here"      # For OpenAI provider
    export GEMINI_API_KEY="your-key-here"      # For Gemini provider
    # etc.
    
    # Run with default config (from tgrag/configs/config.yaml)
    python build_graph.py --output_dir ./graph_output --num_docs 3
    
    # Build from a single text file
    python build_graph.py --output_dir ./graph_output --corpus_path ./my_document.txt
    
    # Build from a directory of text files
    python build_graph.py --output_dir ./graph_output --corpus_path ./my_documents/
    
    # Override config values
    python build_graph.py --output_dir ./graph_output --num_docs 3 --chunk_size 1000
"""

import os
import sys
import json
import gzip
import argparse
import logging
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

# Configure logging - default to ERROR to reduce noise, but allow DEBUG via environment variable
debug_mode = os.getenv("TG_RAG_DEBUG", "false").lower() == "true"
log_level = logging.DEBUG if debug_mode else logging.ERROR

logging.basicConfig(
    level=log_level,
    format='%(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if debug_mode:
    print("🔍 Debug mode enabled - verbose logging active")

# Load environment variables from .env file if present
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from tgrag package (simplified API)
from tgrag import create_temporal_graphrag_from_config


def load_documents_from_corpus(corpus_path: Path, num_docs: int = 3) -> List[Dict]:
    """
    Load documents from the ECT-QA corpus.
    
    Args:
        corpus_path: Path to the corpus file (base.jsonl.gz)
        num_docs: Number of documents to load
        
    Returns:
        List of document dictionaries
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    documents = []
    try:
        with gzip.open(corpus_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_docs:
                    break
                doc = json.loads(line)
                documents.append(doc)
        print(f"✅ Loaded {len(documents)} documents from corpus")
        return documents
    except Exception as e:
        raise RuntimeError(f"Error loading corpus: {e}")


def load_documents_from_txt_file(txt_path: Path) -> List[Dict]:
    """
    Load a single text file as a document.
    Supports common text formats: .txt, .md, .rst, .text, .log, and files without extensions.
    
    Args:
        txt_path: Path to the text file
        
    Returns:
        List containing a single document dictionary
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"Text file not found: {txt_path}")
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            print(f"⚠️  Warning: File {txt_path} is empty, skipping")
            return []
        
        # Use filename (without extension) as title
        title = txt_path.stem if txt_path.suffix else txt_path.name
        
        return [{
            'title': title,
            'doc': content
        }]
    except UnicodeDecodeError:
        raise RuntimeError(f"File {txt_path} is not a valid text file (binary?)")
    except Exception as e:
        raise RuntimeError(f"Error loading text file {txt_path}: {e}")


def load_documents_from_txt_directory(txt_dir: Path) -> List[Dict]:
    """
    Load all text-based files from a directory as documents.
    Supports common text formats: .txt, .md, .rst, .text, .log, and files without extensions.
    Other file types are ignored.
    
    Args:
        txt_dir: Path to the directory containing text files
        
    Returns:
        List of document dictionaries
    """
    if not txt_dir.exists():
        raise FileNotFoundError(f"Directory not found: {txt_dir}")
    
    if not txt_dir.is_dir():
        raise ValueError(f"Path is not a directory: {txt_dir}")
    
    # Supported text file extensions
    TEXT_EXTENSIONS = {'.txt', '.md', '.rst', '.text', '.log', ''}
    
    # Find all text files recursively
    all_files = list(txt_dir.rglob("*"))
    text_files = [
        f for f in all_files 
        if f.is_file() and (f.suffix.lower() in TEXT_EXTENSIONS or f.suffix == '')
    ]
    
    if not text_files:
        # Check if there are any files at all to provide a helpful error message
        non_dir_files = [f for f in all_files if f.is_file()]
        if non_dir_files:
            file_extensions = {f.suffix for f in non_dir_files if f.suffix}
            raise ValueError(
                f"No supported text files found in directory: {txt_dir}\n"
                f"Found {len(non_dir_files)} file(s) with extension(s): {', '.join(sorted(file_extensions)) or 'none'}\n"
                f"Supported extensions: {', '.join(sorted(TEXT_EXTENSIONS - {''})) or 'none'} (and files without extensions)"
            )
        else:
            raise ValueError(f"No text files found in directory: {txt_dir}")
    
    # Count non-text files for informational message
    non_text_files = [f for f in all_files if f.is_file() and f.suffix.lower() not in TEXT_EXTENSIONS and f.suffix != '']
    if non_text_files:
        print(f"ℹ️  Found {len(non_text_files)} non-text file(s) in directory (ignored)")
    
    documents = []
    for text_file in sorted(text_files):
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                print(f"⚠️  Warning: File {text_file} is empty, skipping")
                continue
            
            # Use relative path from txt_dir as title (preserves subdirectory structure)
            rel_path = text_file.relative_to(txt_dir)
            # Remove extension for title, but keep the path structure
            title = str(rel_path.with_suffix('')) if rel_path.suffix else str(rel_path)
            
            documents.append({
                'title': title,
                'doc': content
            })
        except UnicodeDecodeError:
            print(f"⚠️  Warning: File {text_file} is not a valid text file (binary?), skipping")
            continue
        except Exception as e:
            print(f"⚠️  Warning: Error loading {text_file}: {e}, skipping")
            continue
    
    print(f"✅ Loaded {len(documents)} documents from {len(text_files)} text files")
    return documents


def prepare_documents_for_insertion(documents: List[Dict]) -> List[Dict]:
    """
    Convert documents to the format expected by TemporalGraphRAG.insert().
    Automatically detects the document format and processes accordingly.
    
    Args:
        documents: List of documents (either from corpus or txt files)
        
    Returns:
        List of documents in format {"title": str, "doc": str}
    """
    if not documents:
        return []
    
    # Auto-detect format: check if first document has 'title' and 'doc' keys (text format)
    # or 'cleaned_content'/'raw_content' keys (corpus format)
    first_doc = documents[0]
    is_corpus_format = 'cleaned_content' in first_doc or 'raw_content' in first_doc
    
    if not is_corpus_format:
        # Already in the correct format (from txt files)
        # Just validate and return
        for doc in documents:
            if 'title' not in doc or 'doc' not in doc:
                raise ValueError(f"Document missing required keys 'title' or 'doc': {list(doc.keys())}")
        return documents
    
    # Process corpus format documents
    prepared_docs = []
    for doc in documents:
        content = doc.get('cleaned_content', doc.get('raw_content', ''))
        if not content:
            print(f"⚠️  Warning: Document {doc.get('company_name', 'Unknown')} has no content, skipping")
            continue
        
        # Create a descriptive title
        company = doc.get('company_name', 'Unknown')
        year = doc.get('year', '')
        quarter = doc.get('quarter', '')
        if year and quarter:
            title = f"{company} {year} Q{quarter.upper()}"
        elif year:
            title = f"{company} {year}"
        else:
            title = company
        
        prepared_docs.append({
            'title': title,
            'doc': content
        })
    
    return prepared_docs




def main():
    """Main function to build the graph."""
    parser = argparse.ArgumentParser(
        description="Build Temporal GraphRAG knowledge graph from documents (ECT-QA corpus, text files, or directories) using config.yaml",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default='tgrag/configs/config.yaml',
        help='Path to configuration file (default: tgrag/configs/config.yaml)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for graph storage (overrides config.working_dir if set)'
    )
    parser.add_argument(
        '--num_docs',
        type=int,
        default=3,
        help='Number of documents to process from the corpus'
    )
    parser.add_argument(
        '--corpus_path',
        type=str,
        default='ect-qa/corpus/base.jsonl.gz',
        help='Path to the corpus file (.jsonl.gz), text file (.txt/.md/.rst/.text/.log), or directory of text files (overrides config.corpus_path if set)'
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=None,
        help='Override chunk size from config'
    )
    parser.add_argument(
        '--chunk_overlap',
        type=int,
        default=None,
        help='Override chunk overlap from config'
    )
    
    args = parser.parse_args()
    
    # Prepare override config
    override_config = {}
    if args.corpus_path:
        override_config['corpus_path'] = args.corpus_path
    if args.chunk_size:
        override_config['chunk_size'] = args.chunk_size
    if args.chunk_overlap:
        override_config['chunk_overlap'] = args.chunk_overlap
    if args.output_dir:
        override_config['working_dir'] = args.output_dir
    
    # Create TemporalGraphRAG from config (simplified!)
    print("="*60)
    print("Loading Configuration and Initializing TemporalGraphRAG")
    print("="*60)
    print(f"Config file: {args.config}")
    if override_config:
        print(f"Overrides: {override_config}")
    print()
    
    try:
        graph_rag = create_temporal_graphrag_from_config(
            config_path=args.config,
            config_type="building",
            override_config=override_config if override_config else None
        )
        print("✅ TemporalGraphRAG initialized from config")
        print(f"   Working directory: {graph_rag.working_dir}")
        print(f"   Chunk size: {graph_rag.chunk_token_size} tokens")
        print(f"   Chunk overlap: {graph_rag.chunk_overlap_token_size} tokens")
        print(f"   Entity summarization: {'Disabled' if graph_rag.disable_entity_summarization else 'Enabled'}")
        print(f"   Community summary: {'Enabled' if graph_rag.enable_community_summary else 'Disabled'}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error initializing TemporalGraphRAG: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load documents from corpus path (use config or override)
    from tgrag.src.config.config_loader import ConfigLoader
    config_loader = ConfigLoader(config_path=args.config)
    config = config_loader.get_config("building", override_args=override_config if override_config else None)
    corpus_path = Path(config.get('corpus_path', args.corpus_path))
    
    # Detect input type and load accordingly
    try:
        if corpus_path.is_file():
            if corpus_path.suffix == '.gz' or corpus_path.suffixes[-2:] == ['.jsonl', '.gz']:
                # JSONL.gz corpus file (e.g., ECT-QA)
                print(f"📚 Loading from corpus file: {corpus_path}")
                documents = load_documents_from_corpus(corpus_path, args.num_docs)
            elif corpus_path.suffix.lower() in {'.txt', '.md', '.rst', '.text', '.log'} or corpus_path.suffix == '':
                # Single text file
                print(f"📄 Loading from text file: {corpus_path}")
                documents = load_documents_from_txt_file(corpus_path)
            else:
                raise ValueError(
                    f"Unsupported file type: {corpus_path.suffix}\n"
                    f"Supported: .jsonl.gz (corpus), .txt/.md/.rst/.text/.log (text files), or files without extensions"
                )
        elif corpus_path.is_dir():
            # Directory of text files
            print(f"📁 Loading from directory: {corpus_path}")
            documents = load_documents_from_txt_directory(corpus_path)
        else:
            raise FileNotFoundError(f"Path not found: {corpus_path}")
        
        if not documents:
            print("❌ No documents loaded")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        sys.exit(1)
    
    # Prepare documents (auto-detects format)
    prepared_docs = prepare_documents_for_insertion(documents)
    print(f"✅ Prepared {len(prepared_docs)} documents for insertion")
    
    # Insert documents
    print("\n" + "="*60)
    print("Inserting documents and building graph...")
    print("="*60)
    print(f"Processing {len(prepared_docs)} documents...")
    print("This may take minutes to hours depending on document size and LLM response time.")
    print()
    
    try:
        graph_rag.insert(prepared_docs)
        print("\n✅ Graph building completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during graph building: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up HTTP clients to avoid unclosed session warnings
        try:
            from tgrag.src.llm.client import get_client_manager
            import asyncio
            client_manager = get_client_manager()
            # Create event loop if needed and close clients
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule cleanup
                    asyncio.create_task(client_manager.close_clients())
                else:
                    # If loop is not running, run cleanup
                    loop.run_until_complete(client_manager.close_clients())
            except RuntimeError:
                # No event loop, create one temporarily
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(client_manager.close_clients())
                loop.close()
        except Exception:
            # Ignore cleanup errors
            pass
    
    # Summary
    print("\n" + "="*60)
    print("BUILD SUMMARY")
    print("="*60)
    print(f"✅ Documents processed: {len(prepared_docs)}")
    print(f"✅ Graph stored in: {Path(graph_rag.working_dir).absolute()}")
    print(f"✅ Working directory: {graph_rag.working_dir}")
    print(f"✅ Configuration: {args.config}")
    print("="*60)


if __name__ == "__main__":
    main()

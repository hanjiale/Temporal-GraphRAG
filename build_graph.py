#!/usr/bin/env python3
"""
Build Temporal GraphRAG knowledge graph from documents.

This script:
1. Loads documents from the ECT-QA corpus
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


def prepare_documents_for_insertion(documents: List[Dict]) -> List[Dict]:
    """
    Convert corpus documents to the format expected by TemporalGraphRAG.insert().
    
    Args:
        documents: List of documents from the corpus
        
    Returns:
        List of documents in format {"title": str, "doc": str}
    """
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
        description="Build Temporal GraphRAG knowledge graph from ECT-QA corpus using config.yaml",
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
        help='Path to the corpus file (overrides config.corpus_path if set)'
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
    
    try:
        documents = load_documents_from_corpus(corpus_path, args.num_docs)
        if not documents:
            print("❌ No documents loaded")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        sys.exit(1)
    
    # Prepare documents
    prepared_docs = prepare_documents_for_insertion(documents)
    print(f"✅ Prepared {len(prepared_docs)} documents for insertion")
    
    # Insert documents
    print("\n" + "="*60)
    print("Inserting documents and building graph...")
    print("="*60)
    print(f"Processing {len(prepared_docs)} documents...")
    print("This may take several minutes depending on document size and LLM response time.")
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

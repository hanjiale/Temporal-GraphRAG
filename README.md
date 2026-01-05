# Temporal-GraphRAG (TG-RAG)

![arXiv](https://img.shields.io/badge/arXiv-2510.13590-b31b1b.svg)

Official implementation of **"RAG Meets Temporal Graphs: Time-Sensitive Modeling and Retrieval for Evolving Knowledge"**.

## Overview

Temporal-GraphRAG (TG-RAG) addresses the temporal blindness in conventional RAG systems by modeling knowledge as a bi-level temporal graph. This enables precise time-aware retrieval and efficient incremental updates as corpora evolve.

**Key Advantages:**
- 🕐 Explicit temporal fact representation
- 📊 Multi-granularity temporal summaries
- 🔄 Efficient incremental updates
- 🎯 Dynamic time-aware retrieval

## Installation

```bash
git clone https://github.com/hanjiale/Temporal-GraphRAG.git
cd Temporal-GraphRAG

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

**1. Set up API keys** (required for LLM and embedding providers):

```bash
# Create .env file or set environment variables
export OPENAI_API_KEY="your-openai-key-here"      # For OpenAI provider
export GOOGLE_API_KEY="your-google-key-here"      # For Gemini provider (or use GEMINI_API_KEY)
```

**2. Build and query:**

```bash
# Build a graph from documents
python build_graph.py --output_dir ./graph_output --corpus_path ./my_documents/

# Query the graph
python query_graph.py --question "Your question here" --working_dir ./graph_output --mode global
```



## Configurations

<details>
<summary><b>Entity Types</b></summary>

Customize which entity types are extracted by editing `tgrag/configs/prompts.yaml`:

```yaml
defaults:
  entity_types:
    - "financial concept"
    - "business segment"
    - "event"
    - "company"
    - "person"      
    - "product"
    - "location"
```

The system will only extract entities matching these configured types.

</details>

<details>
<summary><b>LLM and Embedding Providers</b></summary>

Configure in `tgrag/configs/config.yaml`:

```yaml
building:
  provider: "gemini"  # Options: openai, azure, bedrock, gemini, ollama
  model: "gemini-2.5-flash-lite"
  embedding_provider: "openai"
```

**Supported Providers:**
- **OpenAI** - Requires `OPENAI_API_KEY`
- **Azure OpenAI** - Requires Azure credentials (set via Azure SDK)
- **Amazon Bedrock** - Requires AWS credentials and `aioboto3`
- **Google Gemini** - Requires `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- **Ollama** - Requires local Ollama server (default: `http://localhost:11434`)

Set API keys via environment variables or `.env` file:
```bash
export OPENAI_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"  # or GEMINI_API_KEY
```

</details>

## Usage Examples

<details>
<summary><b>Building the Graph</b></summary>

The `build_graph.py` script automatically detects input type:

**ECT-QA corpus (JSONL.gz):**
```bash
python build_graph.py --output_dir ./graph_output --corpus_path ./ect-qa/corpus/base.jsonl.gz --num_docs 10
```

**Single text file:**
```bash
python build_graph.py --output_dir ./graph_output --corpus_path ./my_document.txt
```

**Directory of text files (recursive):**
```bash
python build_graph.py --output_dir ./graph_output --corpus_path ./my_documents/
```

Supported text formats: `.txt`, `.md`, `.rst`, `.text`, `.log`, and files without extensions.

</details>

<details>
<summary><b>Query Modes</b></summary>

```bash
# Local mode - for specific facts
python query_graph.py --question "What was Company X's revenue in Q3 2023?" --mode local

# Global mode - for trends and summarization
python query_graph.py --question "How did tech companies navigate 2023 challenges?" --mode global

# Naive mode - simple RAG
python query_graph.py --question "What is artificial intelligence?" --mode naive
```

</details>

<details>
<summary><b>Python API Examples</b></summary>

```python
from tgrag import create_temporal_graphrag_from_config

# Build the graph
graph_rag = create_temporal_graphrag_from_config(
    config_path="tgrag/configs/config.yaml",
    config_type="building"
)

# Insert documents
graph_rag.insert([{"title": "Doc 1", "doc": "content..."}])

# Query the graph
graph_rag = create_temporal_graphrag_from_config(
    config_path="tgrag/configs/config.yaml",
    config_type="querying"
)
answer = graph_rag.query("Your question here", mode="global")
```

</details>


## ECT-QA Dataset

High-quality benchmark for time-sensitive question answering:

- **Corpus:** 480 earnings call transcripts (24 companies, 2020-2024)
- **Questions:** 1,005 specific + 100 abstract temporal queries

The dataset is also available on Hugging Face: [austinmyc/ECT-QA](https://huggingface.co/datasets/austinmyc/ECT-QA)

You can load it using:
```python
from datasets import load_dataset

# Load questions dataset
questions = load_dataset("austinmyc/ECT-QA", "questions")

# Load corpus dataset
corpus = load_dataset("austinmyc/ECT-QA", "corpus")
```


## Repository Structure
```
Temporal-GraphRAG/
├── tgrag/                          
│   ├── configs/                        
│   │   ├── config.yaml             # Main configuration
│   │   └── prompts.yaml            # prompts for indexing and querying
│   └── src/               
│       ├── temporal_graphrag.py    
│       └── ...  
├── ect-qa/                         # ECT-QA dataset               
│   ├── corpus/                     
│   │   ├── base.jsonl.gz           # 2020 - 2023
│   │   └── new.jsonl.gz            # 2024
│   └── questions/           
│       ├── local_base.jsonl 
│       ├── local_new.jsonl 
│       ├── global_base.jsonl 
│       └── global_new.jsonl    
├── graph_storage/
│   └── ...                         # Output graphs         
├── build_graph.py                  # Script to build knowledge graph
├── query_graph.py                  # Script to query the graph
├── requirements.txt                                      
├── README.md                       
├── LICENSE                         
└── .gitignore                      
```
## Citation

```bibtex
@article{han2025rag,
  title={RAG Meets Temporal Graphs: Time-Sensitive Modeling and Retrieval for Evolving Knowledge},
  author={Han, Jiale and Cheung, Austin and Wei, Yubai and Yu, Zheng and Wang, Xusheng and Zhu, Bing and Yang, Yi},
  journal={arXiv preprint arXiv:2510.13590},
  year={2025}
}
```

## Acknowledgments

Paper available at: [arXiv:2510.13590](https://arxiv.org/abs/2510.13590)

---

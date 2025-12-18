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

## Configuration

TG-RAG supports multiple LLM and embedding providers. Configure in `tgrag/configs/config.yaml` and via `.env`.

## Quick Start

### Building the Graph

Build a temporal knowledge graph from documents using the `build_graph.py` script:

```bash
# Build graph with default config
python build_graph.py --output_dir ./graph_output --num_docs 10
```

The script loads configuration from `tgrag/configs/config.yaml` and builds the graph from the ECT-QA corpus (or your specified corpus path).

### Querying the Graph

```bash
# Query with a question (local mode for specific facts)
python query_graph.py --question "What was Company X's revenue in Q3 2023?" --mode local

# Query with global mode for trends/summarization
python query_graph.py --question "How did tech companies navigate 2023 challenges?" --mode global

```

### Using the Python API

You can also use the Python API directly:

```python
from tgrag import create_temporal_graphrag_from_config

# Build the graph
graph_rag = create_temporal_graphrag_from_config(
    config_path="tgrag/configs/config.yaml",
    config_type="building"
)
graph_rag.insert([{"title": "Doc 1", "doc": "content..."}])

# Query the graph
graph_rag = create_temporal_graphrag_from_config(
    config_path="tgrag/configs/config.yaml",
    config_type="querying"
)
answer = graph_rag.query("Your question here", mode="global")
```


## ECT-QA Dataset

High-quality benchmark for time-sensitive question answering:

- **Corpus:** 480 earnings call transcripts (24 companies, 2020-2024)
- **Questions:** 1,005 specific + 100 abstract temporal queries


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

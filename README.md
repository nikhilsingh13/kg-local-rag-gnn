# kg-local-rag-gnn

## Interactive Knowledge Graph Visualization of GNN Research Papers using Local RAG

This project creates an interactive knowledge graph from a selected set of Graph Neural Network (GNN) research papers. The reserach papers are highly cited papers from arXiv on GNN topic. By leveraging Retrieval-Augmented Generation (RAG) technique, the important information such as models, datasets, tasks, and authors are extracted and to visualize their interconnections. 

This experiment is to navigate and understand the GNN landscape and also to make one more knowledgeable on the topic of RAGs and LLM usage.

---

## Features

* **PDF to Text Extraction**: Converts research papers from PDF format into clean, readable text.
* **Information Extraction**: Use LLMs to identify entities (e.g., GNN models, datasets) and relationships between them.
* **Knowledge Graph Construction**: Builds a structured knowledge graph using the extracted data, representing papers, concepts, and their links.
* **Local RAG Integration**: Augments user queries with relevant context from the knowledge graph and paper content, providing insightful answers.
* **Interactive Visualisation**: A web application for dynamic exploration of the knowledge graph, allowing users to filter, search, and discover connections.

---

## Papers Included

This project is built upon the analysis of the following foundational and influential GNN research papers, all sourced from arXiv for open access and research purposes:

All papers are placed in the `data/raw/` directory in their original PDF format.

1.  **Semi-Supervised Classification with Graph Convolutional Networks**
    * **Authors:** Thomas N. Kipf, Max Welling
    * **Year:** 2017
    * **arXiv ID:** `1609.02907`
    * **Filename Example:** `1609.02907v4.pdf`

2.  **Inductive Representation Learning on Large Graphs (GraphSAGE)**
    * **Authors:** William L. Hamilton, Rex Ying, Jure Leskovec
    * **Year:** 2017
    * **arXiv ID:** `1706.02216`
    * **Filename Example:** `1706.02216v4.pdf`

3.  **Graph Attention Networks (GAT)**
    * **Authors:** Petar Veličković et al.
    * **Year:** 2018
    * **arXiv ID:** `1710.10903`
    * **Filename Example:** `1710.10903v3.pdf`

4.  **Neural Message Passing for Quantum Chemistry (MPNN)**
    * **Authors:** Justin Gilmer et al.
    * **Year:** 2017
    * **arXiv ID:** `1704.01212`
    * **Filename Example:** `1704.01212v2.pdf`

5.  **How Powerful are Graph Neural Networks? (GIN)**
    * **Authors:** Keyulu Xu et al.
    * **Year:** 2019
    * **arXiv ID:** `1810.00826`
    * **Filename Example:** `1810.00826v3.pdf`

6.  **DeepWalk: Online Learning of Social Representations**
    * **Authors:** Bryan Perozzi, Rami Al-Rfou, Steven Skiena
    * **Year:** 2014
    * **arXiv ID:** `1403.6652`
    * **Filename Example:** `1403.6652v2.pdf`

7.  **node2vec: Scalable Feature Learning for Networks**
    * **Authors:** Aditya Grover, Jure Leskovec
    * **Year:** 2016
    * **arXiv ID:** `1607.00653`
    * **Filename Example:** `1607.00653v1.pdf`

8.  **A Comprehensive Survey on Graph Neural Networks (Survey)**
    * **Authors:** Zonghan Wu et al.
    * **Year:** 2019
    * **arXiv ID:** `1901.00596`
    * **Filename Example:** `1901.00596v4.pdf`

---

## Getting Started

To get this project up and running locally, follow these steps:

### Prerequisites

* Python 3.11+
* `uv` (recommended for dependency management: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
  * Please follow `uv` [official documentation](https://docs.astral.sh/uv/getting-started/installation/) for more details.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nikhilsingh13/kg-local-rag-gnn.git
    cd kg-local-rag-gnn
    ```

2.  **Install dependencies using `uv`:**
    ```bash
    uv sync
    ```
    This will install all required packages defined in `pyproject.toml`.

3.  **Place PDF Papers:**
    - The 8 papers listed above are a part of the repo. You can find them in the folder `data/raw/`.
    - One can add more papers and go through the notebooks to connect more papers.

### Running the Pipeline

The project can be run in stages using the provided notebooks or potentially via `main.py` once it's set up to orchestrate the full pipeline.

1.  **PDF Extraction:**
    Open and run `notebooks/01_pdf_extraction.ipynb` to process the PDFs into raw text.

2.  **Information Extraction:**
    Proceed to `notebooks/02_information_extraction.ipynb` to extract entities and relationships. 
    
    *Note: Place your API key in `.env` file as `OPENAI_API_KEY=your_key_here`*

3.  **Graph Construction:**
    Execute `notebooks/03_graph_construction.ipynb` to build the knowledge graph.

### Running the Interactive Application

Once the graph data is processed, you can launch the interactive app:
```bash
uv run app/app.py
```
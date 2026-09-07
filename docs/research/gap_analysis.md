# MemoryGraph-AI: Novelty Claims & Gap Analysis (Task R6)

This document establishes the competitive positioning, theoretical gaps in existing paradigms, and explicit novelty claims of **MemoryGraph-AI**.

---

## 1. Comparative Architectural Taxonomy

The table below contrasts MemoryGraph-AI against five dominant paradigms across six core dimensions:

| Architectural Dimension | Traditional PKM (Obsidian, Roam) | Flat Vector RAG (DPR, LangChain) | Microsoft GraphRAG (Leiden Communities) | HippoRAG (Hippocampal KG) | MemGPT / Zep (Agentic Memory) | **MemoryGraph-AI (Ours)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Indexing & Graph Topology** | Manual bi-directional links (`[[wiki]]`) | Flat chunk embeddings (no topology) | Offline hierarchical Leiden clusters | OpenIE entity triples + PPR | Relational SQLite + Flat vector cache | **Continuous Dual-Store: Typed Hierarchical Entity-Relation Graph + Vector** |
| **2. Multi-Hop Reasoning** | Manual visual graph inspection | Poor (retrieves top-$k$ disconnected chunks) | High (hierarchical community summaries) | High (Personalized PageRank) | Moderate (context window swapping) | **High (Multi-tier BFS / PPR + Cross-Document Edge Traversal)** |
| **3. Temporal Cognitive Decay** | None (static graph) | None (uniform chunk weighting) | None (static snapshot indexing) | None (static edge weights) | Basic LRU/FIFO buffer eviction | **ACT-R / FSRS-inspired dynamic activation decay $\mathcal{A}(n, t)$** |
| **4. Spatial Semantic Representation** | 2D Force-Directed (Visual Hairball at $N > 10^3$) | None (high-D latent vectors only) | None (textual tree hierarchy) | None (graph adjacency only) | None | **3D Parametric UMAP + Topology-Constrained Spring Manifold** |
| **5. Learning Path Synthesis** | Manual MOCs (Maps of Content) | None | None | None | None | **Automated Prerequisite DAG Extraction & Knowledge Gap Induction** |
| **6. Real-Time Incremental Updates** | Instant (file-system level) | Fast (append chunk vector) | Prohibitive (full graph re-clustering needed) | Moderate | Fast | **Real-time incremental node insertion, alias resolution, & UMAP projection** |

---

## 2. Theoretical & Practical Gaps in Prior Art

### Gap 1: The "Context-Free Chunks" vs. "Static Monolithic Graph" Dilemma
* Standard Dense RAG breaks documents into isolated snippets, stripping structural and relational context.
* Existing GraphRAG approaches (Edge et al., 2024) require heavy batch processing to compute global community summaries, making them unsuitable for personal knowledge bases where notes are continuously created, edited, and linked.
* *MemoryGraph-AI Resolution*: Provides real-time incremental schema extraction and local graph updates coupled with hybrid vector-graph retrieval.

### Gap 2: The Absence of Cognitive Decay in AI Memory Systems
* Contemporary RAG and memory frameworks treat a document written 5 years ago with the same saliency as a note reviewed 5 minutes ago if lexical/semantic match is high.
* Human memory operates on recency, frequency, and spaced reinforcement to maintain an optimal working set.
* *MemoryGraph-AI Resolution*: Implements an adaptive temporal decay model $\mathcal{A}(n, t)$ that dynamically modulates retrieval ranking and visual prominence based on interaction history.

### Gap 3: Visual Scalability Collapse in PKM Graphs
* 2D force-directed graphs in existing PKM software collapse into unreadable hairballs as notes scale into the thousands, with no semantic spatial organization (e.g., related topics randomly scattered across 2D space).
* *MemoryGraph-AI Resolution*: 3D Parametric UMAP maps high-dimensional semantic embeddings to Euclidean $\mathbb{R}^3$ while preserving semantic clusters, reinforced by force-directed edge springs to maintain topological fidelity.

---

## 3. Four Core Novelty Claims

1. **Adaptive Temporal-Cognitive Activation**: Unification of ACT-R power-law decay and FSRS stability updates with hybrid retrieval ranking to emulate human working memory.
2. **Topology-Constrained 3D Semantic Manifold Projection**: A hybrid loss formulation combining parametric UMAP cross-entropy with graph spring tension for stable, clutter-free 3D personal knowledge graphs.
3. **Multi-Stage Adaptive Hybrid Fusion**: A composite reranking framework harmonizing dense embeddings, BM25 lexical matches, personalized graph proximity, and temporal decay weights.
4. **Automated Prerequisite DAG Learning Synthesis**: Automated extraction of directed acyclic concept graphs for active curriculum planning and knowledge-gap detection from unstructured personal notes.

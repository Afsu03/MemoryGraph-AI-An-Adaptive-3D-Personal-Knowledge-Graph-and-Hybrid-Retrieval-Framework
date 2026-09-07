# MemoryGraph-AI: Systematic Literature Review (Tasks R1–R5)

This document provides a comprehensive synthesis of foundational and state-of-the-art literature across five research pillars underpinning **MemoryGraph-AI**: Personal Knowledge Management (PKM), Retrieval-Augmented Generation (RAG) & Graph-RAG, Automatic Knowledge Graph Construction (AKGC), Computational Cognitive Memory Models, and High-Dimensional 3D Semantic Visualization.

---

## 1. Personal Knowledge Management (PKM) & Cognitive Note-Taking Systems (Task R1)

Personal Knowledge Management (PKM) systems have evolved from hierarchical, folder-based storage (e.g., Evernote, OneNote) to network-oriented associative structures (e.g., Roam Research, Obsidian, Logseq).

### 1.1 The Zettelkasten Method & Associative Indexing
* **Luhmann's Zettelkasten** established the principle of atomic note units linked via bi-directional alphanumeric cross-references. Modern digital implementations replace alphanumeric indexes with wiki-links (`[[Concept]]`).
* **Cognitive Limiters in Flat Markdown Graphs**:
  1. *Graph Entropy & Visual Hairballs*: As the number of notes $N > 1,000$, force-directed graph representations degenerate into dense visual "hairballs" with low interpretability and high cognitive clutter (Ahrens, 2017).
  2. *Unstructured Link Semantics*: Traditional PKM tools treat all links as homogeneous undirected relations, failing to differentiate between causal (`causes`), hierarchical (`is_a`), prerequisite (`prerequisite_of`), or associative (`relates_to`) edges.
  3. *Static Topology*: Link structures remain static unless manually updated by the user, lacking mechanisms to reflect cognitive decay, forgetting, or emerging thematic clusters.

### 1.2 Key Literature
* Ahrens, S. (2017). *How to Take Smart Notes: One Simple Technique to Boost Writing, Learning and Thinking*.
* Bush, V. (1945). "As We May Think." *The Atlantic Monthly*. (The foundational Memex associative trails).
* Jones, W., & Teevan, J. (2007). *Personal Information Management*. University of Washington Press.

---

## 2. Retrieval-Augmented Generation (RAG) & Graph-RAG Paradigms (Task R2)

Retrieval-Augmented Generation (RAG) mitigates LLM hallucinations and knowledge obsolescence by grounding generation in external text corpora.

### 2.1 Dense Vector RAG vs. Graph-Augmented RAG
* **Dense Passage Retrieval (DPR)** (Lewis et al., 2020; Karpukhin et al., 2020) maps query $q$ and passage $p$ into a shared dense latent space $\mathbb{R}^d$ using dual-encoders. Similarity is computed via inner product $\langle \mathbf{e}_q, \mathbf{e}_p \rangle$.
* **Failure Modes of Dense Vector RAG**:
  1. *Multi-Hop Reasoning Blindness*: Dense retrieval struggles with multi-hop associative queries (e.g., "How does Concept A connect to Concept D through intermediate research papers?") because passages are indexed independently without explicit relation traversal.
  2. *Global Sensemaking Deficit*: Vector RAG is optimized for point-lookup queries but fails at corpus-level summarization or conceptual synthesis across fragmented files.
* **GraphRAG Paradigms**:
  * **Microsoft GraphRAG** (Edge et al., 2024): Constructs entity-relation graphs from text chunks, partitions them using the Leiden community detection algorithm, and generates pre-computed hierarchical summaries. While powerful for global sensemaking, it incurs high token indexing costs and lacks real-time incremental update capabilities.
  * **HippoRAG** (Gutiérrez et al., 2024): Inspired by hippocampal indexing theory, HippoRAG uses Personalized PageRank (PPR) over open-domain knowledge graphs constructed by LLMs, achieving rapid multi-hop associative recall without exhaustive fine-tuning.

### 2.2 Key Literature
* Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.
* Edge, D., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." *arXiv:2404.16130*.
* Gutiérrez, B. J., et al. (2024). "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models." *arXiv:2405.14831*.

---

## 3. Automatic Knowledge Graph Construction (AKGC) & LLM Extraction (Task R3)

Transforming unstructured personal notes into formal knowledge graphs requires extracting entities, resolving aliases, and establishing typed relational edges.

### 3.1 LLM-Driven Information Extraction (IE)
* Foundation models (e.g., GPT-4o, Claude 3.5, Llama 3) exhibit strong zero-shot and few-shot capabilities for joint Entity and Relation Extraction (Joint NER+RE) when guided by strict structured decoding libraries (e.g., Instructor, Pydantic, JSON Schema constraints).
* **Entity Resolution & Disambiguation (ERD)**:
  * In personal knowledge graphs, entities often appear with aliases (e.g., "Backprop", "BP", "Error Backpropagation").
  * Hybrid ERD utilizes normalized character n-gram distance (Levenshtein / Jaro-Winkler) combined with dense embedding cosine similarity $\cos(\mathbf{e}_1, \mathbf{e}_2) \ge \tau_{\text{merge}}$ and LLM verification for borderline ambiguity cases ($0.75 \le \text{sim} \le 0.88$).

### 3.2 Key Literature
* Dagdelen, J., et al. (2024). "Structured Information Extraction from Complex Scientific Text with Large Language Models." *Nature Communications*.
* Zaratiana, U., et al. (2023). "GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer." *arXiv:2311.08526*.
* Wang, X., et al. (2023). "A Survey on Large Language Models for Knowledge Graph Construction and Reasoning." *IEEE TKDE*.

---

## 4. Computational Cognitive Memory Models & Forgetting Curves (Task R4)

Human cognition does not treat all memories equally; knowledge accessibility decays over time unless reinforced through spaced retrieval or active application.

### 4.1 Classical & Modern Decay Models
1. **Ebbinghaus Forgetting Curve (1885)**:
   $$R(t) = e^{-\frac{t}{S}}$$
   where $R$ is memory retention, $t$ is elapsed time, and $S$ is memory stability (half-life).
2. **ACT-R (Adaptive Control of Thought-Rational) Activation** (Anderson et al., 2004):
   $$B_i = \ln\left( \sum_{j=1}^{n} (t - t_j)^{-d} \right) + \sum_{k} W_k S_{ki}$$
   Base-level activation $B_i$ reflects the power-law decay of historical access instances $t_j$ with decay rate $d \approx 0.5$.
3. **Free Spaced Repetition Scheduler (FSRS-4.5)** (Ye et al., 2023):
   Models memory using three cognitive parameters: Retrievability ($R$), Stability ($S$), and Difficulty ($D$), updating stability exponentially upon successful recall:
   $$S' = S \cdot \left(1 + e^{w_1} \cdot (11 - D) \cdot S^{-w_2} \cdot (e^{(1-R) w_3} - 1)\right)$$

### 4.2 Integration into MemoryGraph-AI
MemoryGraph-AI adopts an adapted continuous ACT-R/FSRS formulation: every node in the personal knowledge graph maintains a dynamic activation state $\mathcal{A}(n, t)$. Nodes that are frequently referenced or reinforced remain salient, while stale, unreferenced concepts gradually fade in visual opacity and retrieval prioritization, mirroring biological human working memory.

---

## 5. High-Dimensional Dimensionality Reduction & 3D Spatial Visualization (Task R5)

Presenting complex multi-relational graphs to users requires projecting high-dimensional semantic spaces ($\mathbb{R}^d, d \ge 1536$) into an intuitive 3-dimensional Euclidean coordinate manifold ($\mathbb{R}^3$).

### 5.1 Manifold Learning: UMAP vs. t-SNE vs. Force-Directed
* **t-SNE (van der Maaten & Hinton, 2008)**: Excels at local clustering but distorts global topological distances and does not support fast out-of-sample projection for newly ingested documents.
* **UMAP (Uniform Manifold Approximation and Projection)** (McInnes et al., 2018):
  * Preserves both local neighborhood structure and global Riemannian manifold geometry using fuzzy simplicial sets.
  * Supports *Parametric UMAP* (Sainburg et al., 2021), allowing real-time $\mathbb{R}^d \to \mathbb{R}^3$ inference via a lightweight neural network encoder.
* **Hybrid Spatial Relaxation**:
  Pure manifold projection ignores explicit topological graph edges (e.g., `PREREQUISITE_OF`), whereas pure force-directed layouts ignore global semantic similarities. MemoryGraph-AI unifies both by combining parametric UMAP coordinates as spatial anchor priors with spring-tension edge constraints.

### 5.2 Key Literature
* McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." *arXiv:1802.03426*.
* Sainburg, T., et al. (2021). "Parametric UMAP: Learning embeddings with deep neural networks for representation and semi-supervised dimension reduction." *Pattern Recognition*.
* Jacomy, M., et al. (2014). "ForceAtlas2, a Continuous Graph Layout Algorithm for Handy Network Visualization Designed for the Gephi Software." *PLOS ONE*.

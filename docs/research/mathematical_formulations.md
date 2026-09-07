# MemoryGraph-AI: Mathematical Formulations (Tasks R7–R9)

This document provides the formal mathematical derivations, definitions, and objective functions governing **MemoryGraph-AI**.

---

## 1. Temporal Memory-Decay Activation Model (Task R7)

Let $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{W})$ be a heterogeneous personal knowledge graph where $\mathcal{V} = \mathcal{V}_{\text{doc}} \cup \mathcal{V}_{\text{chunk}} \cup \mathcal{V}_{\text{entity}}$ and $\mathcal{E}$ represents typed relational edges.

### 1.1 Node Base Activation Energy
For any node $n \in \mathcal{V}$ at query time $t$, let $\mathcal{H}_n = \{(t_1, r_1), (t_2, r_2), \dots, (t_k, r_k)\}$ be the historical log of interaction events, where $t_i \le t$ is the timestamp of the $i$-th interaction and $r_i \in (0, 1]$ represents the interaction quality/relevance feedback.

The base-level cognitive activation energy $\mathcal{A}(n, t)$ is defined as:

$$\mathcal{A}(n, t) = \ln\left( \sum_{i=1}^{k} r_i \cdot \left(\frac{t - t_i + \epsilon}{t_0}\right)^{-d} \right) + \beta \cdot \Pi(n)$$

Where:
* $t_0 > 0$ is the characteristic time unit (e.g., 1 hour or 1 day).
* $\epsilon > 0$ is an infinitesimal smoothing factor preventing division by zero for real-time interactions ($t \to t_k$).
* $d \in (0, 1)$ is the cognitive decay power parameter (empirically initialized to $d = 0.5$ following ACT-R literature).
* $\Pi(n) = \frac{\text{deg}(n)}{\max_{v \in \mathcal{V}} \text{deg}(v)}$ is the normalized topological prior centrality.
* $\beta \in [0, 1]$ balances topological importance against temporal recency.

### 1.2 Dynamic Memory Stability Update (FSRS-Inspired)
Each node maintains a dynamic memory stability parameter $\mathcal{S}(n) > 0$, representing the half-life of knowledge retrievability. Upon a new retrieval reinforcement event at timestamp $t_{\text{new}}$ with relevance feedback $r \in [0, 1]$:

$$\mathcal{S}_{\text{new}}(n) = \mathcal{S}_{\text{old}}(n) \cdot \left( 1 + \alpha \cdot r \cdot \exp\left(-\frac{\Delta t}{\mathcal{S}_{\text{old}}(n)}\right) \right)$$

where $\Delta t = t_{\text{new}} - t_{\text{last}}$, and $\alpha > 0$ is the memory consolidation gain rate.

### 1.3 Normalized Memory-Decay Weight for Retrieval
The bounded activation factor $\omega_{\text{decay}}(n, t) \in (0, 1]$ utilized in downstream retrieval reranking is obtained via the logistic sigmoid transform:

$$\omega_{\text{decay}}(n, t) = \sigma(\gamma_0 \cdot \mathcal{A}(n, t) + \gamma_1) = \frac{1}{1 + e^{-(\gamma_0 \mathcal{A}(n, t) + \gamma_1)}}$$

---

## 2. Adaptive Hybrid Fusion Ranking Function (Task R8)

Given a user query $q$, the retrieval engine operates across four heterogeneous channels:
1. **Dense Semantic Embedding Similarity** $\mathcal{S}_{\text{dense}}(q, d) = \cos(\mathbf{e}_q, \mathbf{e}_d) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\| \|\mathbf{e}_d\|}$
2. **Lexical BM25 Score** $\mathcal{S}_{\text{lexical}}(q, d) = \text{BM25}(q, d)$
3. **Graph Structural Proximity / Personalized PageRank (PPR)** $\mathcal{S}_{\text{graph}}(q, d)$
4. **Temporal Memory Activation** $\omega_{\text{decay}}(d, t)$

### 2.1 Graph Proximity via Seeded Personalized PageRank
Let $\mathcal{V}_{\text{seed}} \subset \mathcal{V}$ be the top-$M$ seed nodes matched by the initial dense and lexical retrieval stages. The personalized teleport vector $\mathbf{p} \in \mathbb{R}^{|\mathcal{V}|}$ is defined as:

$$p_i = \begin{cases} \frac{\mathcal{S}_{\text{dense}}(q, v_i)}{\sum_{u \in \mathcal{V}_{\text{seed}}} \mathcal{S}_{\text{dense}}(q, u)} & \text{if } v_i \in \mathcal{V}_{\text{seed}} \\ 0 & \text{otherwise} \end{cases}$$

The steady-state Personalized PageRank vector $\mathbf{\pi} \in \mathbb{R}^{|\mathcal{V}|}$ satisfies:

$$\mathbf{\pi} = (1 - \delta) \mathbf{P}^\top \mathbf{\pi} + \delta \mathbf{p}$$

where $\mathbf{P}$ is the row-stochastic transition probability matrix of graph $\mathcal{G}$ and $\delta \in (0, 1)$ is the restart probability ($\delta = 0.15$). For any document/chunk $d$, $\mathcal{S}_{\text{graph}}(q, d) = \pi_d$.

### 2.2 Weighted Reciprocal Rank Fusion (RRF) with Adaptive Decay Modulation
To reconcile heterogeneous score distributions without fragile calibration, we apply Weighted Reciprocal Rank Fusion over rank positions $\text{rank}_k(d)$ augmented by linear modulation:

$$\mathcal{S}_{\text{hybrid}}(q, d) = \left[ \sum_{k \in \{\text{dense}, \text{lexical}, \text{graph}\}} \frac{\lambda_k}{c + \text{rank}_k(d)} \right] \cdot \left( \lambda_{\text{decay}} \cdot \omega_{\text{decay}}(d, t) + (1 - \lambda_{\text{decay}}) \right)$$

where:
* $c = 60$ is the standard RRF smoothing constant.
* $\lambda_{\text{dense}}, \lambda_{\text{lexical}}, \lambda_{\text{graph}} \ge 0$ are modality weights satisfying $\sum \lambda_k = 1$.
* $\lambda_{\text{decay}} \in [0, 1]$ controls the strength of temporal memory gating.

---

## 3. 3D Parametric Semantic Spatialization & Layout Optimization (Task R9)

We seek a coordinate mapping $f_\theta: \mathbb{R}^D \to \mathbb{R}^3$ that embeds high-dimensional document/concept vectors $\mathbf{x}_i \in \mathbb{R}^D$ into 3D positions $\mathbf{y}_i = f_\theta(\mathbf{x}_i) \in \mathbb{R}^3$.

### 3.1 High-Dimensional Fuzzy Simplicial Set
In the high-dimensional embedding space, the directional fuzzy membership strength $p_{j|i}$ between node $i$ and $j$ is:

$$p_{j|i} = \exp\left( -\frac{\max(0, \|\mathbf{x}_i - \mathbf{x}_j\|_2 - \rho_i)}{\sigma_i} \right)$$

where $\rho_i$ is the distance to the nearest neighbor of $\mathbf{x}_i$, and $\sigma_i$ satisfies $\sum_{j} \exp\left(-\frac{\max(0, \|\mathbf{x}_i - \mathbf{x}_j\| - \rho_i)}{\sigma_i}\right) = \log_2(k)$. Symmetrized probabilities are $p_{ij} = p_{i|j} + p_{j|i} - p_{i|j} p_{j|i}$.

### 3.2 Low-Dimensional 3D Manifold Kernel
In the 3D target space $\mathbb{R}^3$, the pairwise similarity $q_{ij}$ is modeled with a smooth curve parameterization:

$$q_{ij} = \left( 1 + a \|\mathbf{y}_i - \mathbf{y}_j\|_2^{2b} \right)^{-1}$$

### 3.3 Composite Topology-Constrained Objective Function
The unified spatial optimization loss $\mathcal{L}_{\text{spatial}}$ combines UMAP cross-entropy, topological edge spring tension, and electrostatic repulsion:

$$\mathcal{L}_{\text{spatial}} = \mathcal{L}_{\text{UMAP}} + \mathcal{L}_{\text{spring}} + \mathcal{L}_{\text{repulsion}}$$

Where:
$$\mathcal{L}_{\text{UMAP}} = \sum_{i \neq j} \left[ p_{ij} \ln\left(\frac{p_{ij}}{q_{ij}}\right) + (1 - p_{ij}) \ln\left(\frac{1 - p_{ij}}{1 - q_{ij}}\right) \right]$$

$$\mathcal{L}_{\text{spring}} = \gamma \sum_{(u, v) \in \mathcal{E}} w_{uv} \cdot \|\mathbf{y}_u - \mathbf{y}_v\|_2^2$$

$$\mathcal{L}_{\text{repulsion}} = \mu \sum_{u \neq v} \frac{1}{\|\mathbf{y}_u - \mathbf{y}_v\|_2^2 + \kappa}$$

Here $\gamma > 0$ enforces that connected graph entities remain in adjacent visual clusters, while $\mu > 0$ and $\kappa > 0$ prevent node overlap and visual singularity collapse in 3D WebGL rendering.

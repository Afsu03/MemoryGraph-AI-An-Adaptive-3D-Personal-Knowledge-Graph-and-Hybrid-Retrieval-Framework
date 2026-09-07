# MemoryGraph-AI: Simulation & Mathematical Verification Suite

This directory contains standalone Python verification scripts validating the core theoretical and mathematical models of **MemoryGraph-AI** (Phase 1):

1. **`decay_model_sim.py`**:
   - Implements and simulates Task R7's ACT-R power-law decay function $\mathcal{A}(n, t)$ and FSRS stability consolidation rule $\mathcal{S}(n)$ across a 180-day virtual learning timeline.
   - Run: `python decay_model_sim.py`

2. **`hybrid_fusion_sim.py`**:
   - Simulates Task R8's 4-channel Weighted Reciprocal Rank Fusion (RRF) with graph Personalized PageRank and temporal decay gating.
   - Run: `python hybrid_fusion_sim.py`

3. **`spatial_3d_sim.py`**:
   - Implements Task R9's composite loss optimization ($\mathcal{L}_{\text{spatial}}$) mapping high-dimensional semantic vectors to Euclidean $\mathbb{R}^3$ with spring-tension edge constraints.
   - Run: `python spatial_3d_sim.py`

"""
MemoryGraph-AI: Adaptive Hybrid Retrieval Fusion Simulation
Validates the Weighted Reciprocal Rank Fusion (RRF) with Graph Proximity and Temporal Gating (Task R8).
"""

import numpy as np
from typing import List, Dict, Tuple

class HybridRetrievalSimulator:
    def __init__(self, c: int = 60, lambda_dense: float = 0.35, lambda_lex: float = 0.25,
                 lambda_graph: float = 0.40, lambda_decay: float = 0.30):
        self.c = c
        self.lambda_dense = lambda_dense
        self.lambda_lex = lambda_lex
        self.lambda_graph = lambda_graph
        self.lambda_decay = lambda_decay

    def compute_rrf_rank(self, ranked_items: List[str]) -> Dict[str, int]:
        """Maps an ordered list of item IDs to 1-based rank positions."""
        return {item: i + 1 for i, item in enumerate(ranked_items)}

    def fuse(self,
             dense_ranking: List[str],
             lexical_ranking: List[str],
             graph_ranking: List[str],
             decay_weights: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Fuses candidates using Weighted Reciprocal Rank Fusion modulated by temporal activation:
        S_hybrid(q, d) = [ sum_k (lambda_k / (c + rank_k(d))) ] * [ lambda_dec * w_dec(d) + (1 - lambda_dec) ]
        """
        dense_ranks = self.compute_rrf_rank(dense_ranking)
        lex_ranks = self.compute_rrf_rank(lexical_ranking)
        graph_ranks = self.compute_rrf_rank(graph_ranking)

        all_candidates = set(dense_ranks.keys()) | set(lex_ranks.keys()) | set(graph_ranks.keys())
        scored_candidates = []

        default_rank = 1000  # Penalty rank for unretrieved candidates in a channel

        for doc_id in all_candidates:
            r_d = dense_ranks.get(doc_id, default_rank)
            r_l = lex_ranks.get(doc_id, default_rank)
            r_g = graph_ranks.get(doc_id, default_rank)

            rrf_base = (
                (self.lambda_dense / (self.c + r_d)) +
                (self.lambda_lex / (self.c + r_l)) +
                (self.lambda_graph / (self.c + r_g))
            )

            w_dec = decay_weights.get(doc_id, 0.5)
            temporal_mod = self.lambda_decay * w_dec + (1.0 - self.lambda_decay)
            final_score = rrf_base * temporal_mod

            scored_candidates.append((doc_id, final_score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates


def run_hybrid_simulation():
    print("=" * 70)
    print("MemoryGraph-AI: Adaptive Hybrid Retrieval Fusion Benchmark")
    print("=" * 70)

    # Documents
    # doc1: Multi-hop connected concept, recently reinforced (True Target)
    # doc2: High lexical keyword match but obsolete/stale note
    # doc3: High dense embedding similarity but disconnected from graph context
    # doc4: Moderate in all channels, well-connected
    
    dense_results = ["doc3", "doc1", "doc4", "doc2"]
    lexical_results = ["doc2", "doc3", "doc1", "doc4"]
    graph_ppr_results = ["doc1", "doc4", "doc3", "doc2"]

    decay_weights = {
        "doc1": 0.95,  # Active working memory
        "doc2": 0.12,  # Stale, unreinforced note
        "doc3": 0.45,  # Moderate recency
        "doc4": 0.80   # Frequently visited
    }

    simulator = HybridRetrievalSimulator()
    final_ranking = simulator.fuse(dense_results, lexical_results, graph_ppr_results, decay_weights)

    print(f"{'Rank':<5} | {'Doc ID':<10} | {'Fused Hybrid Score':<20} | {'Memory Weight':<15}")
    print("-" * 70)
    for rank, (doc_id, score) in enumerate(final_ranking, 1):
        print(f"{rank:<5} | {doc_id:<10} | {score:<20.6f} | {decay_weights[doc_id]:<15.2f}")

    print("=" * 70)
    top_doc = final_ranking[0][0]
    print(f"Top Retrieved Document: {top_doc}")
    assert top_doc == "doc1", "Validation Failed: Multi-hop active document should rank #1"
    print("Verification: Hybrid fusion successfully balances multi-hop topology and cognitive memory gating.")

if __name__ == "__main__":
    run_hybrid_simulation()

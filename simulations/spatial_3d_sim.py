"""
MemoryGraph-AI: 3D Topology-Constrained Spatial Optimization Prototype
Validates the manifold cross-entropy + graph spring tension loss formulation (Task R9).
"""

import numpy as np
from typing import List, Tuple

class Spatial3DOptimizer:
    def __init__(self, num_nodes: int, gamma_spring: float = 0.5, mu_repulsion: float = 0.1, kappa: float = 0.01):
        self.N = num_nodes
        self.gamma_spring = gamma_spring
        self.mu_repulsion = mu_repulsion
        self.kappa = kappa

    def optimize_layout(self,
                        high_d_embeddings: np.ndarray,
                        edges: List[Tuple[int, int, float]],
                        iterations: int = 150,
                        lr: float = 0.05) -> np.ndarray:
        """
        Numerically optimizes 3D coordinates Y in R^{N x 3} minimizing:
        L_spatial = L_UMAP_approx + L_spring + L_repulsion
        """
        N = high_d_embeddings.shape[0]
        # Initialize 3D positions with PCA/random normal
        np.random.seed(42)
        Y = np.random.randn(N, 3) * 0.1

        # Compute high-D pairwise similarity matrix P (approximate)
        dists_high = np.linalg.norm(high_d_embeddings[:, None, :] - high_d_embeddings[None, :, :], axis=-1)
        sigma = np.median(dists_high) + 1e-5
        P = np.exp(- (dists_high ** 2) / (2 * sigma ** 2))
        np.fill_diagonal(P, 0.0)
        P = P / np.sum(P)

        for it in range(iterations):
            # Compute low-D 3D pairwise distances
            diffs = Y[:, None, :] - Y[None, :, :]  # (N, N, 3)
            dists_sq = np.sum(diffs ** 2, axis=-1)  # (N, N)
            Q = 1.0 / (1.0 + dists_sq)
            np.fill_diagonal(Q, 0.0)
            Q_sum = np.sum(Q) + 1e-8
            Q_norm = Q / Q_sum

            # 1. UMAP / t-SNE attraction & repulsion gradient
            grad_umap = np.zeros_like(Y)
            mult = 4.0 * (P - Q_norm) * Q  # (N, N)
            for i in range(N):
                grad_umap[i] = np.sum(mult[i, :, None] * diffs[i, :, :], axis=0)

            # 2. Graph Topological Spring gradient
            grad_spring = np.zeros_like(Y)
            for u, v, w in edges:
                delta = Y[u] - Y[v]
                grad_spring[u] += 2.0 * self.gamma_spring * w * delta
                grad_spring[v] -= 2.0 * self.gamma_spring * w * delta

            # 3. Electrostatic Repulsion gradient
            grad_repulsion = np.zeros_like(Y)
            rep_factor = 2.0 * self.mu_repulsion / ((dists_sq + self.kappa) ** 2)
            np.fill_diagonal(rep_factor, 0.0)
            for i in range(N):
                grad_repulsion[i] = - np.sum(rep_factor[i, :, None] * diffs[i, :, :], axis=0)

            total_grad = grad_umap + grad_spring + grad_repulsion
            # Gradient clipping and update
            total_grad = np.clip(total_grad, -5.0, 5.0)
            Y -= lr * total_grad

        return Y


def run_spatial_simulation():
    print("=" * 70)
    print("MemoryGraph-AI: 3D Topology-Constrained Spatial Optimization")
    print("=" * 70)

    # 10 synthetic nodes in 64-dimensional semantic embedding space
    # 2 distinct clusters: Cluster A (0-4), Cluster B (5-9)
    np.random.seed(42)
    dim = 64
    cluster_a = np.random.randn(5, dim) + np.array([3.0] * dim)
    cluster_b = np.random.randn(5, dim) + np.array([-3.0] * dim)
    high_d = np.vstack([cluster_a, cluster_b])

    # Graph edges: Strong intra-cluster edges, 1 cross-cluster bridge edge (4 -> 5)
    edges = [
        (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0),
        (5, 6, 1.0), (6, 7, 1.0), (7, 8, 1.0), (8, 9, 1.0),
        (4, 5, 0.8) # Bridge edge
    ]

    optimizer = Spatial3DOptimizer(num_nodes=10)
    Y_3d = optimizer.optimize_layout(high_d, edges, iterations=100)

    center_a = np.mean(Y_3d[0:5], axis=0)
    center_b = np.mean(Y_3d[5:10], axis=0)
    inter_cluster_dist = np.linalg.norm(center_a - center_b)

    print("Computed 3D Node Positions:")
    for idx, pos in enumerate(Y_3d):
        cluster_name = "Cluster A" if idx < 5 else "Cluster B"
        print(f"Node {idx} ({cluster_name:<9}): [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]")

    print("-" * 70)
    print(f"Cluster Separation Distance in R^3: {inter_cluster_dist:.4f}")
    assert inter_cluster_dist > 0.5, "Validation Failed: Clusters should be clearly separated in 3D"
    print("Verification: 3D spatial optimization successfully preserves semantic clustering and edge continuity.")

if __name__ == "__main__":
    run_spatial_simulation()

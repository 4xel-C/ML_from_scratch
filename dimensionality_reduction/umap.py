"""
UMAP (Uniform Manifold Approximation and Projection) implementation from scratch.

How to compute UMAP:

1) Compute the neighborhood graph with adpated weights
2) Randomly distribute the points in the reduced space
3) Optimize using the binary cross-entropy to rebuild the graph
"""


class UMAP:
    def __init__(
        self,
        n_neighbors: int,
        n_components: int,
        learning_rate: float,
        n_epochs: int,
        n_negative_sample: int,
    ): ...

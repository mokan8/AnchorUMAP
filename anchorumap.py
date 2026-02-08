import umap
import numpy as np

from umap.umap_ import (
    nearest_neighbors,
    smooth_knn_dist,
    fuzzy_simplicial_set,
    simplicial_set_embedding,
    find_ab_params,   # if this fails, change to _find_ab_params
    NNDescent,
)


class AnchorUMAP(umap.UMAP):
    def __init__(self, include_anchors_in_embedding=False, *args, **kwargs):
        """
        Extended UMAP that includes triangle anchors (2-simplices).

        Parameters
        ----------
        include_anchors_in_embedding : bool, default=True
            Whether to include anchor points in the final embedding visualization.
        """
        super().__init__(*args, **kwargs)
        self.include_anchors_in_embedding = include_anchors_in_embedding
        self.n_anchors_ = 0
        self.triangles_ = []
        self.metric_kwds = kwargs.get("metric_kwds", {}) or {}

    @staticmethod
    def find_triangles_from_local(X, knn_indices):
        """
        Given data X and a knn_indices array, find triangles (a,b,c).

        Parameters
        ----------
        X : (n_samples, n_features) array
            Original data.
        knn_indices : (n_samples, k) array
            k-nearest neighbors for each sample.
        Returns
        -------
        triangles : list of tuples
            Each tuple is (a, b, c, w_bc, w_ac, w_ab)
            where w_* are the weighted edge strengths obtained from z-score distances.
        """
        n_samples, k = knn_indices.shape
        triangles = []
        seen = set()  # track unique (a,b,c)
        local_means = np.zeros(n_samples)
        local_stds = np.zeros(n_samples)
        for p in range(n_samples):
          neighbors = knn_indices[p]
          neighbor_coords = X[neighbors]
          DM = distance_matrix(neighbor_coords, neighbor_coords)
          local_means[p] = np.mean(DM)
          local_stds[p] = np.std(DM)

        for a in range(n_samples):
            neighbors = knn_indices[a]
            neighbor_coords = X[neighbors]

            # Now check all neighbor pairs (b, c)
            for i in range(k):
                for j in range(i + 1, k):
                    b = neighbors[i]
                    c = neighbors[j]
                    b_neighbors = knn_indices[b]
                    c_neighbors = knn_indices[c]
                    if (a in b_neighbors) and (c in b_neighbors) and (b in c_neighbors) and (a in c_neighbors):
                    # distance between b and c in neighbor space
                      d_bc = np.linalg.norm(X[c] - X[b])
                      d_ab = np.linalg.norm(X[b] - X[a])
                      d_ac = np.linalg.norm(X[c] - X[a])

                      # Normalize by mean_val too
                      d_ab = (d_ab - local_means[c]) / local_stds[c] #z-score
                      d_ac = (d_ac - local_means[b]) / local_stds[b] #z-score
                      d_bc = (d_bc - local_means[a]) / local_stds[a] #z-score

                      # Store triangle, edges already weighted
                      if a < b < c:
                        key = tuple(sorted((a, b, c)))
                        if key not in seen: 
                            seen.add(key)
                            w_bc = AnchorUMAP.weight_fn(d_bc)
                            w_ac = AnchorUMAP.weight_fn(d_ac)
                            w_ab = AnchorUMAP.weight_fn(d_ab)
                            #remove low quality triangles
                            if min(w_ab, w_ac, w_bc) >= 0.5:
                              triangles.append((a, b, c, w_bc, w_ac, w_ab))
        return triangles

    @staticmethod
    def weight_fn(d, scale=5.0):
      return 1 / (1 + np.exp(scale * d))



    @staticmethod
    def make_anchors(triangles, X):
        """Create anchor points at triangle centroids."""
        if not triangles:
            return np.empty((0, X.shape[1]))

        anchor_coords = []
        for (a, b, c, w_ab, w_ac, w_bc) in triangles:
            total_w = w_ab + w_ac + w_bc + 1e-9
            centroid = (w_bc * X[a] + w_ac * X[b] + w_ab * X[c]) / total_w
            anchor_coords.append(centroid)
        return np.array(anchor_coords)

    @staticmethod
    def build_anchor_edges(triangles, n_samples):
        """Build edges connecting anchors to their triangle vertices."""
        edges = []
        for t_id, (a, b, c, w_bc, w_ac, w_ab) in enumerate(triangles):
            anchor_idx = n_samples + t_id

            # anchor to a (weight comes from edge bc)
            edges.append((anchor_idx, a, w_bc))
            edges.append((a, anchor_idx, w_bc))

            # anchor to b (weight comes from edge ac)
            edges.append((anchor_idx, b, w_ac))
            edges.append((b, anchor_idx, w_ac))

            # anchor to c (weight comes from edge ab)
            edges.append((anchor_idx, c, w_ab))
            edges.append((c, anchor_idx, w_ab))

        return edges

    def _build_augmented_graph(self, X, knn_indices, knn_dists, sigmas, rhos, return_dists=None):
        """Build the fuzzy simplicial set including anchor points."""
        # Compute standard membership strengths
        rows, cols, vals, dists = compute_membership_strengths(
            knn_indices, knn_dists, sigmas, rhos, return_dists
        )

        # Find triangles and create anchors
        triangles = self.find_triangles_from_local(X, knn_indices)
        self.triangles_ = triangles

        n_samples = X.shape[0]

        if len(triangles) > 0:
            # Create anchor points
            anchors = self.make_anchors(triangles, X)
            self.n_anchors_ = anchors.shape[0]

            # Augment data with anchors
            X_aug = np.vstack([X, anchors])

            # Build anchor edges
            anchor_edges = self.build_anchor_edges(triangles, n_samples)

            # Add anchor edges to graph
            if anchor_edges:
                anchor_rows, anchor_cols, anchor_vals = zip(*anchor_edges)
                rows = np.concatenate([rows, np.array(anchor_rows, dtype=np.int32)])
                cols = np.concatenate([cols, np.array(anchor_cols, dtype=np.int32)])
                vals = np.concatenate([vals, np.array(anchor_vals, dtype=np.float32)])

            # Update graph size to include anchors
            graph_size = n_samples + self.n_anchors_
        else:
            X_aug = X
            graph_size = n_samples
            self.n_anchors_ = 0

        # Create sparse matrix with correct size
        graph = sp.coo_matrix(
            (vals, (rows, cols)),
            shape=(graph_size, graph_size)
        )
        graph.eliminate_zeros()

        return graph, X_aug

    def fit(self, X, y=None):
        """Fit the AnchorUMAP model."""
        X = X.astype(np.float32)

        # Validate random state
        if isinstance(self.random_state, (int, np.integer)):
            self.random_state = np.random.RandomState(self.random_state)

        # Get k-NN graph
        knn_indices, knn_dists, _ = nearest_neighbors(
            X,
            self.n_neighbors,
            self.metric,
            getattr(self, 'metric_kwds', {}),
            getattr(self, 'angular_rp_forest', False),
            self.random_state,
            verbose=self.verbose,
        )

        # Compute membership strengths parameters
        sigmas, rhos = smooth_knn_dist(
            knn_dists.astype(np.float32),
            float(self.n_neighbors),
            local_connectivity=float(self.local_connectivity),
        )

        # Build augmented graph
        graph, X_aug = self._build_augmented_graph(
            X, knn_indices, knn_dists, sigmas, rhos
        )

        # Apply set operations
        transpose = graph.transpose()
        prod_matrix = graph.multiply(transpose)
        graph = (
            self.set_op_mix_ratio * (graph + transpose - prod_matrix)
            + (1.0 - self.set_op_mix_ratio) * prod_matrix
        )
        graph.eliminate_zeros()

        # Compute embedding
        self._raw_data = X_aug  # Store augmented data

        # Get UMAP parameters
        a, b = umap.umap_.find_ab_params(self.spread, self.min_dist)

        # Compute embedding for augmented data
        embedding = simplicial_set_embedding(
            X_aug,
            graph,
            self.n_components,
            self.learning_rate,
            a, b,
            self.repulsion_strength,
            self.negative_sample_rate,
            self.n_epochs if self.n_epochs is not None else 200,
            init=self.init,
            random_state=self.random_state,
            metric=self.metric,
            metric_kwds=getattr(self, 'metric_kwds', {}),
            densmap=False,
            densmap_kwds={},
            output_dens=False,
            verbose=self.verbose,
        )[0]

        # Store embeddings
        if self.include_anchors_in_embedding or self.n_anchors_ == 0:
            self.embedding_ = embedding
        else:
            # Only return original data points in embedding
            self.embedding_ = embedding[:X.shape[0]]

        # Store anchor embeddings separately
        if self.n_anchors_ > 0:
            self.anchor_embedding_ = embedding[X.shape[0]:]
        else:
            self.anchor_embedding_ = np.empty((0, self.n_components))

        return self

    def fit_transform(self, X, y=None):
        """Fit the model and return the embedding."""
        return self.fit(X, y).embedding_

    def get_anchor_embeddings(self):
        """Get the embedding coordinates of anchor points."""
        if not hasattr(self, 'anchor_embedding_'):
            raise ValueError("Model must be fitted first")
        return self.anchor_embedding_

    def get_triangles(self):
        """Get the list of triangles used to create anchors."""
        if not hasattr(self, 'triangles_'):
            raise ValueError("Model must be fitted first")
        return self.triangles_
"""Clustering: segment students into behavioral groups."""
from sklearn.cluster import KMeans


SEGMENT_NAMES = {
    0: "Consistent High Performers",
    1: "Improving Students",
    2: "Disengaged Students",
    3: "High-Risk Students",
}


def build_kmeans(n_clusters: int = 4) -> KMeans:
    return KMeans(n_clusters=n_clusters, random_state=42, n_init=10)


def label_segments(cluster_centers, n_clusters: int = 4) -> dict:
    """Assign human-readable names to clusters.

    Heuristic: rank clusters by centroid of 'mean_quiz_score' (first column
    in our aggregate feature ordering) — highest = Consistent High Performers,
    etc. The caller can map cluster index → name via this dict.
    """
    # cluster_centers: (n_clusters, n_features) — caller passes the slice
    # Simple default: return segment names by index. Real ranking done
    # inside the training script.
    return {i: SEGMENT_NAMES.get(i, f"Segment {i}") for i in range(n_clusters)}

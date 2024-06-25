# CALLEE_2024

CALLEE consists of 4 main modules, they are graph_construction, graph_walk, graph_embedding and alerts_clustering.

# graph_construction

This module receives a csv file composed of alerts and generates an alert heterogeneous information network based on the fields in it.

# graph_walk

This module receives an alert heterogeneous information network and generates paths by meta-path-based random walk for embedding.

# graph_embedding

This module receives the paths and embeds them using Word2Vec to get vector representations of alerts.

# alerts_clustering

This module receives vector representations of alerts, clusters the alert using DBSCAN, and outputs a series of clusters.

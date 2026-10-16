"""Physical directed graph statistics used by the revised manuscript.

An active binary relation or hyperedge transfers from each source to each
target in one hop.  The depth is one plus the largest *finite shortest*
directed distance, rather than a longest path (which is undefined on cycles).
"""

from collections import deque


def directed_adjacency(graph):
    """Return the physical adjacency, retaining isolated nodes."""
    node_ids = [node["id"] for node in graph.get("nodes", [])]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Physical node IDs must be unique within a graph")
    adjacency = {node_id: set() for node_id in node_ids}
    for relation in graph.get("relations", []):
        sources, targets = relation.get("sources", []), relation.get("targets", [])
        if not sources or not targets:
            raise ValueError("Every active relation needs source and target members")
        for endpoint in [*sources, *targets]:
            if endpoint not in adjacency:
                raise ValueError(f"Unknown active relation endpoint: {endpoint}")
        for source in sources:
            adjacency[source].update(targets)
    return adjacency


def shortest_distances(adjacency, source):
    """All finite shortest directed distances, including source distance 0."""
    distances = {source: 0}
    pending = deque([source])
    while pending:
        node = pending.popleft()
        for target in sorted(adjacency[node]):
            if target not in distances:
                distances[target] = distances[node] + 1
                pending.append(target)
    return distances


def graph_statistics(graph):
    """Compute exact physical hop/depth statistics from active relations only."""
    adjacency = directed_adjacency(graph)
    maximum = 0
    total = 0
    finite_pairs = 0
    for source in sorted(adjacency):
        distances = shortest_distances(adjacency, source)
        maximum = max(maximum, max(distances.values()))
        total += sum(distances.values())
        finite_pairs += len(distances)
    relations = graph.get("relations", [])
    return {
        "graph_id": graph.get("graph_id"),
        "node_count": len(adjacency),
        "relation_count": len(relations),
        "binary_relation_count": sum(r.get("kind") not in {"CTRL", "DATA"} for r in relations),
        "hyperedge_count": sum(r.get("kind") in {"CTRL", "DATA"} for r in relations),
        "physical_adjacency_count": sum(map(len, adjacency.values())),
        "depth": maximum + 1 if adjacency else 0,
        "maximum_finite_shortest_distance": maximum if adjacency else None,
        "finite_ordered_pairs_including_diagonal": finite_pairs,
        "mean_finite_shortest_distance_including_diagonal": total / finite_pairs if finite_pairs else None,
        "definition": "1 + maximum finite shortest directed distance; empty graph depth 0; hyperedge transfer 1 hop",
    }


def graph_depth(graph):
    """Return the revised physical directed depth."""
    return graph_statistics(graph)["depth"]

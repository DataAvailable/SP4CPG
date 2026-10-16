import pytest

from sp4cpg.statistics import graph_depth, graph_statistics


def make_graph(nodes, pairs):
    return {"graph_id": "statistics", "nodes": [{"id": n} for n in nodes],
            "relations": [{"kind": "CFG", "sources": [s], "targets": [t]} for s, t in pairs]}


def test_empty_and_isolated_graph_depth():
    assert graph_depth(make_graph([], [])) == 0
    assert graph_depth(make_graph(["a"], [])) == 1
    stats = graph_statistics(make_graph(["a", "b"], []))
    assert stats["finite_ordered_pairs_including_diagonal"] == 2
    assert stats["mean_finite_shortest_distance_including_diagonal"] == 0


def test_depth_uses_shortest_paths_and_excludes_unreachable_pairs():
    graph = make_graph(["a", "b", "c", "d", "isolated"], [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")])
    assert graph_depth(graph) == 3  # a->c or b->d, not longest a->b->c->d
    graph["relations"].append({"kind": "AST", "sources": ["d"], "targets": ["a"]})
    assert graph_depth(graph) == 4  # c->d->a->b


def test_hyperedges_count_as_one_directed_hop_and_packing_preserves_depth():
    binary = make_graph(["a", "b", "x", "y", "z"], [(s, t) for s in ("a", "b") for t in ("x", "y")] + [("y", "z")])
    packed = make_graph(["a", "b", "x", "y", "z"], [("y", "z")])
    packed["relations"].append({"kind": "DATA", "sources": ["a", "b"], "targets": ["x", "y"]})
    assert graph_depth(binary) == graph_depth(packed) == 3
    assert graph_statistics(binary)["physical_adjacency_count"] == graph_statistics(packed)["physical_adjacency_count"] == 5


def test_cycles_are_finite_and_no_reverse_edges_are_invented():
    assert graph_depth(make_graph(["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "a")])) == 3
    stats = graph_statistics(make_graph(["a", "b"], [("a", "b")]))
    assert stats["finite_ordered_pairs_including_diagonal"] == 3
    assert stats["mean_finite_shortest_distance_including_diagonal"] == pytest.approx(1 / 3)


def test_absent_active_member_is_rejected():
    with pytest.raises(ValueError, match="Unknown active relation endpoint"):
        graph_depth(make_graph(["a"], [("a", "missing")]))

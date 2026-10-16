import math

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch

from models.taf_hgcn import TAFHGCN, TAFHGCNLayer, normalized_propagate
from models.standard_hgnn import HGNN
from sp4cpg.data import GraphData


def make_data(x, sources, targets, kinds=None):
    pairs, roles = [], []
    for relation, (source, target) in enumerate(zip(sources, targets)):
        for members, role in ((source, 0), (target, 1)):
            for node in members:
                pairs.append((node, relation))
                roles.append(role)
    incidence = (torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs
                 else torch.empty((2, 0), dtype=torch.long))
    return GraphData(x=torch.tensor(x, dtype=torch.float32).reshape(len(x), -1),
                     hyperedge_index=incidence,
                     incidence_role=torch.tensor(roles, dtype=torch.long),
                     relation_types=torch.tensor(kinds or [1] * len(sources), dtype=torch.long),
                     node_types=torch.zeros(len(x), dtype=torch.long),
                     edge_index=torch.empty((2, 0), dtype=torch.long),
                     num_nodes=len(x), num_hyperedges=len(sources), schema_version=2)


def test_directed_chain_has_no_backward_or_self_message():
    data = make_data([1., 2., 7.], [[0], [1]], [[1], [2]])
    actual = normalized_propagate(data.x, data.hyperedge_index, data.incidence_role,
                                  torch.ones(2))
    torch.testing.assert_close(actual[:, 0], torch.tensor([0., 1., 2.]))
    changed = data.x.clone()
    changed[2] = 9999.
    torch.testing.assert_close(actual, normalized_propagate(
        changed, data.hyperedge_index, data.incidence_role, torch.ones(2)))


def test_weighted_degrees_and_source_cardinality_match_hand_calculation():
    data = make_data([2., 6., 100., 999.], [[0, 1], [1]], [[2], [3]])
    weights = torch.tensor([2., 1.], requires_grad=True)
    actual = normalized_propagate(data.x, data.hyperedge_index, data.incidence_role, weights)
    expected = torch.tensor([[0.], [0.], [1. + math.sqrt(6)], [2 * math.sqrt(3)]])
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(weights.grad).all(), "zero-degree backward must stay finite"


def test_one_to_many_distributes_to_targets_only():
    data = make_data([3., 8., 9.], [[0]], [[1, 2]])
    actual = normalized_propagate(data.x, data.hyperedge_index, data.incidence_role,
                                  torch.ones(1))
    torch.testing.assert_close(actual[:, 0], torch.tensor([0., 3., 3.]))


def test_layer_averages_exactly_hops_one_to_k_and_no_zero_hop():
    data = make_data([1., 2., 7.], [[0], [1]], [[1], [2]])
    layer = TAFHGCNLayer(1, 1, hop_size=2, use_types=False)
    with torch.no_grad():
        layer.linear.weight.fill_(1)
    torch.testing.assert_close(layer(data.x, data)[:, 0], torch.tensor([0., .5, 1.5]))
    layer.hop_size = 1
    torch.testing.assert_close(layer(data.x, data)[:, 0], torch.tensor([0., 1., 2.]))


def test_depth_is_independent_of_hop_size_without_residuals():
    data = make_data([1., 2., 7.], [[0], [1]], [[1], [2]])
    model = TAFHGCN(0., 1, 1, num_layers=3, hop_size=1, use_types=False)
    with torch.no_grad():
        for layer in model.layers:
            layer.linear.weight.fill_(1)
    torch.testing.assert_close(model.encode_nodes(data), torch.zeros((3, 1)))
    assert all(layer.linear.bias is None for layer in model.layers)


def test_kind_softmax_is_over_all_six_kinds_not_observed_relations():
    layer = TAFHGCNLayer(1, 1, type_dim=1, use_types=True)
    layer.type_score = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        layer.type_score.weight.fill_(1)
    embeddings = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    weights = layer.kind_weights(embeddings)
    torch.testing.assert_close(weights, torch.softmax(torch.arange(6, dtype=torch.float64), 0))
    assert weights.numel() == 6
    # An absent type remains in the denominator.
    assert weights[0].item() < .01


def test_standard_hgnn_uses_union_cardinality_unit_weights_and_ignores_types():
    data = make_data([2., 6., 100.], [[0]], [[1]])
    model = HGNN(0., 1, 1, num_layers=1)
    with torch.no_grad():
        model.layers[0].weight.fill_(1)
    torch.testing.assert_close(model.encode_nodes(data)[:, 0], torch.tensor([4., 4., 0.]))
    data.node_types.fill_(16)
    data.relation_types.fill_(5)
    torch.testing.assert_close(model.encode_nodes(data)[:, 0], torch.tensor([4., 4., 0.]))
    assert not any("embedding" in name for name, _ in model.named_parameters())
    with pytest.raises(ValueError, match="hop_size=1"):
        HGNN(hop_size=3)


def test_undirected_ablation_retains_union_once_for_binary_self_loop():
    data = make_data([3.], [[0]], [[0]])
    actual = normalized_propagate(data.x, data.hyperedge_index, data.incidence_role,
                                  torch.ones(1), directed=False)
    torch.testing.assert_close(actual, torch.tensor([[3.]]))


def test_zero_relations_zero_incoming_rows_and_mean_pooling():
    data = make_data([3., 5.], [], [])
    model = TAFHGCN(0., 1, 2, num_layers=2, use_types=True)
    assert torch.equal(model.encode_nodes(data), torch.zeros((2, 2)))
    assert torch.equal(model.encode_graph(data), torch.zeros((1, 2)))
    logits = model(data)
    assert logits.shape == (1, 2)
    logits.sum().backward()
    assert torch.isfinite(model.classifier.bias.grad).all()
    chain = make_data([1., 2., 7.], [[0], [1]], [[1], [2]])
    mean_model = TAFHGCN(0., 1, 1, num_layers=1, hop_size=2, use_types=False)
    with torch.no_grad():
        mean_model.layers[0].linear.weight.fill_(1)
    torch.testing.assert_close(mean_model.encode_graph(chain), torch.tensor([[2./3.]]))


@pytest.mark.parametrize("model_class,hop", [(TAFHGCN, 2), (HGNN, 1)])
def test_batched_outputs_equal_independent_graph_outputs(model_class, hop):
    torch.manual_seed(45)
    first = make_data([1., 2., 3.], [[0], [1]], [[1], [2]])
    # Four relations on two nodes exercises relation offsets != node offsets.
    second = make_data([4., 6.], [[0], [1], [0], [1]], [[1], [0], [1], [0]], [0, 1, 4, 5])
    model = model_class(0., 1, 3, num_layers=2, hop_size=hop)
    model.eval()
    expected = torch.cat((model(first), model(second)), dim=0)
    actual = model(Batch.from_data_list([first, second]))
    torch.testing.assert_close(actual, expected)


def test_types_are_learned_and_relation_weight_gradients_are_finite():
    torch.manual_seed(11)
    data = make_data([1., 3., 2.], [[0], [0], [1]], [[1], [2], [2]], [0, 1, 5])
    data.node_types = torch.tensor([0, 7, 11])
    model = TAFHGCN(0., 1, 8, num_layers=1, hop_size=1, type_dim=4)
    model(data).square().sum().backward()
    assert model.node_type_embedding.weight.grad is not None
    assert model.relation_type_embedding.weight.grad is not None
    assert torch.isfinite(model.relation_type_embedding.weight.grad).all()
    assert not hasattr(TAFHGCN(use_types=False), "node_type_embedding")


@pytest.mark.parametrize("tiny_weight", [1e-26, 1e-40])
def test_tiny_positive_relation_weight_has_finite_cancelled_gradient(tiny_weight):
    data = make_data([1., 0.], [[0]], [[1]])
    weights = torch.tensor([tiny_weight], dtype=torch.float32, requires_grad=True)
    output = normalized_propagate(data.x, data.hyperedge_index, data.incidence_role, weights)
    torch.testing.assert_close(output[:, 0], torch.tensor([0., 1.]))
    output.sum().backward()
    assert torch.isfinite(weights.grad).all()
    torch.testing.assert_close(weights.grad, torch.zeros(1), atol=1e-6, rtol=0.)


def test_extreme_six_kind_softmax_retains_positive_weights_and_finite_gradients():
    layer = TAFHGCNLayer(1, 1, type_dim=1, hop_size=1)
    layer.type_score = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        layer.type_score.weight.fill_(1)
        layer.linear.weight.fill_(1)
    embeddings = torch.tensor([[-100.], [-60.], [0.], [1.], [2.], [3.]], requires_grad=True)
    weights = layer.kind_weights(embeddings)
    assert (weights > 0).all()
    data = make_data([1., 2., 0.], [[0], [0], [1]], [[1], [2], [2]], [0, 1, 5])
    output = layer(data.x, data, embeddings)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert torch.isfinite(embeddings.grad).all()
    assert torch.isfinite(layer.type_score.weight.grad).all()


@pytest.mark.parametrize("arguments", [{"num_layers": 0}, {"hop_size": 0}, {"dropout": 1.}])
def test_invalid_taf_settings_are_explicit(arguments):
    with pytest.raises(ValueError):
        TAFHGCN(**arguments)

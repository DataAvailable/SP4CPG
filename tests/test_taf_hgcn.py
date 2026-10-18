import copy
import math
from types import SimpleNamespace

import pytest
import torch

from models.hgcn_models import FlowSensitiveLayer, HGCN, TAFHGCN, directed_propagation
from models.hgnn_models import HGCN as SharedHGCN


def graph(features, relations, kinds=None, node_types=None):
    x = torch.as_tensor(features, dtype=torch.float32)
    members, roles = [], []
    for relation, (sources, targets) in enumerate(relations):
        for nodes, role in ((sources, 0), (targets, 1)):
            for node in nodes:
                members.append((node, relation))
                roles.append(role)
    incidence = torch.tensor(members, dtype=torch.long).t().contiguous() if members else torch.empty((2, 0), dtype=torch.long)
    return SimpleNamespace(x=x, hyperedge_index=incidence, incidence_role=torch.tensor(roles, dtype=torch.long),
                           relation_types=torch.tensor(kinds or [3] * len(relations), dtype=torch.long),
                           node_types=torch.tensor(node_types or [11] * len(features), dtype=torch.long),
                           num_hyperedges=len(relations), schema_version=2,
                           edge_index=torch.empty((2, 0), dtype=torch.long))


def propagate(data, scores=None):
    scores = torch.zeros(6, dtype=torch.float64) if scores is None else scores
    return directed_propagation(data.x, data.hyperedge_index, data.incidence_role, data.relation_types, scores)


def scalar_model(layers=1, hops=1):
    model = TAFHGCN(0, in_channels=1, hidden_channels=1, type_dim=1, num_layers=layers, hop_size=hops).eval()
    with torch.no_grad():
        model.node_type_embedding.weight.zero_()
        for i, layer in enumerate(model.layers):
            layer.transform.weight.fill_(1)
            if i == 0:
                layer.transform.weight[:, 1] = 0
        model.classifier.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        model.classifier.bias.zero_()
    return model


def test_weighted_source_target_degrees_and_source_cardinality():
    data = graph([[2], [4], [0], [0]], [([0, 1], [2]), ([0], [2, 3])], kinds=[4, 5])
    scores = torch.tensor([0, 0, 0, 0, math.log(2), 0], dtype=torch.float64)
    actual = propagate(data, scores)
    expected = torch.tensor([[0], [0], [(2 * (2 / math.sqrt(3) + 4 / math.sqrt(2)) / 2 + 2 / math.sqrt(3)) / math.sqrt(3)], [2 / math.sqrt(3)]])
    torch.testing.assert_close(actual, expected)


def test_direction_zero_degrees_and_no_self_feature_channel():
    data = graph([[3], [7], [11]], [([0], [1])])
    torch.testing.assert_close(propagate(data), torch.tensor([[0.0], [3.0], [0.0]]))
    reversed_data = graph([[3], [7], [11]], [([1], [0])])
    torch.testing.assert_close(propagate(reversed_data), torch.tensor([[7.0], [0.0], [0.0]]))
    empty = graph([[3], [7]], [])
    torch.testing.assert_close(propagate(empty), torch.zeros(2, 1))
    model = scalar_model(layers=2)
    torch.testing.assert_close(model.encode_nodes(data), torch.zeros(3, 1))


def test_positive_hop_mean_is_distinct_from_trainable_layer_count():
    data = graph([[1], [0], [0], [0]], [([0], [1]), ([1], [2]), ([2], [3])])
    model = scalar_model(hops=3)
    torch.testing.assert_close(model.encode_nodes(data), torch.tensor([[0.0], [1 / 3], [1 / 3], [1 / 3]]))
    torch.testing.assert_close(model(data), torch.tensor([[0.25, -0.25]]))
    two_layers = scalar_model(layers=2, hops=1)
    torch.testing.assert_close(two_layers.encode_nodes(data), torch.tensor([[0.0], [0.0], [1.0], [0.0]]))
    assert len(two_layers.layers) == 2 and len(model.layers) == 1


def test_node_type_embedding_is_concatenated_with_content_features():
    data = graph([[0], [0]], [([0], [1])], node_types=[5, 6])
    model = scalar_model()
    with torch.no_grad():
        model.node_type_embedding.weight[5, 0] = 2
        model.layers[0].transform.weight.copy_(torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(model(data), torch.tensor([[1.0, -1.0]]))


def test_softmax_normalizes_six_kind_scores_and_same_kind_weights_are_shared():
    model = scalar_model()
    layer = model.layers[0]
    with torch.no_grad():
        model.relation_type_embedding.weight.copy_(torch.arange(1.0, 7.0).view(6, 1))
        layer.type_mlp[0].weight.fill_(1)
        layer.type_mlp[0].bias.zero_()
        layer.type_mlp[2].weight.fill_(1)
        layer.type_mlp[2].bias.zero_()
    weights = layer.type_weights(model.relation_type_embedding.weight)
    torch.testing.assert_close(weights, torch.softmax(torch.arange(1.0, 7.0, dtype=torch.float64), dim=0))
    assert weights.shape == (6,) and weights.dtype == torch.float64
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    data = graph([[1], [2], [0]], [([0], [2]), ([1], [2])], kinds=[3, 3])
    torch.testing.assert_close(propagate(data, layer.type_scores(model.relation_type_embedding.weight)),
                               propagate(data, torch.zeros(6)))


def test_extreme_scores_have_finite_nonzero_gradients_without_clipping():
    data = graph([[2], [4], [0], [0]], [([0, 1], [2]), ([0], [2, 3])], kinds=[4, 5])
    scores = torch.tensor([0, 0, 0, 0, -1000.0, -1001.0], dtype=torch.float64, requires_grad=True)
    data.x.requires_grad_()
    result = propagate(data, scores)
    shifted = scores.detach().clone()
    shifted[4] += 1000
    shifted[5] += 1000
    torch.testing.assert_close(result, propagate(data, shifted))
    result.square().sum().backward()
    assert torch.isfinite(scores.grad).all() and torch.isfinite(data.x.grad).all()
    assert scores.grad[4].abs() > 1e-6 and scores.grad[5].abs() > 1e-6
    assert torch.equal(scores.grad[[0, 1, 2, 3]], torch.zeros(4, dtype=torch.float64))


def test_all_used_types_can_have_scores_far_below_unused_types():
    data = graph([[5], [0], [9]], [([0], [1])])
    scores = torch.tensor([1000, 1000, 1000, -1000, 1000, 1000], dtype=torch.float64, requires_grad=True)
    result = propagate(data, scores)
    torch.testing.assert_close(result, torch.tensor([[0.0], [5.0], [0.0]]))
    result.sum().backward()
    assert torch.isfinite(scores.grad).all()


def test_unconnected_large_kind_score_preserves_local_type_differences():
    data = graph([[2], [4], [0]], [([0], [2]), ([1], [2])], kinds=[3, 5])
    layer = FlowSensitiveLayer(1, 1, 1, 1).double()
    embeddings = torch.tensor([1e30, 0, 0, -1000, 0, -1001], dtype=torch.float64).view(6, 1)
    with torch.no_grad():
        layer.transform.weight.fill_(1)
        layer.type_mlp[0].weight.fill_(1)
        layer.type_mlp[0].bias.fill_(2000)
        layer.type_mlp[2].weight.fill_(1)
        layer.type_mlp[2].bias.fill_(-2000)
    expected_target = (2 + 4 * math.exp(-0.5)) / math.sqrt(1 + math.exp(-1))
    expected = torch.tensor([[0], [0], [expected_target]], dtype=torch.float64)
    actual = layer(data.x.double(), data.hyperedge_index, data.incidence_role,
                   data.relation_types, embeddings)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert abs(actual[2].item() - 6 / math.sqrt(2)) > 0.1


def test_incident_factorization_matches_dense_matrix_values_and_gradients():
    data = graph([[2, 3], [4, 5], [1, 0], [0, 2]],
                 [([0, 1], [2]), ([0], [2, 3]), ([2], [1])], kinds=[4, 5, 3])
    x = data.x.double().requires_grad_()
    scores = torch.tensor([0.2, -0.3, 0.4, 0.7, -0.2, 0.1], dtype=torch.float64, requires_grad=True)
    source = torch.zeros(4, 3, dtype=torch.float64)
    target = torch.zeros_like(source)
    for index, role in zip(data.hyperedge_index.t(), data.incidence_role):
        (source if int(role) == 0 else target)[index[0], index[1]] = 1
    weights = torch.softmax(scores, dim=0)[data.relation_types]
    source_degree, target_degree = source @ weights, target @ weights
    def inverse_root(degree):
        positive = degree > 0
        return torch.where(positive, degree, torch.ones_like(degree)).rsqrt() * positive
    matrix = (torch.diag(inverse_root(target_degree)) @ target @ torch.diag(weights)
              @ torch.diag(source.sum(dim=0).reciprocal()) @ source.t()
              @ torch.diag(inverse_root(source_degree)))
    expected = matrix @ x
    actual = directed_propagation(x, data.hyperedge_index, data.incidence_role, data.relation_types, scores)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    target_gradient = torch.arange(8, dtype=torch.float64).view(4, 2)
    expected_gradients = torch.autograd.grad((expected * target_gradient).sum(), (x, scores), retain_graph=True)
    actual_gradients = torch.autograd.grad((actual * target_gradient).sum(), (x, scores))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-11, rtol=1e-11)


def test_batched_graph_outputs_equal_individual_graph_outputs():
    torch.manual_seed(9)
    left = graph([[1, 2], [3, 4]], [([0], [1])])
    right = graph([[5, 6], [7, 8], [9, 10]], [([0], [1]), ([1], [2])], kinds=[1, 3])
    combined = SimpleNamespace(x=torch.cat((left.x, right.x)),
                              hyperedge_index=torch.cat((left.hyperedge_index, right.hyperedge_index + torch.tensor([[2], [1]])), dim=1),
                              incidence_role=torch.cat((left.incidence_role, right.incidence_role)),
                              relation_types=torch.cat((left.relation_types, right.relation_types)),
                              node_types=torch.cat((left.node_types, right.node_types)),
                              num_hyperedges=torch.tensor([1, 2]), schema_version=torch.tensor([2, 2]),
                              batch=torch.tensor([0, 0, 1, 1, 1]))
    model = TAFHGCN(0, in_channels=2, hidden_channels=3, type_dim=2, num_layers=2, hop_size=2).eval()
    torch.testing.assert_close(model(combined), torch.cat((model(left), model(right))), atol=1e-6, rtol=1e-6)
    bad = copy.deepcopy(combined)
    bad.hyperedge_index[0, 1] = 2
    with pytest.raises(ValueError, match="different graphs"):
        model(bad)


@pytest.mark.parametrize("field", ["hyperedge_index", "incidence_role", "node_types", "relation_types", "schema_version"])
def test_missing_direction_or_types_are_rejected(field):
    data = graph([[1], [2]], [([0], [1])])
    delattr(data, field)
    with pytest.raises(ValueError, match="required"):
        scalar_model()(data)


def test_edge_index_does_not_supply_incidence():
    data = graph([[1], [2]], [([0], [1])])
    model = scalar_model()
    expected = model(data)
    data.edge_index = torch.tensor([[999], [-10]])
    torch.testing.assert_close(model(data), expected)


@pytest.mark.parametrize("field,value", [("schema_version", 1), ("node_types", torch.tensor([17, 0])),
                                         ("relation_types", torch.tensor([6])), ("incidence_role", torch.tensor([0, 0])),
                                         ("num_hyperedges", 2)])
def test_invalid_metadata_is_rejected(field, value):
    data = graph([[1], [2]], [([0], [1])])
    setattr(data, field, value)
    with pytest.raises(ValueError):
        scalar_model()(data)


def test_duplicate_memberships_and_cross_role_hyperedge_members_are_rejected():
    data = graph([[1], [2]], [([0, 0], [1])])
    with pytest.raises(ValueError, match="Duplicate"):
        scalar_model()(data)
    data = graph([[1], [2]], [([0], [0, 1])], kinds=[5])
    with pytest.raises(ValueError, match="disjoint"):
        scalar_model()(data)
    data = graph([[1], [2], [3]], [([0, 1], [2])], kinds=[3])
    with pytest.raises(ValueError, match="Binary relation"):
        scalar_model()(data)


def test_backward_reaches_both_embedding_tables_and_layer_parameters():
    torch.manual_seed(4)
    data = graph([[2], [4], [0], [0]], [([0, 1], [2]), ([0], [2, 3])], kinds=[4, 5])
    model = TAFHGCN(0, in_channels=1, hidden_channels=4, type_dim=3, num_layers=1, hop_size=1)
    model(data).square().sum().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert model.node_type_embedding.weight.grad.abs().sum() > 0
    assert model.relation_type_embedding.weight.grad.abs().sum() > 0


def test_public_entrypoints_share_parameters_and_reject_other_operators():
    assert HGCN is TAFHGCN and SharedHGCN is TAFHGCN
    assert HGCN(0.4, in_channels=768, hidden_channels=512).num_layers == 3
    for option in ("use_residual", "use_attention", "use_layer_norm"):
        with pytest.raises(ValueError, match=option):
            TAFHGCN(**{option: True})
    for option in ("in_channels", "hidden_channels", "num_layers", "hop_size", "type_dim"):
        with pytest.raises(ValueError, match=option):
            TAFHGCN(**{option: 0})

# Graph schema and preservation boundary

Schema version 2 is a JSON object with `graph_id`, `nodes`, `relations`,
`dictionary` and `metadata`. All node and relation IDs are strings. JSON is the
canonical format. New DOT files embed the complete object in a `sp4cpg_json`
graph attribute so dictionary entries, parallel relations and bindings survive
round trips. Old binary Joern DOT can be imported. Old comma-encoded hyperedges
without source/target/label associations cannot be reconstructed safely and are
rejected; rebuild from their original CPGs.

## Physical topology and logical entries

```json
{
  "schema_version": 2,
  "graph_id": "sample-1",
  "nodes": [
    {"id": "a", "kind": "CALL", "code": "x=5", "attributes": {}, "entries": ["a"]},
    {"id": "b", "kind": "RETURN", "code": "return x", "attributes": {}, "entries": ["b"]}
  ],
  "relations": [{
    "id": "r1", "kind": "DDG", "sources": ["a"], "targets": ["b"],
    "bindings": [{"source": "a", "target": "b", "kind": "DDG", "label": "x",
                  "entity": "", "context": "", "attributes": {}, "path": []}],
    "attributes": {}
  }],
  "dictionary": {},
  "metadata": {"representation": "CPG"}
}
```

`normalize_graph()` fills original dictionary entries for newly imported binary
CPGs. Each entry preserves original kind, code and attributes, plus a current
`representative`, optional `carrier`, `status` and applied `rule`. Status is
`retained`, `merged`, `carried` or `removed`. Removed occurrences carried by a
retained expression keep their own logical identities. Shared declaration
containers keep separate entries and do not make their logical ports interchangeable.

Relations have unique IDs even for equal memberships. The six kinds are AST,
CFG, CDG, DDG, CTRL and DATA. CTRL packs existing CFG edges; DATA packs existing
DDG edges. Bindings retain original logical endpoints, original relation kind,
raw labels, entity/variable IDs, control context and original attributes. CFG
bypasses additionally retain path and per-step provenance. Data annotations may
carry an explicit source constant; its raw dependency label remains intact.
Structured entity/context attributes are canonicalized without dropping fields.

Every physical source-target pair of a hyperedge needs an actual binding
witness. Data decoding restores recorded source-target-label associations;
it does not expand `S × T × L`. Construction history cannot substitute for
missing active output relations.

## Rule-conditioned pruning

No rule uses target/vulnerability labels. Missing facts mean **retain**.

1. Rule 1 requires the redundant BLOCK → assignment → allocation pattern,
   content redundancy and a checkable surviving carrier. The current conservative
   implementation requires AST-only incident edges and
   `attributes.pruning.rule1 = {"redundant": true, "carrier": "adjacent-original-id"}`.
   This is stricter than treating a node-kind match as sufficient evidence.
2. Rule 2 requires an output-only debugging call with resolved argument/resource
   validity, unused result, no independent DDG/CDG role and a safe same-context
   linear CFG bypass. `attributes.pruning.rule2` records `call_name`, `format`,
   `constant_output`, `result_unused`, `output_only`, `arguments_valid`,
   `control_context_preserved` (all proof flags true), and
   `vulnerability_relevant` (false). The implementation recognizes only a small
   output-call set and literal formats (allowing `%%`); buffer-writing calls,
   `%n`, unresolved format/resource validity and unknown effects remain.
   Proof fields must be supplied by trusted input-program analysis, not invented
   by a formatter, model prediction or gold label. Comments without sufficient
   proof also remain.
3. Rule 3 checks the immutable original graph: a LITERAL/IDENTIFIER has only one
   incident AST edge and no independent dependency/binding role. Its actual
   parent/carrier must retain the value/name and occurrence association. Removing
   earlier dependencies cannot make a formerly dependency-bearing leaf eligible.
4. Rule 4 uses the same node kind and known `attributes.declared_type` within a
   known `function_context`. LOCAL/PARAM also need `name` and `scope`;
   TYPE_REF/METHOD_RETURN need distinct `occurrence` identities. Current merging
   stays within one function context, preserves original entries and all redirected
   edge bindings, and creates a fresh representative. Unknown/ANY types remain.

The importer preserves source attributes rather than guessing declaration
identity, call side effects or constants from unrelated text. Optional
`attributes.explicit_constant` belongs to its original source entry and is
attached to data annotations without inferring values along an arbitrary chain.

## Hyperedge construction

CFG and DDG pools are processed separately in deterministic original-ID order,
using many-to-many, one-to-many and many-to-one modes, in that order. CFG pools
are partitioned by recorded control context; unknown contexts remain binary.
Candidate groups must have disjoint source/target sets, at least two current
binary edges, complete pair witnesses and compatible original-entry bindings.
Only accepted edges are consumed, and later candidates are recomputed from the
remaining pool. Duplicate relation identities and all label associations survive.

## Learning representation

`GraphData.edge_index` is a binary node-to-node witness projection for generic
GNNs. `hyperedge_index` is `[node_index, relation_index]`, with separate batch
offsets for node count and relation count. `incidence_role` distinguishes source
(0) and target (1). `node_types` indexes the paper's 17 categories and
`relation_types` indexes AST/CFG/CDG/DDG/CTRL/DATA (0…5). Type embeddings are
learned by the classifier, not baked into frozen CodeBERT vectors.

The binding dictionary is used for logical validation; physical incidence is
used for feature propagation. Thus directed learning is on the compressed
physical topology, while dependency checks retain original logical identities.
These two levels must not be conflated.

Graph depth is the maximum **finite directed shortest-path distance + 1** over
physical nodes, with one hyperedge source-to-target traversal counting as one
hop. Empty graphs have depth 0, isolated nodes depth 1. Cycles do not imply an
infinite depth; this is not a longest-path statistic. Self-distance is 0 for
this statistic. Dependency validation separately checks positive-length
reachability, including self-cycles.

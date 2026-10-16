# Paper revision to repository mapping

Target: the final manuscript and point-by-point responses supplied in the
October 2026 revision workspace. Final manuscript SHA-256 is recorded in
`results/paper_reported.json`. This mapping describes implemented behavior and
protocols; it does not establish reproduction of historical experimental values.

| Paper item | Repository implementation |
| --- | --- |
| M1: narrow preservation scope | README and schema/reproducibility docs define rule-conditioned preservation of input-CPG dependency information. |
| M2: pruning Rules 2–4 | `sp4cpg/graph.py` checks input facts, original dependencies, content carriers and identity bindings; unresolved candidates remain. |
| M3: algorithms 1/2 and deterministic packing | Canonical dictionary/bindings, fixed MM/SM/MS grouping, distinct relation IDs, lossless DOT/JSON, shared serial/parallel core. |
| M3: directed model and notation | `models/taf_hgcn.py` separates incidence roles, applies six-kind weights and weighted normalization, handles zero degrees and implements mean 1…K positive hops. |
| M3: types, classification and hop/depth definitions | 17 node kinds and six relation kinds; mean graph pooling/logits/cross-entropy; layer count separate from K; `sp4cpg/statistics.py` uses finite shortest-path depth. |
| M4: representation consistency | `scripts/representation_consistency.py` compares paired mean CodeBERT vectors, without TAF propagation or classifier scores. |
| M5: direct dependency validation | `sp4cpg/validation.py`, `scripts/validate_dependencies.py` independently compare projected originals against active output relations and expose all failures/micro counts. |
| M6: distinguish representation/model contributions | `scripts/run_matched.py`, separate graph/encoder configuration, common IDs and ablation flags; efficiency timing boundaries explicitly recorded. |
| M7: standard HGNN control | `models/standard_hgnn.py` uses unordered incidence and unit weights; four configurations share split, data membership and 2025/2026/2027 training seeds. |
| M8: LLM prompt/settings and scope | Full Appendix-B `preprocess/prompt.txt`, provider-independent prepare/import workflow, common token eligibility, one call and strict error reporting. |
| M9: related-work distinctions | Documentation distinguishes HCPG construction from model learning, direct validation from embedding similarity, and generic GNNs from external named detectors. |
| Complete-test evaluation | Per-run summaries include complete test, original-CPG >500 and size bins; per-seed P/R/F1 are averaged separately. |
| Final consistency corrections | Reference summaries distinguish reported full/large-graph results, 54,142 complete pairs, unseen-CWE totals and the repaired-condition illustration. |

The paper's CVE-2022-0407 discussion identifies the vulnerable order of the loop
condition: dereferencing `bd->textstart+s-1` before testing `s>0` can read before
the line when s=0. The linked `figures/CVE-2022-0407-CPG.*` and `HCPG.*` artifacts
are representation illustrations of the repaired condition. They are not the
vulnerable classifier input or evidence of the reported prediction. The final
case-study source is Vim's fix, commit
[`44db8213d38c39877d2148eff6a72f4beccfb94e`](https://github.com/vim/vim/commit/44db8213d38c39877d2148eff6a72f4beccfb94e).

The temporal evaluation is retrospective: train disclosures through May 1,
2020 and later-disclosed test CVEs without training-CVE overlap. Describe it as
later-disclosed vulnerability evaluation, not discovery of unknown zero-days.
The final unseen-CWE table totals are 1,606 examples, 1,214 detections, 75.6%
recall. These remain manuscript-reported values until the original evaluation
records are available and independently rerun.

The existing aggregate revision JSON files include superseded entries. This
repository publishes a curated reference summary based on the final manuscript
and its confirmed results, with internal audit histories kept separate.

# imitation/case4 — Kaggle tutorial head の BC ポート

## Hypothesis

Kaggle ノートブック `kashiwaba/orbit-wars-reinforcement-learning-tutorial` の
**per-source × candidate categorical** head は case3 の 3-head (`from / target_template /
ships_bucket`) より supervisable / decode が単純で、特に target head の macro-F1 が頭打ち
(case3 best 0.31) となっている問題を回避できるはずである。

具体的に:

- target_template head はクラス境界を完全には特定できず、val macro-F1 が頭打ち。
  **per-candidate categorical** に変えれば「具体的にこの惑星に撃つ」を直接分類でき、
  template 解決の曖昧さがなくなる。
- ships_head は case3 で top1=0.84 と既にほぼ正しく学習できており、tutorial 流の
  **rule-based ships** (`max(target.ships+1, 20)`) で代替しても勝率に悪影響は出ない
  と仮定する。
- from_head は不要。`candidate slot 0 = no-op` で per-source の "fire/no-op" は表現できる。

成功指標 (local 1v1 vs `baseline_v1`):

- 30 戦の interim signal で勝率 ≥ 5% (case3 iter9 の 5/100 と同等以上の生存兆候)
- 強い主張は ≥ 300 戦評価で初めて行う。300 戦未満は `project_imitation_case1_phase3` 通り
  noise として注釈する。

## Scope

### Backbone (変更なし、case3 から完全コピー)

- Graph U-Net (kNN k=8, hidden=128, TopK pool 3 段, Encoder/Decoder skip-add)
- planet_feats 35 / global_feats 20 / template_ctx は維持
- `featurizer_phase2.py` の `HistoryState` も同じ (planet history 3 列, global launch history 4 列)

ユーザー指示「バックボーンは変えずに」を遵守。`policy/model.py` の Graph U-Net 層
(`GraphConv`, `TopKPool`, `_knn_adjacency`, encoder/decoder/bottleneck) はビット単位で
case3 と同一。

### Heads (再設計)

| 項目 | case3 | case4 |
|------|-------|-------|
| from_head | `Linear(2H → 1)` per planet (sigmoid) | **削除**: slot 0 == no-op で代用 |
| target_head | `Linear(2H + TEMPLATE_CTX_DIM → NUM_TEMPLATES=8)` per planet | **削除** (template 廃止) |
| ships_head | `Linear(2H → 4 buckets)` per planet | **削除** (rule-based) |
| candidate_head | — | **新設**: per-source slot K=8 の categorical logits |

新 `candidate_head` の入力は notebook 流に
`[h_node (H), ctx (H), candidate_features (CAND_FEAT_DIM)]` を per-candidate に concat した
3 ストリーム MLP。pool 後の per-candidate スカラーが logit。slot 0 (no-op) は
`candidate_features = zeros` で固定。

### Input features (additive)

- planet_feats (35 dim): **変更なし**。notebook の self_features は全て case3 の subset。
- global_feats (20 dim): **変更なし**。同上。
- candidate_features (新設, 14 dim/slot, K=8): notebook 公式 14 dim を完全踏襲:
  - is_valid (slot 0 のみ 1.0 で残り)
  - is_neutral / is_mine / is_enemy
  - tgt.x, tgt.y, dx, dy, dist (board_size 正規化)
  - tgt.ships, tgt.production
  - tgt_is_rotating, **crosses_sun** (← notebook に有り、case3 に無い唯一の新シグナル)
  - src.ships (各 candidate にコピー、reference signal)

candidate slot 0 は no-op 用に `is_valid=1.0` 以外全 0.0。slot 1..7 は notebook の
`build_candidates` ロジック (enemy_quota=2, neutral_quota=2, friendly_quota=3 from K-1=7)
を距離昇順で埋め、足りないものは `fallback` で他から距離昇順に補充。

candidate_mask (B, P, K) bool: slot 0 は常に True。slot 1..7 は
`ships_needed > 0 AND not crosses_sun AND src.ships >= ships_needed` のとき True。

### Implementation steps

1. `pipeline/imitation/case4/main.py` — `sys.path.insert(0, str(Path.cwd()))` パターンで
   `from policy.agent import agent` を re-export。
2. `pipeline/imitation/case4/policy/featurizer.py` — case3 の `featurizer_phase2.py` を
   コピーし、`build_candidates` / `build_candidate_features` を追加。返り値の
   `BatchFeatures` に `candidate_feats (B, P, K, 14)` / `candidate_mask (B, P, K)` /
   `candidate_pid (B, P, K)` を追加。
3. `pipeline/imitation/case4/policy/types.py` — `BatchFeatures` / `PolicyOutput` を更新
   (template_ctx 削除、candidate_feats / candidate_pid / candidate_mask 追加; from_logits
   / target_logits / ships_logits 削除、`candidate_logits (B, P, K)` のみ)。
4. `pipeline/imitation/case4/policy/model.py` — case3 の Graph U-Net から head 部だけ差し替え。
   `candidate_head` は notebook 流の 3-stream MLP (self_h + global_h + cand_h → logit)。
5. `pipeline/imitation/case4/policy/decoder.py` — model 出力 `candidate_logits` の argmax
   を取り、slot 0 なら no-op。slot ≥ 1 は対応する candidate_pid を target に、
   `ships = max(target.ships + 1, 20)` ルール。`aim_with_prediction` で angle を再計算。
   案 4 の overfire 抑制 (committed dict) は維持。
6. `pipeline/imitation/case4/policy/geometry.py` / `templates.py` — `geometry.py` は
   case3 から完全コピー (aim_with_prediction が必要)。`templates.py` は case4 では未使用
   だが、`crosses_sun` 計算ヘルパ (notebook の `shot_crosses_sun`) を独自に置くために
   `policy/__init__.py` には含めない。
7. `pipeline/imitation/case4/policy/agent.py` — `agent_phase2.py` をベースにし、
   featurizer 差し替えと decoder 呼び出しを更新。HistoryState 管理は同じ。
8. `pipeline/imitation/case4/training/preprocess.py` — case3 の preprocess をコピーし、
   per-source ラベルを **target_per_src (template id)** から
   **candidate_slot_per_src (0..K-1)** に変更。`_resolve_action_target` で得た
   `target_pid` が candidate slots のどれに該当するかを引き、該当なしなら
   slot=0 (no-op) ではなく **そのフレームをこの src で UNUSED 扱い**
   (label=-1) にする。fired src で candidate に出現しないケースは稀だが起こり得るので
   候補拡大やそのフレーム drop ではなく「学習対象から外す」が安全。
9. `pipeline/imitation/case4/training/dataset.py` — parquet schema を更新
   (`target_per_src` → `cand_slot_per_src`, `template_ctx` 列削除, candidate_feats 列追加)。
10. `pipeline/imitation/case4/training/losses.py` — 1-head BC loss:
    - `from_head` の focal/BCE は削除 (slot 0 が暗黙の no-op)
    - per-source で `cross_entropy(candidate_logits[fired_src], cand_slot_label)` のみ
    - 「fire しなかった src」は my_planet_mask & ~from_multihot に該当 → label=0 (no-op slot)
      で学習 (notebook と同じ思想: 全 my_planet について 1 回ずつ判断)
    - クラス不均衡 (slot 0 = no-op が多数) は `class_weight` (effective number) を有効化。
11. `pipeline/imitation/case4/training/train.py` — case3 の train.py を流用、
    losses / dataset / model / metrics を case4 仕様に。`run.json` schema は同じ。
12. `pipeline/imitation/case4/evaluation/eval_metrics.py` — top1/top5 acc / no-op precision
    などの簡易メトリクスのみ (target macro-F1 は廃止)。
13. `pipeline/imitation/case4/evaluation/eval_vs_baseline.py` — case3 から流用。
14. `pipeline/imitation/case4/configs/il_case4.yaml` — case3 il_phase2.yaml をベースに、
    loss_weights から `target_*` / `ships_*` 関連削除、`candidate` 1-head 設定に統一。
15. `src/dataset/selfplay/agents.py` — `"il_v4": "pipeline.imitation.case4.policy.agent:agent"`
    を追加。
16. `src/vast/cli.py` の `CASE_DEFAULTS` に `case4` エントリを追加 (stage =
    `train_imitation_case4`, train_module = `pipeline.imitation.case4.training.train`,
    config = `pipeline/imitation/case4/configs/il_case4.yaml`).
17. `backend/pipeline/imitation/case4/__init__.py` 作成 (空)。
18. `tests/pipeline/imitation/case4/__init__.py` + 最低限の test
    (test_featurizer / test_decoder / test_model integration)。

### Local validation

```bash
cd backend
uv run python -c "from pipeline.imitation.case4.policy.agent import agent; print(agent)"
uv run pytest tests/pipeline/imitation/case4 -x -v
cd .. && dev/test-backend
```

### Remote training (Vast.ai)

```bash
git push origin feature/refactor-imitation-head
dev/vast train <SHA> --case case4 --stage train_imitation_case4
```

- Stage: `train_imitation_case4`
- 期待 epoch: 15 (case3 と同設定)
- 期待 wall-time: ~1–2h (RTX 3090, dataset サイズ・GraphUNet 規模 case3 と同等)
- artifact: `data/output/models/imitation/case4/runs/<run_id>/{best.pt, history.jsonl,
  summary.json, run.json}`

### Evaluation (after pull)

```bash
dev/vast pull <run_id>
uv run --directory backend python -m pipeline.imitation.case4.evaluation.eval_vs_baseline \
    --episodes 30 --seed 0   # interim
# n>=300 が必要なときは別途
```

判定基準:

- val candidate top1 acc が case3 target macro-F1 (0.31) より高ければ head 再設計が効いた指標。
  ただし指標形が違うので比較は粗い。**実勝率 (vs baseline_v1) を優先。**
- 30 戦で勝利が 1 つでも出れば "head redesign が崩壊していない" の interim 合格。
- 300 戦で勝率 ≥ 5% なら adopt 候補。それ未満は次の改善案 (encoder 拡張等) を検討。

## Risks / known unknowns

- **k=8 candidate quota が小さすぎる可能性**: 36 惑星時点では fallback で埋めるが、
  pro player が "8 番目より遠い" 惑星を撃つフレームは label が UNUSED に落ちる。
  preprocess で UNUSED 落ちフレーム比率をログし、5% 以上なら K=12 など拡大検討。
- **slot 0 (no-op) class imbalance**: 1v1 episode で fire/turn ratio はおおよそ 1/10 前後。
  `class_weight` (effective number) で補正するが、weight が極端だと train が崩れるので
  beta=0.999 程度で控えめに。
- **n<300 noise**: case3 で 5/100 が 0/300 だった前例があるので 30 戦は飽くまで生存確認。
- **重みの非互換**: case3 の `weights_phase2.pt` は heads 形が違うので load 不可。
  case4 は scratch から学習する。

## Decision log

- 2026-05-01: head 再設計案を case3 baseline と分離し、case4 として新設。
  user 承認済み (Vast.ai launch 含む)。

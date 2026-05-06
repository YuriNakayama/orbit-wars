# imitation/case6 — Attention Backbone (case5 コピー + GraphConv → GAT)

> 作成日: 2026-05-04
> 関連:
>   - `bot/pipeline/imitation/case5/policy/model.py` (コピー元 backbone)
>   - `docs/experiment/imitation/20260503_case5_ship_prediction/plan.md` (case5 featurizer 仕様)
>   - `docs/experiment/imitation/20260501_case4_kaggle_tutorial_head/iter2_result.md` (RunPod BC 5-6 分 / $0.23 実績)
> スコープ: case5 を case6 として複製し、Graph U-Net backbone の GraphConv 層を attention (GAT) に置換する 1 軸変更

## 仮説 (Hypothesis)

**attention 化で「どの planet が重要か」をモデルに明示学習させると、target/ships head のスコアリング精度が向上し、対戦勝率を押し上げる。**

現状 backbone (`case5/policy/model.py:98-115` の `GraphConv`) は近傍 planet を degree-normalized mean で aggregation している。各 planet (中立/敵/自軍最前線) を等価に扱うため、「敵の最前線 planet」「production の高い中立」を選択的に重視できない。`_pairwise_geometry` (`model.py:55-69`) は (dx, dy, dist, ship_log_diff, tgt_is_enemy, tgt_is_neutral) を計算する関数として既に定義されているが、現在の backbone では未使用。これを edge feature として attention に流せば、距離・敵味方関係に応じた重み付けが学習可能になる。

## 既存コードの現状 (from Step 1)

- imitation 配下は 5 case 構成 (`case1`〜`case5`)。`case1/2/3/5` は同一の Graph U-Net backbone (273 行同型)、差分は featurizer / template の入出力次元のみ。`case4` のみ candidate head 系で別アーキ。
- canonical は **case1** (iter15 まで weight 蓄積、Phase 2 で 0/100 → 5/100 だが再評価 0/300、`project_imitation_case1_phase3` 参照)。
- **case5** は `bot/pipeline/imitation/case5/` に存在し、case1 をコピーして ship-prediction featurizer 6 列を planet feat に追加 (PLANET_FEAT_DIM=17) した構成。学習はまだ未実施 (Cycle 4 以降の予定、`docs/experiment/imitation/20260503_case5_ship_prediction/result.md`)。
- **次の空き case 番号は case6**。`bot/src/dataset/selfplay/agents.py` の `AGENT_REGISTRY` には `il_v1`〜`il_v5` まで登録済み。
- backbone の核心:
  - `_knn_adjacency` (`model.py:72-95`): kNN k=8 + symmetric, padding row isolation
  - `GraphConv` (`model.py:98-115`): `h' = ReLU(W1·h + W2·mean(neighbours))`, **edge feature 未使用**
  - `TopKPool` (`model.py:118-151`): differentiable Gao & Ji 2019, 比率 2/3 → 1/2 で 2 段
  - decoder: unpool + skip add (`model.py:225-231`), 既存 3-head: from / target / ships

## スコープ (Scope)

- **新規 case ディレクトリ**: `bot/pipeline/imitation/case6/` (`case5` を丸ごとコピー)
- **変更ファイル**: `case6/policy/model.py` のみ
  - `GraphConv` クラス → `GraphAttention` クラスに置換
  - `_knn_adjacency` の出力 (adj) は流用、新たに `_pairwise_geometry` を edge feature として消費
  - encoder/decoder の 5 箇所の GraphConv 呼び出し (`enc0`/`enc1`/`enc2`/`dec1`/`dec0`) を attention 版に差し替え
  - `TopKPool` / unpool / 3 head は不変 (head・規模・プーリングは線形保存の方針)
- **不変**: featurizer (case5 の 17 dim をそのまま継承)、templates、decoder.py、agent.py、training/, evaluation/
- **ハイパーパラメータ**: hidden=128 / KNN_K=8 / ships_buckets=4 を維持。新規追加: `attn_heads=4` (multi-head 4)
- **`AGENT_REGISTRY` 追加**: `"il_v6": "pipeline.imitation.case6.policy.agent:agent"`
- **DVC stage 追加**: `dvc.yaml` に `preprocess_imitation_case6` / `train_imitation_case6` / `eval_imitation_case6` を case5 と同形で追加 (case5 の preprocess output が同一なら共有も検討)

## 実装ステップ (Implementation outline)

1. **case6 ディレクトリ作成**: `cp -r bot/pipeline/imitation/case5 bot/pipeline/imitation/case6`、その後 `__init__.py` / `README.md` / `policy/model.py` のヘッダ docstring を case6 用に書き換え。
2. **`case6/policy/model.py` の差し替え**:
   - 新規クラス `GraphAttention(in_dim, out_dim, num_heads=4, edge_dim=PAIR_FEAT_DIM)` を実装。
     - 各 head ごとに query/key projection、edge feature を加算した attention score、softmax over masked neighbours、value aggregation
     - 出力: `Linear(num_heads * head_dim, out_dim)` で連結射影
   - `GraphUNetPolicy.__init__` 内の `enc0/enc1/enc2/dec1/dec0` を `GraphAttention(h, h, num_heads=4, edge_dim=PAIR_FEAT_DIM)` に置換
   - `forward` 冒頭で `pair_feats = _pairwise_geometry(x)` (B, P, P, 6) を計算し、各 attention 層に `(h, adj, mask, pair_feats)` を渡す
   - pool 後の階層 (h1, h2) では、pooling 時に保存した `top_idx` を使って `pair_feats` を gather し縮約 (新ヘルパ `_gather_pair_feats(pair_feats, top_idx)`)
   - クラス名は `GraphAttentionUNetPolicy`、`DeepSetsPolicy = GraphAttentionUNetPolicy` の alias は維持 (agent.py が import している)
3. **unit test**: `bot/tests/pipeline/imitation/case6/test_model.py` を case5 のテストをコピーして作成。
   - forward shape: `from_logits` (B,P), `target_logits` (B,P,NUM_TEMPLATES), `ships_logits` (B,P,4)
   - mask 動作: padding planet が `-inf` で潰されること
   - パラメータ数: 既存 (~165k) → 新規 (~250k 想定) の桁が合うこと
   - smoke train: 1 epoch のミニバッチで loss が減少することを確認
4. **`bot/src/dataset/selfplay/agents.py`** に `"il_v6"` を追加。
5. **DVC stage**: `dvc.yaml` の `train_imitation_case5` を template に case6 stage を追加。`params.yaml` には `imitation.case6.attn_heads` などの新設キー。

## 検証方法 (Validation method)

- **ローカル sanity**:
  - `dev/test-bot` (format / lint / type / pytest)
  - `uv run --directory bot pytest tests/pipeline/imitation/case6 -x`
  - `uv run --directory bot python -m submit submit imitation/case6 --dry-run --skip-validation -m "case6 dry-run"` (Path.cwd() trick が動くか確認)
- **リモート学習**:
  - `git push origin feature-imitation-model-structure`
  - `dev/runpod train <commit-sha> --case case6 --cloud-type SECURE` (case4 iter2 実績: RTX3090, 5-6 分, $0.23)
  - 想定 cost: $0.3〜$0.5 / run (cost-limit $1.5 内)
  - 完了監視: `dev/runpod watch <run_id>` または `--watch` で起動同時監視
- **評価**:
  - 主要対戦: **vs baseline_v1 (rulebase/case1) 50 戦** (user 指定、デフォルトの ≥300 ルールから引き下げ)
  - 補助シグナル: 学習履歴の **val_total / val F1** (case5 が未実施なので絶対比較は不可、収束有無のみ確認)
  - 採否しきい値: vs baseline_v1 で勝率 ≥ 5% を「生存」、≥ 20% を「次 iter (300 戦再評価) へ進む価値あり」
- **既知リスク (memory より)**:
  - `project_imitation_case1_phase3`: **n=50 では強い主張不可** (95% Wilson CI 上限が 17pp 程度広がる)。「baseline_v1 に勝てる」と結論する前に必ず 300 戦再評価。
  - `project_runpod_onstart_pitfalls`: case6 は新規 case のため onstart スクリプトの cwd-relative path 周りで失敗しやすい。Step A (smoke) を 1 回挟むこと推奨。

## 期待される結果 (Expected outcomes)

| シナリオ | vs baseline_v1 (50 戦) | 解釈 | 次アクション |
|---|---|---|---|
| 良 | 勝率 ≥ 20% | attention 化が効いている兆候 | 300 戦再評価 → 効果確認できれば il_v6 を canonical 候補に |
| 中 | 勝率 5–20% | il_v1 (3/300) よりは改善した可能性 | 300 戦再評価で seed variance の影響を切り分け |
| 否 | 勝率 < 5% | attention 化単独では効果なし | head 設計 / 規模拡大 / pooling を別 iter で検証 (Round 1 で保留にした 3 軸) |

## 採用しなかった案 (Rejected alternatives)

- **case1 を直接改変**: canonical を壊す。weight_iter15 までの履歴と互換性が消える → リスク高で却下
- **case5 を直接改変**: featurizer 変更 (ship-prediction) と backbone 変更が同時に走り、効果分離不能 → 却下
- **4 軸同時変更 (attention + 規模 + pooling + head)**: Round 2 でユーザーが「attention のみ」を選択。原因特定可能性を優先
- **vs il_v1 で評価**: case5 → case6 で featurizer 同一なので backbone 効果分離としては il_v1 より理想だが、ユーザーは vs baseline_v1 を選択 (絶対閾値志向)

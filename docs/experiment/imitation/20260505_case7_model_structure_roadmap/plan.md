# imitation/case7-10 — Model Structure Roadmap (4-iter ablation)

> 作成日: 2026-05-05
> 関連:
>   - `docs/experiment/imitation/20260504_case6_attention_backbone/result.md` (case6 完走、val ablation で backbone 改修が val に効くと実証)
>   - `data/output/experiment/imitation_case6_attention_backbone/case5_vs_case6_comparison.png` (case5 GraphConv vs case6 Attention の ablation 4-panel グラフ)
>   - memory `project_runpod_5_traps_2026_05_04.md` (RunPod インフラ trap 8 件、修正済 commit `88ba83a`/`fe554bc`/`e87445c` まで反映)
> スコープ: case7→8→9→10 の 4 iter で model structure を順次切替、val_target_acc / val_loss を case5/case6 と ablation 比較
> 注意: docs.md の「1 directory = 1 hypothesis」原則の例外。ユーザー明示要求により 4 構造案を 1 plan.md に集約。各 case の詳細実装/結果は `iter{N}_result.md` で個別に記録する

## 仮説 (Hypothesis)

**case6 で「backbone 改修は val 指標を有意に押し上げる」(val_loss −2.1%, val_target +6%) が証明された。残課題は val_target_acc 0.42 の頭打ち = target diversity 不足。本ロードマップは 4 つの異なる構造軸でこの bottleneck を破れるかを順次検証する。**

各 axis の why-it-should-work:

| iter | 構造案 | 何が val_target_acc を伸ばす期待か |
|------|------|------|
| case7 (A) | Set Transformer + cross-attention | template を learnable query にして「from に条件付けた template 選択分布」を直接学習。3-head 独立性が target diversity 欠如の根因という仮説の直球テスト |
| case8 (B) | Hierarchical Pointer Network | from→target→ships を autoregressive 化し前段の決定が後段を条件付ける。「from=A 選んだ後の target 分布」が現在は学べていない |
| case9 (C) | Equivariant GNN (E(2)-EGNN) | 回転・並進対称な座標表現で学習効率↑。BC のデータ効率 bottleneck (944 episodes / 300k frames) を破る方向 |
| case10 (D) | Larger backbone (h=256, L=5, heads=8) | case6 epoch 14 まで train_loss 下がり続け = capacity 不足。最も無骨だが「規模を上げれば val_target も伸びる」のボトルネック診断 |

## 既存コードの現状 (from Step 1)

- imitation 配下 case1〜case6 まで existing、次空き = case7 (`il_v7` は AGENT_REGISTRY 未登録)
- case6 = 358 行の `GraphAttention U-Net (multi-head=4 + edge feat from `_pairwise_geometry`) + TopKPool + 3-head (from/target/ships)`
- 同 featurizer (PLANET_FEAT_DIM=17, ship-prediction 6 列) を case5/case6 が共有 — case7-10 でも同 featurizer を踏襲し ablation 効果分離
- 過去 iter の所見: case5 (GraphConv) → case6 (Attention) の ablation で val_loss −0.077 / val_target +0.024 / val_from +0.041 / val_ships +0.030 を実測。 backbone 単独でこれだけ動くなら head 軸も期待値あり

## 4-iter Roadmap (実装順)

各 case の詳細スコープは下記。**実装は case7 から順次、各 case 完走後に `iter{N}_result.md` 作成 → 次 iter の go/no-go 判断**。1 case あたり想定コスト ~$0.50/run (case6 で確立された infra 修正済 commit を使用)。

---

### iter1: case7 — Set Transformer + cross-attention head ⭐ (最優先)

**主仮説**: template を learnable query にした cross-attention head なら val_target_acc 0.42 → 0.45+ に押し上げ可能。

**変更箇所** (case6 をコピー → case7):

- `bot/pipeline/imitation/case7/policy/model.py`
  - **encoder**: `GraphAttentionUNetPolicy` を **Set Transformer encoder (SAB / ISAB)** に置換。SAB = self-attention block, ISAB = induced set attention で計算量 O(P^2) → O(P·m) (m = inducing points 数)
  - **target_head**: 既存 `Linear(2H + TEMPLATE_CTX_DIM, NUM_TEMPLATES)` を **PMA + cross-attention** に置換
    - learnable query Q ∈ R^(NUM_TEMPLATES × H)、planet set を K/V に (B, P, H) を multi-head attention
    - 出力: (B, P, NUM_TEMPLATES) ロジット (planet ごとに per-template スコア)
  - from_head / ships_head は case6 のまま (per-source linear)
- `bot/pipeline/imitation/case7/configs/il_case7.yaml` — case6 yaml の `attn_heads=4` に加え `inducing_points=16` 追加
- `dvc.yaml` に `preprocess_imitation_case7` / `train_imitation_case7` を case6 と同形で追加
- `bot/src/dataset/selfplay/agents.py` に `"il_v7"` 登録
- `bot/src/runpod_io/cli.py` の `CASE_DEFAULTS` に `"case7"` 追加

**評価**:
- 主指標: **val_target_acc** (case6 比 +3pp = 0.45+ で「効いた」と判定)、val_loss (case6 比 −1% で「効いた」)
- 補助: val_from_acc / val_ships_acc / per-template prediction 分布 (entropy で diversity を定量、case5/case6 と並べる)
- 対戦: 50 戦 sanity だけ (n<300 では強い主張不可、 memory `project_imitation_case1_phase3`)

**想定コスト**: $0.50-0.70/run (preprocess 30 分 + train 7 分、A6000 or 4090 SECURE)

---

### iter2: case8 — Hierarchical Pointer Network

**主仮説**: from→target→ships を autoregressive 化すると、 「from=A 選んだ後の target 分布」が学べ val_target +5pp 期待。

**変更箇所** (case7 から fork、 backbone は case7 と同じ Set Transformer or case6 GraphAttention を再利用 — case7 結果次第):

- `bot/pipeline/imitation/case8/policy/model.py`
  - encoder: case6/7 と同じ Graph encoder
  - **decoder**: Pointer Network 風 LSTM/Transformer decoder
    - step 0: from を pointer attention で planet set から選択 → softmax(B, P)
    - step 1: 選んだ from の embedding を condition に target distribution を attention で生成
    - step 2: target を condition に ships を生成
  - 学習時は teacher forcing、推論時は argmax cascade
- 残り (config, dvc.yaml, agent_registry, runpod_cli) は case7 と同形で `il_v8` 登録

**評価**: 同上。主に val_target_acc + per-template diversity。

**想定コスト**: $0.50-0.70/run

---

### iter3: case9 — Equivariant GNN (E(2)-EGNN)

**主仮説**: 座標 (x, y) を回転・並進 equivariant に扱うと、 BC のデータ効率向上で val_loss −2% / val_target +3pp 期待。

**変更箇所**:

- `bot/pipeline/imitation/case9/policy/model.py`
  - **EGNN layer (Satorras 2021)** をスクラッチ実装 (or `egnn-pytorch` lib 採用)
    ```
    m_ij = φ_e(h_i, h_j, ||x_i - x_j||^2, e_ij)
    x_i' = x_i + Σ_j (x_i - x_j) φ_x(m_ij)
    h_i' = h_i + φ_h(h_i, Σ_j m_ij)
    ```
  - kNN graph + ship_log_diff / is_enemy / is_neutral を edge features に含める
  - head は case6 のまま (3-head)、 backbone のみ EGNN 化
- 計算量: 既存 attention U-Net と同等、 forward 1.5× 遅い見込み (relative coord computation)

**評価**: 同上。さらに **回転 augmentation 入れた場合の val_loss 変化** が小さいことを確認 (= equivariance の sanity check)。

**想定コスト**: $0.70-1.00/run (forward 遅め)

---

### iter4: case10 — Larger Backbone (capacity diagnosis)

**主仮説**: hidden 128→256, layers 3→5, heads 4→8 の単純 scale up で val_loss が下げ止まらず → capacity 不足が真の bottleneck と診断できる。

**変更箇所**:

- `bot/pipeline/imitation/case10/policy/model.py` = case6 の hyperparameter のみ変更
  - `ModelConfig`: `hidden=256`, `attn_heads=8`、層数を 3→5
- それ以外 case6 と同型

**評価**: 同上。これは 「scale が効くか / 効かないか」のみを切り分ける baseline 実験。

**想定コスト**: $1.00-1.50/run (params ~700k で train 3-4 分長くなる + memory )

---

## 全 4 iter 共通の検証方法 (Validation method)

- **ローカル smoke**: 各 case で `il_case<N>_smoke.yaml` (max_episodes=10, epochs=2) を case7 の sanity run で 1 度だけ確立、以降の case はそれを mirror
  - `uv run --directory bot python -m pipeline.imitation.case<N>.training.preprocess --config pipeline/imitation/case<N>/configs/il_case<N>_smoke.yaml`
  - `uv run --directory bot python -m pipeline.imitation.case<N>.training.train --config pipeline/imitation/case<N>/configs/il_case<N>_smoke.yaml`
- **CI**: `dev/test-bot` (format / lint / type / pytest)
- **unit test**: `bot/tests/pipeline/imitation/case<N>/test_model.py` (forward shape, mask, NaN safe, param count)
- **リモート学習**: `dev/runpod train <commit-sha> --case case<N> --cloud-type SECURE --gpu-name 'NVIDIA RTX A6000' --gpu-name 'NVIDIA GeForce RTX 4090' --image runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2204`
- **評価メトリク**:
  - 主: **val_target_acc** (case6 0.42 比 +3pp で「効いた」)、 val_loss (case6 3.65 比 −1% で「効いた」)
  - 補助: val_from_acc / val_ships_acc / per-template prediction 分布 (target diversity 定量)
  - sanity: vs baseline_v1 50 戦 (主 metric ではない、 n<300 で 0/50 が出ても採否判定に使わない)

## 採否判断 / 次への進め方

- 各 case 完走後に `iter{N}_result.md` 作成、val_target_acc / val_loss を case5/case6/前 case と並べた表で比較
- 「効いた」(主 metric が threshold 超え) 案を後続 case の base に採用、 効かない案は記録だけ残して次へ
- 4 iter 全部走った後に `analysis.md` で総合判断 + Cycle 5 以降の提案 (joint head, BC+RL 混合, Phase 2 fix 流入 等)

## 参考 (References)

| 案 | 出典 / 実装参考 |
|---|---|
| A. Set Transformer | [Lee et al. 2019 (arXiv:1810.00825)](https://arxiv.org/abs/1810.00825) — SAB / ISAB / PMA を提案。 [official PyTorch impl](https://github.com/juho-lee/set_transformer) |
| B. Pointer Network | [Vinyals et al. 2015 (arXiv:1506.03134)](https://arxiv.org/abs/1506.03134) — content-based attention で pointer 出力。 [PyTorch impl](https://github.com/threelittlemonkeys/pointer-networks-pytorch) |
| C. EGNN | [Satorras et al. 2021 (arXiv:2102.09844)](https://arxiv.org/abs/2102.09844) — relative coord-based message passing で E(n) equivariance。 [official impl](https://github.com/vgsatorras/egnn) / [lucidrains impl](https://github.com/lucidrains/egnn-pytorch) |
| D. Larger backbone | (出典不要、scale up baseline) |

## 注記: ユーザー要求と本 plan の運用例外

通常の `.claude/rules/docs.md` 原則では「1 directory = 1 hypothesis」だが、本 plan は ユーザー明示要求により 4 構造案を 1 directory に集約 (`20260505_case7_model_structure_roadmap`)。各 case 完走時には `docs/experiment/imitation/20260505_case7_model_structure_roadmap/iter{N}_result.md` を本 directory 内に追加していく。new directory は作らない。

各 case の実コードは `bot/pipeline/imitation/case<N>/` (case7-10) に独立して存在。 case 独立性ルール (`.claude/rules/bot/pipeline.md`) は各 case 内で守る (cross-case import 禁止)。

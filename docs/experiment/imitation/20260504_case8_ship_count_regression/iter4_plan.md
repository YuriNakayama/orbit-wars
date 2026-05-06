# imitation/case8 — iter4: head 側の振動抑制 (grad_clip + EMA + head dropout)

> 作成日: 2026-05-05
> 関連:
> - [`iter1_plan.md`](./iter1_plan.md) / [`iter1_result.md`](./iter1_result.md) — ship_head 設計、1/50 = 2.0%
> - [`iter2_plan.md`](./iter2_plan.md) / [`iter2_result.md`](./iter2_result.md) — cosine LR + early stop、0/50 = 0.0%
> - [`iter3_plan.md`](./iter3_plan.md) / [`iter3_result.md`](./iter3_result.md) — focal loss、0/50 = 0.0%、val_total が iter1/2 比 24x 縮小も振動残存
>
> スコープ: backbone (Graph U-Net) には**触れず**、head + 学習 pipeline 側の調整で loss / acc 振動を抑制。grad_clip + EMA + head dropout の **3 点同時導入**。

## 仮説 (Hypothesis)

iter1/2/3 の train_total / val_total / val_cand_acc / val_ship_loss は epoch ごとに激しく振動 (val_total が iter2 で 6x、iter3 で 8x の幅で上下)。**根本原因は単一でなく、(a) 勾配爆発、(b) head の過適合、(c) val 評価ノイズ の複合**。

3 つを同時に抑える複合対策で **train/val curve が単調減少に近づき、best.pt 選定が安定**するはず。win_rate uplift は副次目標で、まずは learning dynamics の健全化を最優先する。

メカニズム:

- **grad_clip (max_norm=1.0)** — focal loss 後でも一部 hard fire example で勾配 spike が発生していると推測。`torch.nn.utils.clip_grad_norm_` で L2 ノルムを 1.0 以下に圧縮し、loss spike を遮断
- **EMA (decay=0.999)** — `torch.optim.swa_utils.AveragedModel` で weights の指数移動平均を保持。eval / best.pt 選定時は EMA weights を使い、val_acc 振動を平滑化
- **head dropout (rate=0.2)** — `cand_score` / `ship_head` の MLP に各 1 層 `nn.Dropout(0.2)` を挿入。head の過適合を抑え、generalization を押し上げる (backbone GraphConv は不変)

期待: val_total の epoch ごと max/min 比が **iter2/3 の 6-8x → < 2x** に縮小、best.pt 選定が安定化、win_rate は副次的に改善余地。

## 既存コードの現状 (from Step 1)

- `policy/model.py`: Graph U-Net backbone (3 GraphConv + 2 TopK pool)、head は `cand_score = MLP(3H→H→1)` と `ship_head = MLP(2H→H→1)`、**dropout/BN なし**
- `training/train.py`: AdamW (lr=1e-3, wd=1e-4) + cosine warmup scheduler (iter2 から)、**grad_clip なし**、early stop on val_cand_fire_acc patience=5、**EMA なし**
- `training/losses.py`: focal loss (α=0.25, γ=2.0) + class_weights (iter3 から)
- 過去 iter の所見: iter3 で focal loss により train_total が 75x 縮小したものの val_total は 18k-95k で 5x 以上振動、best.pt (epoch 9) の val_cand_fire_acc=0.210 は iter1/2 比でわずか低下、win_rate は 3 iter 連続 0-2% で plateau

## スコープ (Scope)

### 変更ファイル

| Path | 変更内容 |
|---|---|
| `bot/pipeline/imitation/case8/policy/model.py` | `cand_score` MLP の `Linear(3H→H) → ReLU` の直後に `nn.Dropout(0.2)` を 1 層、`ship_head` MLP の `Linear(2H→H) → ReLU` の直後にも `nn.Dropout(0.2)` を 1 層挿入。backbone (GraphConv / TopKPool / `psi`) は **完全不変** |
| `bot/pipeline/imitation/case8/training/train.py` | (1) `optimizer.zero_grad()` → `loss.backward()` の直後に `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_max_norm)` を挿入。(2) `torch.optim.swa_utils.AveragedModel` で `ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))` を作成、各 batch の `optimizer.step()` 直後に `ema_model.update_parameters(model)` を呼ぶ。(3) val/best.pt 選定時に `ema_model` を eval mode で評価、torch.save する weights も EMA weights を採用。(4) `_stamp` で grad_norm_pre_clip / lr / ema_decay を毎 epoch log |
| `bot/pipeline/imitation/case8/configs/il_case8.yaml` | `train.grad_clip_max_norm: 1.0` / `train.ema.enabled: true` / `train.ema.decay: 0.999` / `train.head_dropout: 0.2` を追加 |
| `bot/tests/pipeline/imitation/case8/test_iter4_stabilization.py` (新規) | (1) grad_clip が大勾配を 1.0 に圧縮することを単体テスト。(2) AveragedModel が weight 更新後に live と異なる値を持つことを確認。(3) Dropout 層が train/eval mode で挙動切替することを確認。(4) `compute_loss` の dropout はテスト時に発火しない (eval mode) こと |

### 変更なし

- `policy/model.py` の **Graph U-Net backbone (in_proj / enc0-2 / pool0-1 / dec0-1 / psi)** は完全に不変
- `policy/{candidates,featurizer,decoder,types,geometry,agent}.py` 不変
- `training/{preprocess,dataset,losses}.py` 不変 (focal loss は iter3 から維持)
- scheduler / early stop / best_metric (= val_cand_fire_acc) — iter2/3 から維持

### Hyperparameters

| Knob | iter3 | iter4 |
|---|---|---|
| grad_clip max_norm | なし | **1.0** |
| EMA | なし | **enabled, decay=0.999** |
| Head dropout | なし | **0.2 (cand_score / ship_head の MLP 中間 ReLU 直後)** |
| Optimizer / LR | AdamW(1e-3) + cosine warmup | 不変 |
| epochs | 30 (early stop epoch 14 で発動) | 30 (不変) |
| Loss | focal (α=0.25, γ=2.0) + class_weight | 不変 |
| best_metric | val_cand_fire_acc max | 不変 (ただし EMA weights で評価) |

## 実装ステップ (Implementation outline)

1. `policy/model.py`: `CandidatePolicy.__init__` で `head_dropout: float = 0.2` パラメータを追加、`cand_score = nn.Sequential(Linear(3H→H), ReLU(), Dropout(p), Linear(H→1))`、`ship_head = nn.Sequential(Linear(2H→H), ReLU(), Dropout(p), Linear(H→1))` に書き換え
2. `training/train.py`: (a) `from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn`、(b) `ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))` を model 構築直後に作成、(c) batch loop の `loss.backward()` の直後に `nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)` 挿入、(d) `optimizer.step()` 後に `ema_model.update_parameters(model)`、(e) val_loop は `ema_model` で評価、best.pt 保存時に `ema_model.module.state_dict()` を save
3. `configs/il_case8.yaml`: `train.grad_clip_max_norm`, `train.ema.{enabled,decay}`, `train.head_dropout` の 3 セクション追加
4. `tests/pipeline/imitation/case8/test_iter4_stabilization.py` 新規: grad_clip / EMA / Dropout の sanity check
5. `dev/test-bot` で format / lint / mypy / pytest 全通過確認
6. RunPod launch、commit + push、cron 監視
7. `dev/runpod pull --from s3` → 50 戦評価 → `iter4_result.md`

## 検証方法 (Validation method)

- **ローカル**:
  - `dev/test-bot` (format / lint / type / pytest)
  - `uv run --directory bot pytest tests/pipeline/imitation/case8 -x`
  - `uv run --directory bot python -c "from pipeline.imitation.case8.policy.model import build_model; m=build_model(); print(m.cand_score)"` で Dropout 配置を確認
- **リモート**:
  - `dev/runpod train <new_sha> --case case8`
  - 想定所要時間: container 起動 ~10 min + dvc pull 5-7 min + train 30 epoch (EMA overhead ~5%) ≒ ~25-30 min train + post ≒ **~50 min total**
  - 想定コスト: ~$0.55 (RTX 4090 SECURE $0.69/h × 0.8h)
- **評価 (主要メトリクス: train/val curve の平滑さ)**:
  - **(主)** `train.log` から epoch ごとの val_total を抽出、**max/min 比 < 2.0** を採用しきい値 (iter2=6x, iter3=8x → 改善目標)
  - **(主)** val_cand_fire_acc の **moving max が単調増加** であること (iter3 は 0.166 → 0.210 で増加するも途中で 0.066 まで dip)
  - **(副)** vs baseline_v1 50戦 win_rate (iter1=2%, iter2=0%, iter3=0% → >5% 期待だが副次)
  - 採否しきい値:
    - **(主)** val_total max/min < 2.0 → 振動 fix 成功と判定、追加 iter5 で win_rate 評価に進む
    - **(主)** val_total max/min ≥ 4.0 → 振動 fix 失敗、別アプローチ (lr 再調整 / EMA decay 調整) に切替
    - **(副)** 50 戦で >5% (3/50) → 300 戦昇格して `dev/runpod promote` 採否判定

## リスク / 注意点

1. **EMA + dropout の相互作用**: dropout は train mode で random、EMA は full weights を平均する。EMA evaluate 時は dropout=0 (eval mode) が標準なので問題ないはず。実装時に `ema_model.eval()` を確実に呼ぶこと
2. **grad_clip max_norm=1.0 が focal loss に対して厳しすぎる可能性**: iter3 の train_total ~40k は focal で down-weight された結果、勾配自体が小さい可能性あり。`_stamp` で `grad_norm_pre_clip` を毎 step log し、実際に clip が発動しているか確認
3. **best.pt が EMA weights だと submit 時に互換性破綻**: 推論時の `policy/agent.py` は `state_dict` を load するだけなので、EMA weights を `model.load_state_dict(ema_state)` で load できれば OK。ただし `AveragedModel.module.state_dict()` の key prefix が変わらないか確認 (PyTorch 2.6 では一致するはず)
4. **`AveragedModel` の memory overhead**: model duplicate 分の VRAM (CandidatePolicy hidden=128 で ~6MB) なので RTX 4090 24GB では誤差
5. **dropout=0.2 で under-fit リスク**: iter3 の val_cand_fire_acc=0.210 をさらに下げる可能性。worst case 0.18 程度を想定、それでも focal loss 自体は機能しているので acceptable

## 参考 (References — Step 3 web 調査)

- [PyTorch Lightning WeightAveraging callback (lightning.ai docs)](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.WeightAveraging.html): `AveragedModel` (PyTorch 標準) を SWA / EMA 両対応で wrap、各 step 後に `update_parameters` を呼ぶ用法を確認
- [GitHub: lucidrains/ema-pytorch](https://github.com/lucidrains/ema-pytorch): SimSiam/MoCo 系で decay=0.999 が標準、small batch でも安定
- [Why Gradient Clipping Accelerates Training (OpenReview, BJgnXpVYwS)](https://openreview.net/pdf?id=BJgnXpVYwS): max_norm 値は 90th percentile of gradient norms ≒ default 1.0 が経験則として妥当
- [Stabilizing LLM Training (Rohan Paul)](https://www.rohan-paul.com/p/stabilizing-llm-training-techniques): warmup + clip + dropout の 3 点セットが loss spike 抑制のクラシックパターン

## 次 iter 候補 (本 plan の範囲外)

- iter5: iter4 で振動 fix 成功なら epochs を 60 まで延長、long-run の追加 generalization 効果を計測
- iter6: ship_head λ_ship sweep (0.5 / 1.0 / 2.0) — 振動が消えてから head balance 探索
- iter7: data 側の oversample (fire/noop 1:1) — head fix では限界の場合の data fix
- 本 PR は iter4 まで取り込んで main マージ、iter5+ は別 PR で

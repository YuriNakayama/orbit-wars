# reinforce/case1 — ppo_stabilization (iter1)

> 作成日: 2026-05-18
> 仮説 ID: H1 (target_kl early stopping)
> hypotheses.md: docs/experiment/reinforce/20260518_case1_ppo_stabilization/hypotheses.md
> 関連: iter0 baseline = `data/output/models/reinforce/case1/runs/20260517-151315__feature-reinforce-learning-case0__f16f343__seed0/` (BC warm-start + train.yaml 設定で走った最初の run)
> スコープ: PPO update loop に target_kl による epoch 中断ロジックを追加

## 仮説 (Hypothesis)

`ppo_update` の epoch ループに `target_kl=0.05` を追加し、各 epoch の minibatch 全体平均 `approx_kl` が `target_kl` を超えたら次の epoch をスキップする。**clipping 単独では守れない trust region を hard 保証** することで、CPU 検証で観測された `approx_kl` 0.25–1.2 のスパイクを抑える。CleanRL / SB3 標準実装。

## 既存コードの現状

- 主要モジュール: `bot/pipeline/reinforce/case1/training/ppo.py` — `ppo_update()` 内に二重ループ `for _ in range(cfg.epochs): for start in range(0, n, cfg.minibatch_size):`。各 minibatch で `approx_kl = (old_lp - new_lp).mean()` を計算しているが、累計 / 平均だけで break ロジックは無い
- 過去 iter の所見 (CPU 検証時):
  - cpu_stable_v1.yaml (vs baseline_v1, 30 iter × 16 ep): `approx_kl` 0.08–1.18, `bc_kl` 0.18–0.49, best_win_rate 0.0625
  - cpu_noop.yaml (vs random_noop, 20 iter × 4 ep): `approx_kl` 0.03–2.54, `bc_kl` 0.10–4.05, 振動大
- `train.yaml` 現状: `ppo_epochs=2`, `minibatch_size=256`, `clip_eps=0.2`, `entropy_coef=0.001`, `kl_beta=0.5`, lr=1e-4

## スコープ

- 変更ファイル:
  - `bot/pipeline/reinforce/case1/training/ppo.py` — `PPOConfig` に `target_kl: float | None = None` を追加、`ppo_update` 内 epoch ループに「当該 epoch の minibatch 平均 approx_kl > target_kl なら次 epoch を skip」ロジック
  - `bot/pipeline/reinforce/case1/training/train.py` — `cfg["training"]["target_kl"]` を読んで PPOConfig に渡す
  - `bot/pipeline/reinforce/case1/configs/train.yaml` — `training.target_kl: 0.05` 追加
- ハイパーパラメータ: `target_kl: null → 0.05`
- データセット / 特徴量変更: なし

## 実装ステップ

1. `bot/pipeline/reinforce/case1/training/ppo.py`:
   - `PPOConfig` に `target_kl: float | None = None` フィールド追加
   - `PPOStats` に `epochs_run: float` を追加 (実際に走った epoch 数の平均)
   - `ppo_update` 内 epoch ループを再構成:
     ```python
     for epoch_idx in range(cfg.epochs):
         epoch_kls = []
         for start in range(0, n, cfg.minibatch_size):
             ...  # 既存処理 + epoch_kls.append(approx_kl.item())
         epochs_run += 1
         if cfg.target_kl is not None and sum(epoch_kls) / len(epoch_kls) > cfg.target_kl:
             # 当該 epoch は完走、ただし次 epoch を skip
             logger.info({"event": "target_kl_early_stop", "epoch": epoch_idx, "mean_kl": ...})
             break
     ```
   - `PPOStats.epochs_run` を平均値で返却 (cfg.epochs 未満なら early stop されたことが分かる)
2. `train.py`:
   - `PPOConfig(target_kl=float(cfg["training"].get("target_kl", 0.0)) or None, ...)` 追加
   - history 辞書に `epochs_run` を含める
3. `configs/train.yaml`:
   - `training.target_kl: 0.05` 追加
4. lint / mypy 通過確認
5. commit + push (RunPod 投入のため)

## 検証方法

### スキップする検証 (from hypotheses.md skip list)

- **ローカル CPU 学習は禁止** — smoke (1-episode self-play) は pod 上で実施。手元では import / ruff / mypy のみ
- **300 対戦評価をしない** — `eval_vs_baseline` は 50 ep までで止める
- **replay 分析は実施しない** — 学習指標 (approx_kl, bc_kl, policy_loss, value_loss, win_rate curve) で採否判定
- Kaggle publicScore は引用しない (project rule)

### 実施する検証

- ローカル (RunPod 投入前):
  - `uv run --directory bot ruff format pipeline/reinforce/case1/`
  - `uv run --directory bot ruff check pipeline/reinforce/case1/`
  - `uv run --directory bot mypy pipeline/reinforce/case1/`
  - `uv run --directory bot pytest tests/unit/src/runpod_io/ --no-cov` (リファクタ regression なし確認)
- リモート: `dev/runpod train <sha> --case reinforce_case1 --cloud-type ALL --gpu-name "NVIDIA L4" --gpu-name "NVIDIA RTX A4500" --gpu-name "NVIDIA RTX A5000" --gpu-name "NVIDIA GeForce RTX 3090" --gpu-name "NVIDIA A40" --watch`
  - 想定所要時間: 約 1 時間 (L4 クラス、cost-limit $1.5/run 内)
  - 100 iter × 16 ep
- 評価:
  - **主要メトリクス (採否判定)**:
    - (a) `max approx_kl` が全 iter で **< 0.1** ⇒ trust region 守れている
    - (b) `bc_kl` curve の iter 間 std が iter0 baseline 比で **< 0.5×** ⇒ 単調安定化
  - **補助メトリクス**:
    - `epochs_run` 平均値 < cfg.epochs (=2) なら early stop が機能している証拠
    - `clip_fraction` の平均値 < 0.3 (現状 0.35-0.5)
    - `policy_loss` / `value_loss` curve の単調性
  - **採否しきい値**:
    - (a) + (b) の両方を満たす → **adopted**
    - 片方のみ → **inconclusive** (deepen 候補)
    - 両方 NG → **rejected**
  - vs baseline_v1 / random は 50 ep ずつだけ補助記録 (n<300 inconclusive 固定)

## リスク / 既知の不確実性

- target_kl=0.05 が **厳しすぎ**て iter0 の高 approx_kl 環境では **毎 epoch すぐ break** され学習が進まない可能性。その場合は H1 修正版として `target_kl=0.10` を deepen 候補に
- early stop が完全に効くと PPO update が薄くなるため、`win_rate` 改善は iter1 単独では限定的。安定化効果のみを評価することに注意
- iter0 baseline の 100 iter 完走結果がまだ DVC pull 待ち。完走 metrics で iter0 比較ベースを確定させてから iter1 を回す方が良い (順序検討: 先に iter0 結果を回収)

# PoC: 20分GPU実験の観測性・失敗耐性 検証

時刻: 2026-06-06 / 対象: `bot/pipeline/reinforce/case7/`

## 目的
`fast_probe.yaml`（h500/ep8/20iter, GPU ~20分）で、以下5要件 + 失敗耐性を実証する。

1. iterごとにweight保存 → ローカルでrulebaseと対戦できる
2. 学習途中でも経過weightをローカル取得できる
3. 学習途中でもloss/勝率/相手/その他指標を取得・可視化できる
4. 学習途中でもGPUサーバにSSH接続し状況（エラー有無/ループ稼働/残り時間）を確認できる
5. 小規模実験でも大規模化前提の効果検証ができる
   - **失敗耐性**: 途中失敗してもweight/ログが消えない

## 実装した変更
| ファイル | 内容 |
|---|---|
| `training/train_jax.py` | `_upload_artifact_to_s3()` 追加。**毎iter** ckpt + metrics.json をS3 upload（best更新時のみ→全iterへ）。失敗耐性のコア |
| `configs/fast_probe.yaml` | GPU 20分config。h500/ep8/20iter、in-JAX相手（self_snapshot pool + baseline_jax_full） |
| `evaluation/plot_metrics.py` | metrics.json → 学習曲線PNG（win/reward/loss/entropy/KL/time）。途中metricsでも描画可（要件3） |
| `evaluation/eval_ckpt_vs_rulebase.py` | 任意 ckpt_i*.pt をtorch変換→rulebase対戦（要件1）。`ORBIT_WARS_CASE7_WEIGHTS` 経由 |
| `policy/agent.py` | `ORBIT_WARS_CASE7_WEIGHTS` で weights.pt を上書き可能に（任意ckpt評価用、Kaggle submitは不変） |

## ローカル事前検証（GPU投入前のコード健全性）
- `train_jax` 2iter smoke（h60, CPU, foreground）: 完走、`ckpt_i000.pt`/`ckpt_i001.pt`/`metrics.json` 生成 ✅
- `plot_metrics`: 学習曲線PNG生成 ✅
- `eval_ckpt_vs_rulebase`: ckpt_i001.pt→torch変換→baseline_v8と4戦 完走（win 0.0、smoke未学習なので妥当）。**変換+対戦パイプライン動作確認** ✅
- lint/format: 全green ✅

## 各要件の検証手順（GPU run）

### 起動
```bash
git push origin feature/reinforcement-learning-pooling-simple
dev/runpod dev "$(git rev-parse HEAD)" --case case7      # interactive pod (SSH可)
dev/runpod status <run_id> --case case7                  # 50_interactive_ready 待ち
```

### 要件4: SSH状況確認（学習中）
```bash
dev/runpod ssh <run_id> --case case7                     # 接続
dev/runpod tail <run_id> --source train                  # ①学習ループ稼働・②エラー有無
dev/runpod tail <run_id> --source gpu                    # GPU稼働
dev/runpod status <run_id> --case case7                  # ③ iter進捗→残り時間推定
```
- **①ループ稼働**: train.log に `iter=N ... rollout=... win=...` が増えるか
- **②エラー**: `Traceback`/`Error`/`OOM` の有無
- **③残り時間**: (iterations - 現iter) × 平均 iter秒

### 要件2/3: 途中weight取得 + 可視化
```bash
# 学習完了を待たず、途中のrun_dirをpull
dev/runpod sync <run_id> --case case7 --pull             # ckpt_i*.pt + metrics.json取得
# or S3経由（失敗耐性経路と同一）
dev/runpod logs <run_id>                                 # S3 markerでupload確認

uv run python -m pipeline.reinforce.case7.evaluation.plot_metrics \
  --metrics <run_dir>/metrics.json --out /tmp/curves.png
```

### 要件1: iter別weight × rulebase対戦
```bash
uv run python -m pipeline.reinforce.case7.evaluation.eval_ckpt_vs_rulebase \
  --ckpt <run_dir>/ckpt_i010.pt --baseline baseline_v8 --episodes 30 --seed 0
```

### 失敗耐性: kill復旧シナリオ
```bash
# 学習を5iter回した後、故意にtrainプロセスをkill (pod preempt相当)
dev/runpod ssh <run_id> --case case7 --exec "pkill -f train_jax"
# ローカル/S3から iter4 までの ckpt + metrics が残存していることを確認
dev/runpod pull <run_id> --case case7 --from s3
ls <run_dir>/ckpts/   # ckpt_i000..004.pt が揃う = 復旧成功
```

### 要件5: 大規模化前提の効果検証
- 20iterの **学習曲線の傾き(trend)** で「続ければ伸びる方向か」を読む（plateau絶対値はGPU本番へ委譲）。
- paired-seed評価（`08-fast-validation-methodology.md`）で少戦数でも方向性を有意判定。

## 結果
<!-- TODO: GPU run 後に各要件の合否・所要時間・コストを記入 -->

| 要件 | 合否 | 証跡 |
|---|:--:|---|
| 1. iter別weight×対戦 | | |
| 2. 途中weight取得 | | |
| 3. 指標取得・可視化 | | |
| 4. SSH状況確認 | | |
| 5. 大規模化前提の効果検証 | | |
| 失敗耐性（kill復旧） | | |

- /iter時間: <!-- --> s → 20iter ≒ <!-- --> 分
- GPUコスト: $<!-- -->

# imitation/case6 — Attention Backbone (iter1 中断結果)

> 作成日: 2026-05-04
> 関連: [`plan.md`](./plan.md)
> 結論: **学習未完了 (5 連続 RunPod 失敗で打ち止め)**。preprocess は 944 episodes
> まで成功、attention backbone のコード+test は確定。case6 の学習・評価は次サイクル送り。
> 本 iter で得られた知見はインフラ修正 5 件として canonical コードに commit 済。

## 1. 結果サマリ

| 項目 | 状態 |
|---|---|
| case6 実装 (model.py の GraphAttention 化) | ✅ 完了 (commit `0e85032`) |
| 単体テスト (forward shape / mask / NaN safe / param count) | ✅ 14/14 pass |
| RunPod 学習 | ❌ 5 連続失敗、累計 $1.59 消費して打ち止め |
| best.pt / metrics.json 取得 | ❌ train 未到達のため artifact 無し |
| vs baseline_v1 評価 | ❌ weights 無く実施不可 |
| **plan.md 仮説の検証** | ❌ **未検証** (attention backbone が効くか不明のまま終了) |

## 2. 5 連続 RunPod 失敗の内訳

各 run の commit / 失敗 marker / 主因。

| # | run_id (短縮) | commit | GPU | 失敗 marker | 失敗原因 | 修正 commit |
|---|---|---|---|---|---|---|
| 1 | `20260504-094444` | `4a41cdd` | RTX 4090 | `45_dvc_pull_full_failed` | `dvc pull` が他 case (case5) の未学習 outs (`.dvc` stub) で fail | `2e484ee` (`--allow-missing` 追加) |
| 2 | `20260504-101450` | `2e484ee` | RTX 4090 | markers 0 件 25 分停滞 | `nvidia-container-cli: cuda>=13.0` driver 不足、container init 失敗 | `--image cu1241` で回避 |
| 3 | `20260504-104914` | `2e484ee` | RTX 4090 | `65_train_failed_exit_1` | `ImportError: mark_progress from runpod_io.progress` (case5 から欠落していた関数) | `628ad10` (実装追加) |
| 4 | `20260504-112440` | `628ad10` | RTX A6000 | `55_mart_dvc_add_failed` | preprocess 944 ep 成功 → `dvc add` が `data/mart/imitation` symlink 配下を拒否 | `88ba83a` (non-fatal 化) |
| 5 | `20260504-134147` | `88ba83a` | RTX 4090 | markers 0 件 11 分 | CUDA 13 trap 再発 (RTX 4090 ノードの driver 古め)、即停止 | (image 既に cu1241、ノードガチャ問題) |

## 3. インフラ起因問題と恒久対策 (本 iter で解消済)

| 問題 | 根本原因 | 修正 |
|---|---|---|
| **dvc pull が他 case の未学習 outs で fail** | `dvc.yaml` の他 case stage に未学習の outs (`.dvc` stub) があると `dvc pull` 全体が fail | `bot/src/runpod_io/onstart.sh.tmpl`: `dvc pull --allow-missing` |
| **mark_progress が import で死ぬ** | case5 を作った時に `runpod_io.progress.mark_progress` を呼ぶようにしたが、関数本体が未実装。case5 自体未走行で発覚していなかった | `bot/src/runpod_io/progress.py` に `mark_progress(run_id, step, payload)` + `get_run_id()` を追加 |
| **mart_dvc_add が symlink 配下で fatal** | `data/mart/imitation` を `/persist/data-mart-imitation` への symlink にしているため、新しい DVC が `dvc add` を拒否 | `onstart.sh.tmpl`: dvc add の失敗を warning に降格、parquet は local-only でも train は走るようにする |
| **CUDA 13 image が一部 RTX 4090 ノードで起動不可** | `runpod/pytorch:1.0.3-cu1300-...` (default) が `cuda>=13.0` を要求するが、RTX 4090 SECURE プールに driver 古めのノードが混在 | `--image runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2204` で CUDA 12.4 へ降格して回避 (ただしノードガチャは残る) |
| **case 独立性ルール違反** | 一時的に case6 を case5 のデータパスに乗せたが `.claude/rules/bot/pipeline.md` の cross-case independence 原則に違反 | `il_case6.yaml` を case6 独自 mart paths に戻す、preprocess_imitation_case6 stage を再追加 |

## 4. preprocess 段階で得られた事実 (run #4 から)

run #4 の preprocess は **944 episodes** を 16 分で完走 (RTX A6000 ノード):
- rating_cutoff: 1085.25
- episodes total: 944 (rating 上位 50% で実測。plan の想定 670 を大きく超過)
- train frames: 300,084
- val frames: 38,278
- 1 chunk (100 episodes) ペース: ~8.7 分

## 5. 採否判断

**採否: 中断 (rejected by infra failure, not by experiment outcome)**

- attention backbone の効果はデータ無く未検証
- ただし **コード自体は完成しており再起動 1 発 (~$0.7-0.9) で完走できる状態**
- 次サイクルの最初に再起動すれば、本 iter のインフラ修正のおかげで pre-fail なく走る見込み

## 6. 次サイクルへの引継ぎ

### 短期 (Cycle 4 候補)

1. **case6 RunPod 再起動 1 発** (`88ba83a` 以降の commit、A6000 を `--gpu-name` で明示指定推奨)
2. preprocess は今回の volume 残骸で skip される可能性あり (保証ない)
3. 完走時 ETA: ~25-50 分、想定コスト ~$0.30-0.85

### 中期改善ポイント

| 改善案 | 効果 | 優先度 |
|---|---|---|
| RunPod create_pod に `allowed_cuda_versions` を渡し driver 不一致ノードを除外 | CUDA trap 構造的回避 | 高 |
| onstart の `mart_dvc_persist` を local-only モード permanent 化 (parquet を `runpod_artifacts/` 経由 S3 に直接 upload) | symlink 制約 trap 回避 | 中 |
| preprocess.py を polars/Rust で 5-10x 高速化 | 1 run の preprocess を 90 分 → 15 分に | 中 |
| 軽量 self-test (`python -c "from runpod_io.progress import mark_progress"`) を `dev/test-bot` に追加 | mark_progress 系 import 漏れの再発防止 | 低 |

### 検証残課題 (case6 学習が完走したら)

- **vs baseline_v1 50戦** (plan の主指標)
- val loss 収束曲線 vs case5 (case5 自体も未走行のため絶対比較は困難、case1 と並べる)
- attention weight 可視化で「どの planet を重視しているか」の定性確認

## 7. 累計コスト記録

| run # | commit | コスト | 失敗段階 |
|---|---|---|---|
| 1 | `4a41cdd` | $0.10 | 45_dvc_pull |
| 2 | `2e484ee` | $0.31 | 0_container_init (CUDA 13) |
| 3 | `2e484ee` | $0.21 | 65_train (mark_progress) |
| 4 | `628ad10` | $0.84 | 55_mart_dvc_add (preprocess は完走済) |
| 5 | `88ba83a` | $0.13 | 0_container (CUDA 13 再発、即停止) |
| **合計** | — | **$1.59** | — |

うち run #4 の $0.84 は preprocess 完走済みの parquet 生成コストなので、**実質 5 件中 4 件が「インフラ trap で即時 fail」型の損失**。

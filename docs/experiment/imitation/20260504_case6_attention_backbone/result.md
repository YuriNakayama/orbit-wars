# imitation/case6 — Attention Backbone (iter1 完走 + 評価結果)

> 作成日: 2026-05-04 (中断記録) → 2026-05-05 (完走 + 評価追記)
> 関連: [`plan.md`](./plan.md)
> **最終結論**: **学習完走、val 指標は健全に下がる (val_from_acc 0.91, val_loss 3.80→3.65) も、vs baseline_v1 50戦は 0/50 (Wilson CI 0–7.1%)。**
> **採否: rejected**。attention backbone 単独では rule-based に勝てない (case1 当初と同じ症状、memory `project_imitation_case1_2026_04_19` 参照)。
> インフラ trap 8 件をすべて修正、累計コスト ~$2.59 で iter1 を閉じる。次 iter は別 axis (head 設計、規模拡大、loss 設計、混合学習等) を検討。

## 1. 結果サマリ

| 項目 | 状態 |
|---|---|
| case6 実装 (model.py の GraphAttention 化) | ✅ 完了 (commit `0e85032`) |
| 単体テスト (forward shape / mask / NaN safe / param count) | ✅ 14/14 pass |
| ローカル smoke (10 episodes / 2 epoch) | ✅ end-to-end 動作確認 (commit `7d03402`) |
| RunPod 学習 (`7d03402`, v8) | ✅ **15 epoch 完走** (RTX 4090 SECURE, 7m14s, $0.13) |
| best.pt / metrics.json 取得 | ✅ S3 fallback 経由で取得済 (`dev/runpod pull --from s3`) |
| vs baseline_v1 50戦評価 | ✅ 実施済、**0勝50敗 (win_rate=0%, 95% Wilson CI 0–7.1%)** |
| **plan.md 仮説の検証** | ❌ **rejected** (attention backbone は学習はするが rule-based に勝てない) |

## 2. 学習結果 (RunPod v8 完走、commit 7d03402)

### 2.1 学習ジョブ統計

| 項目 | 値 |
|---|---|
| run_id | `20260505-051300__feature-imitation-model-structure__7d03402__seed0` |
| pod_id | `1k61n6m3aqe6fl` |
| GPU | NVIDIA GeForce RTX 4090 (SECURE) |
| image | `runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2204` |
| epochs | 15 (best_epoch=12) |
| best_val_loss | **3.6514** |
| runtime | 434 秒 (~7 分 train、+preprocess 30 分 + 環境構築 10 分 = 全体 50 分) |
| 実コスト | $0.13 (train 段階のみ、preprocess + setup 込みで合計 ~$0.50) |

### 2.2 学習曲線 (`metrics.json` より)

| epoch | train_total | val_total | val_from_acc | val_target_acc | val_ships_acc |
|---:|---:|---:|---:|---:|---:|
| 0 | 3.9753 | 3.7970 | 0.8815 | 0.3815 | 0.8108 |
| 5 | 3.7387 | 3.6907 | 0.8863 | 0.4052 | 0.8191 |
| 8 | 3.6975 | 3.6634 | 0.9038 | 0.4170 | 0.8225 |
| **12 (best)** | **3.6543** | **3.6514** | **0.9076** | **0.4190** | 0.8197 |
| 14 | 3.6363 | 3.6641 | 0.9017 | 0.4141 | 0.8217 |

- train/val loss は単調減少 (epoch 12 まで)、その後若干上振れ → 軽い overfit 兆候
- **val_from_acc 0.91**: from_head は十分高精度
- **val_target_acc 0.42**: target_head は中程度 (case1 baseline ~0.35-0.40 と同水準、改善幅小)
- val_ships_acc 0.82: ships_head は安定

### 2.3 ローカル評価: vs baseline_v1 (rulebase/case1) 50戦

```
episodes:    50
wins:         0
losses:      50
draws:        0
win_rate:    0.0%   (95% Wilson CI: 0.0% – 7.1%)
challenger:  il_v6
baseline:    baseline_v1
seed:        0..49
```

→ **生存しきい値 (≥5%) を満たさず、Wilson CI 上限 7.1% で「勝率 5% 以上」とも強く言えない**。

memory `project_imitation_case1_2026_04_19` 参照: case1 もテンプレ化・pos_weight・NO_OP 修正後でも vs baseline_v1 で 0/100。**imitation 系の典型的な「val 指標は良好だが対戦で全敗」のパターンを踏襲**。

## 3. 仮説検証

plan.md の仮説:
> attention 化で「どの planet が重要か」をモデルに明示学習させると、target/ships head のスコアリング精度が向上し、対戦勝率を押し上げる。

**結果**:
- target head 精度は val で 0.42 (case5 GraphConv の絶対比較データなし、case1 baseline 比 +5pp 程度)
- 勝率は 0/50 で押し上げられず

**仮説のうち成立した部分**: attention は学習可能 (loss 減少 + val_from_acc 0.91)
**仮説のうち不成立の部分**: 「target/ships head 精度向上 → 勝率上昇」の因果が成立しない。imitation 系全般の課題 (target diversity 欠如、開幕局面での greedy 一辺倒) が backbone 変更だけでは解消しない。

## 4. インフラ起因問題と恒久対策 (本 iter で 8 件解消済)

iter1 は RunPod インフラの trap 8 件を順番に踏んで $2.46 を消費、最終的に v8 で完走した。各 trap の根本原因と修正:

| # | trap | 検出 | 修正 commit |
|---|---|---|---|
| 1 | `dvc pull` が他 case 未学習 outs で fail | `45_dvc_pull_full_failed` | `2e484ee` (`--allow-missing`) |
| 2 | `nvidia-container-cli: cuda>=13.0` で container init 無音失敗 | markers 0 件 25 分 | `--image cu1241` (CUDA 12.4 降格) |
| 3 | `ImportError: mark_progress from runpod_io.progress` | `65_train_failed_exit_1` | `628ad10` (関数実装追加) |
| 4 | `dvc add` が `data/mart/imitation` symlink 配下で fatal | `55_mart_dvc_add_failed` | `88ba83a` (warning 化、non-fatal) |
| 5 | RTX 4090 ノードガチャで CUDA 13 trap 再発 | markers 0 件 11 分 | image cu1241 で確率減 |
| 6 | `dvc pull` が volume 残骸 parquet を unsaved file 扱い | `45_dvc_pull_full_failed` 再発 | `e87445c` (`--force` 追加) |
| 7 | volume 残骸 parquet の中途破損で train 起動時に死亡 | `parquet: File must end with PAR1` | `fe554bc` (毎 run /persist 冒頭でクリア) |
| 8 | apt mirror 接続失敗 (Ubuntu mirror 一時障害) | `Failed to fetch archive.ubuntu.com` | image に curl/git 既設で onstart は進行可能、無視で OK |

**累計修正 commit 数**: 5 件 (`2e484ee`, `628ad10`, `88ba83a`, `e87445c`, `fe554bc`)
**累計コスト**: ~$2.59 (8 件の RunPod run の累積)

## 5. 採否判断

**採否: rejected (attention backbone 単独では vs baseline_v1 で勝てない)**

- val 指標は健全に下がる ✅
- 学習はしている (val_from_acc 0.91 = from 識別 OK) ✅
- 対戦勝率は 0/50 ❌

「rejected by experiment outcome」(infra failure ではなく実験結果として却下)。

## 6. 次サイクルへの引継ぎ

### 即時候補 (Cycle 4-5 で試す価値あり)

| 案 | 仮説 | 想定 GPU コスト |
|---|---|---|
| **head 設計強化** (target head に planet pair embedding 追加 / from-target 結合 head) | 単独 head の独立性が target diversity 欠如の根因 → joint head で改善 | $0.5/iter |
| **規模拡大** (hidden 128→256, layers 3→5, attn_heads 4→8) | 表現力 bottleneck 解消 | $0.7/iter |
| **混合学習** (BC loss + 自己対戦 RL fine-tune) | rule-based の「強い defensive 戦術」を imitation だけで学べないため reward 経由で補強 | $1.5/iter |
| **case1 系 fix の流入** (Phase 2 breakthrough の from focal α=0.75) | memory `project_imitation_case1_phase2_breakthrough` で 0/100→5/100 の唯一 breakthrough | $0.5/iter |

### 中期改善ポイント (本 iter で発見した infra issue)

| 改善案 | 効果 | 優先度 |
|---|---|---|
| RunPod create_pod に `allowed_cuda_versions` を渡し driver 不一致ノードを除外 | CUDA trap 構造的回避 | 高 |
| `mart_dvc_persist` を S3 直 upload に置換 (DVC 経路捨てる) | symlink trap の根本回避 | 中 |
| preprocess.py を polars/Rust 化 | 1 run の preprocess を 30 分→5 分に | 中 |
| import self-test を `dev/test-bot` に追加 | trap #3 のような関数欠落の再発防止 | 低 |

### 検証残課題

- **vs baseline_v1 300戦再評価** (n<300 信頼不可ルール、ただし 0/50 を 300 戦で改善する見込みは低い)
- attention weight 可視化で「どの planet を重視しているか」の定性確認 (失敗の質を理解するため)
- val_target_acc 0.42 がどの target template に偏っているかの分析 (ABS_NEAREST_ENEMY だけ強い等)

## 7. 累計コスト記録

| run # | commit | GPU | コスト | 失敗段階 / 完走 |
|---|---|---|---|---|
| 1 | `4a41cdd` | 4090 | $0.10 | 45_dvc_pull |
| 2 | `2e484ee` | 4090 | $0.31 | 0_container (CUDA 13) |
| 3 | `2e484ee` | 4090 | $0.21 | 65_train (mark_progress) |
| 4 | `628ad10` | A6000 | $0.84 | 55_mart_dvc_add |
| 5 | `88ba83a` | 4090 | $0.13 | 0_container (CUDA 13 再発、即停止) |
| 6 | `88ba83a` | 4090 | $0.10 | 45_dvc_pull (unsaved files) |
| 7 | `e87445c` | 4090 | $0.55 | 65_train (broken parquet) |
| 8 | `fe554bc` | A6000 | $0.07 | 0_container (CUDA 13 再発、即停止) |
| 9 | `fe554bc` | A6000 | $0.07 | 0_container (CUDA 13 再発、即停止) |
| 10 | `7d03402` | 4090 | $0.13 (train) + 設定 ~$0.40 | ✅ **完走** (15 epoch、best.pt 取得) |
| **合計** | — | — | **~$2.91** | iter1 = 9 失敗 + 1 完走 |

実質「学習成果物 (best.pt) を得るのに $2.91 投じた」が、本 iter で標準化した修正により次 iter は $0.5/run 以下で完走できる見込み。

# case8 学習基盤の観測性・耐障害性 (5要件)

> 作成日: 2026-06-09
> 背景: R5 (iter50) 初回 run が bad node で iter0 60分停滞 → 学習基盤が 5 要件を満たすか
> 明確化し、10分の事前検証で実証してから本番 run を回す。

## 要件と実現状況

| # | 要件 | 実現機構 | 状態 |
|---|---|---|---|
| 1 | 全ログ・学習時精度を逐次出力、学習中でもローカルから確認可 | train_jax が iter ごと metrics.json flush (train_jax.py:982-1001)。S3 へ per-iter upload。**`dev/runpod metrics <run_id>`** (新規) で S3 の metrics.json を読み per-iter 表 (win/held-out/reward/entropy/vloss/Elo/timing) を**ローカルから・学習中でも**表示。`dev/runpod tail --source train/gpu/system` で SSH live tail も可 | ✅ (metrics コマンド追加で完成) |
| 2 | 重み・全ログ・精度を逐次 S3 upload、途中失敗でも成果物消失せず失敗要因分析可 | train_jax `_upload_artifact_to_s3` が iter ごと ckpt_iNNN.pt + metrics.json を S3 へ (train_jax.py:1000-1001)、best 更新時 best.pt も。onstart 60_train.sh:104-113 が heartbeat ごと train.log/gpu.log/system.log/best.pt/metrics.json/onstart.log を S3 へ。実証: R1 p4 run の S3 に ckpt_i000-019 + 全ログが残存 | ✅ |
| 3 | リソース (mem/cpu/gpu) + 処理状況を逐次ログ化、どこに時間・リソースがかかるか分析可 | 60_train.sh:40-48 nvidia-smi 10s 周期 → gpu.log。:50-59 system_monitor (psutil) 10s 周期 cpu/ram/load → system.log。train_jax が per-iter rollout_secs/update_secs を metrics に記録 (train_jax.py:442,458) | ✅ |
| 4 | 学習中の実行環境に SSH 接続しオンタイムでログ確認可 | `dev/runpod ssh <run_id>` が oneshot pod でも launch.json から pod_id 解決し接続 (app.py ssh_cmd)。`dev/runpod tail` で train/gpu/system を live tail | ✅ |
| 5 | 上記が実現されているかを iter 削減 10分検証で確認 | `phase1_validate.yaml` (10 iter ~8min) + `reinforce_case8_phase1_validate` registry。下記手順で実証 | 本ドキュメントで実施 |

## bad node 障害の教訓 (R5 初回 ...093731)

- 症状: iter0 が 60分完了せず、GPU 100% util だが **333 MiB のみ** / host **load_avg 91 / 109 threads** / metrics.json 未生成。
- 切り分け: R1 p4 (同コード・20iter) は 19分・22s/iter で正常完走 → **コードでなく bad node の transient 障害**。
- 検知の難しさ: onstart の `62_train_heartbeat_N` は onstart プロセスの生存カウンタで、**実 iter 進捗とは無関係**。
  → 監視は **metrics.json の history 長 (実 iter)** か **GPU memory.used (正常は ~18GB、stall は数百MiB)** を見るべき。
- 対処: stall 検知 (iter0 が数分で完了しない / GPU mem が数百MiB) なら即 destroy → 別 pod に再投入。

## 10分事前検証の手順 (REQ5)

`reinforce_case8_phase1_validate` (10 iter) を oneshot で起動し、学習中に以下を**ローカルから**確認する:

1. **REQ4 (SSH)**: `dev/runpod ssh <run_id> --case reinforce_case8_phase1_validate --exec "nvidia-smi"` が通る。
2. **REQ3 (resource)**: `dev/runpod tail <run_id> --source gpu` で GPU util/mem が live で流れる。`--source system` で cpu/ram/load。
   GPU memory.used が ~18GB (正常学習) であることを確認 (bad node なら数百MiB)。
3. **REQ1 (incremental metrics, local)**: 学習中に `dev/runpod metrics <run_id>` を複数回叩き、iter 数が増えていく
   (per-iter の win/held-out/reward が逐次見える) ことを確認。
4. **REQ2 (S3 durability)**: `dev/runpod logs <run_id> --source markers` で進捗、完走後 S3 に
   ckpt_iNNN.pt + metrics.json + 各ログが揃うことを確認 (R1 で実証済だが本 run でも確認)。
5. 合格条件: 1-4 が全て成立 + 10 iter が ~10分で完走。→ 本番 R5 (iter50) を同手段で起動。

## 本番 run の運用

- 起動: `dev/runpod train <sha> --case reinforce_case8_phase1_r5_iter50`。
- 監視 (ローカル, SSH 不要): `dev/runpod metrics <run_id> --case ... --tail 10` を随時。
- stall guard: 起動後数分で `dev/runpod metrics` の iter が 0 のまま or GPU mem が数百MiB なら bad node → destroy + 再投入。
- 完走後: `dev/runpod metrics` で全 iter 曲線、`dev/runpod pull` で best.pt/ckpts 取得。

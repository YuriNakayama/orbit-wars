# case7 Pool 形式 RL — リスクと依存

## リスク一覧

| # | リスク | 影響 | 確率 | 緩和策 |
|---|---|---|---|---|
| 1 | **case8 が飽和相手化**(本物 parity ゆえ強く、勝てず勾配消失) | High | High | `exploiter_prob_cap=0.2` で混入抑制 + f_hard 自然減衰。CPU 検証(Step6)で reward/eval_win を監視、暴走なら cap↓ |
| 2 | **case8 featurizer の parity ずれ**(`build_world_features_from_state` が本物と微差→学習相手が別物) | Med | Med | Step1 で case8 単体 rollout の挙動を本物 case8 agent と突合。memory `project_rulebase_jax_parity_failure_mode`(float32 tie-break)に注意、x64 不要なら float32 許容 |
| 3 | **eval 相手バイアス**(eval_opponent に過適合し別相手で弱い) | Med | Med | iter15 で実証済(self-play win 無相関)。eval_opponent を case8(本物近似)にし、最終採否は別相手(rl_v0/v1)で外部確認 |
| 4 | **小規模で v1 0/10 の天井**(本機能でも v1 越え不可の可能性大) | High | High | memory 既知。目標を「弱〜互角相手攻略 + 設計健全化」に置き、v1 越えは GPU 段階(Step7)へ委譲。期待値を過大にしない |
| 5 | **PFSP rollout コスト増**(exploiter forward 分) | Med | Low | case8 は in-JAX で軽量(python_v* と違い host hop 無)。memory `project_reinforce_self_snapshot_cost` は python 相手の話。CPU で /iter 時間を実測 |
| 6 | **horizon 設定ミス再発**(<500 で terminal 報酬消失) | High | Low | config 正典化 + テストで horizon=500 を assert。memory 既知バグ |
| 7 | **best.pt=最新 への退行**(eval gate 実装漏れ) | Med | Low | Step3 で gate を eval_win に置換 + per-iter ckpt 併存。テストで best=eval最大 を検証 |
| 8 | **GPU コスト超過**(PFSP で uptime 長期化) | Med | Med | 段階拡大、cost cap、中間 S3 upload(規約)、uptime 手動監視。CPU 採用判定後のみ起動 |

## 外部依存
- `pipeline.rulebase.case8.baseline_jax`(内部、既存、72 unit + 2 e2e pass 済)。
- `orbit_wars_jax`(内部 JAX sim、parity-tested)。
- RunPod 基盤(Step7 のみ、既存 `dev/runpod`)。3090/4090 在庫切れ有(memory)、40min backoff。

## 技術的負債
- `training/` から `pipeline.rulebase.case8.*` を絶対 import(case 独立規約の例外だが
  training は submission 外なので許容)。case8 が消えると case7 train が壊れる結合は残る。
- 旧 loop_iter* config を `_archive/` に残す(削除しないため肥大)。

## 未決事項
- eval_opponent の最終選定(`baseline_jax_case8` か `il_v0` 相当か)→ Step6 CPU 検証で比較決定。
- exploiter に lite を含めるか(既定は full+case8 のみ、lite は飽和実証済で除外寄り)。
- GPU 段階の iterations 規模(research 準拠 H100×36h は非現実的、$ 上限内で最大化)。

# rulebase-to-jax — Risks and Dependencies

## Risk List

| # | Risk | Impact | Probability | Mitigation |
|---|------|--------|-------------|------------|
| 1 | **world_model 8-turn シミュレーションの port が複雑で full parity 到達せず** (最難所) | High | High | Step 2 を最初に隔離して検証。ledger 単位で parity test。到達不能なら該当 case のみ lite 近似に格下げ判断 (要 user 確認) |
| 2 | **float reduction 順序差で score tie がズレ action 不一致** | High | Med | tie-break ルール(index 最小)を Python/JAX 両方で明示統一。parity test で一致率を可視化 |
| 3 | **case5 monolith(2455行) の分解で挙動取りこぼし** | Med | High | mission マッピング表を先に作成 (Step 6 work item)。一致率が落ちた obs を replay して原因特定 |
| 4 | **全戦略を毎回計算する dispatcher が PFSP pool 拡大でコスト線形増** | Med | Med | 数種までは許容。問題化したら JaxMARL 流 group 別 vmap へ退避 (architecture に明記済) |
| 5 | **Python int seed の jit 再 trace でコンパイルキャッシュ肥大→SIGABRT** (過去事例) | High | Low | seed は host 側で state 構築、jit には array のみ渡す。`python-to-jax` skill の pattern 遵守 |
| 6 | **core_jax を case1 配下に置く設計が case 間結合を生む** | Low | Med | Python 版が同構造 (case1/baseline/core 共有) なので許容。将来 simulator/jax への昇格余地を残す |
| 7 | **live 300 戦で互角にならない (train/eval ギャップ再来)** | High | Med | action 一致率 100% を先に達成してから live 検証。不一致なら parity test に戻る (NFR 順序を厳守) |
| 8 | **case4 internal rollout(case3 由来) が JAX 化困難** | Med | Med | rollout を並列 score で近似 or 該当部分のみ簡略化。Step 7 で扱い判断 |
| 9 | **8-turn ループを `lax.scan` で書くと GPU で unroll より遅い** (kernel launch overhead, GPU 固有) | Med | Med | HORIZON=8 と短いので Python unroll を初版に。コンパイル時間/メモリ過大時のみ scan へ。Step 10 bench で確定 ([jax#16611](https://github.com/jax-ml/jax/issues/16611)) |
| 10 | **全 mission/全 opponent を毎回計算 (jnp.where 両branch評価) で大盤面 OOM** | Med | Low | MAX_PLANETS=48 と小さく実害低。score 行列 `[48,48,5]` は軽量。万一は mission 数を必要分に絞る |

## External Dependencies

- `orbit_wars_jax` (simulator/jax): EnvState/reset/step に依存。変更なしで利用。
- `reinforce/case6`: opponent enum / rollout_jax / PFSP pool への統合先。
- RunPod GPU (`dev/runpod`): bench / 速度検証。memory `project_runpod_3090_4090_stockout` — 在庫切れ時 40min backoff。
- `python-to-jax` skill: TDD parity workflow の規約源。

## Technical Debt

- core_jax を case1 配下に置くため、長期的には `simulator/jax` 等への共有 core 昇格が望ましい (本イテレーションでは見送り)。
- case0/3/6–9 の port 未実施 → opponent pool は 4 種に留まる。
- `baseline_jax_full` (既存) と新 full port の二重管理可能性 (Step 4 で吸収 or 廃止を判断)。

## Open Items

- Step 4: 既存 `baseline_jax_full` を core_jax に吸収して廃止するか温存するか (実装時判断)。
- Step 7: case4 internal rollout の JAX 化方針 (full port / 近似 / 簡略)。
- live 勝率が互角にならない場合の許容しきい値 (45–55% で合意済だが port 別に再確認余地)。

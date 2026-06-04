# rulebase/case1 JAX full port — plan

> 作成日: 2026-06-03 / 状態: in_progress / ループ: /loop 10m (cron c0a10ea6)
> 主要メトリクス: action 一致率 (JAX vs 本物 Python, 目標 100%) + 0勝回避 (4game tripwire 最低1勝)

## 背景・目的

rulebase/case1 baseline_v1 を **full faithful JAX port** する。最大の問題は過去の JAX 化が
本物に劣化し**勝率ほぼ0**になったこと (memory `project_rulebase_jax_parity_failure_mode`)。
root cause = float32 reduction 差 → argmax tie-break ズレ → 別 action → 盤面発散。

PoC0/1/2 で full port が FEASIBLE と実測確定済 (docs/plans/rulebase-to-jax/09)。

## フロー (ユーザー指示)

- **Step1**: ローカル高速 (1試合≤10分) な結合テスト確立。大規模検証は避け 0勝回避の最小テスト。→ ✅ 完了
- **Step2**: JAX 書き換え方針を検証し、高速 かつ 劣化なしの実装方法を考える。← いまここ
- **実装**: PoC 完了後 rulebase JAX化を実装、随所で高速性+劣化なしを確認。

## Step2: 書き換え方針 (検証対象)

劣化(0勝)を避ける核心は **float32 でも argmax 選択が本物と一致すること**。方針:

1. **tie-break 統一**: 本物 Python の sort/argmax 順序を JAX で再現。score 同値時は
   index 最小等のルールを明示。`score - eps*index` で決定論化。
2. **x64 で先に parity 確立 → float32 で劣化幅を測定** (07 原則2)。x64 で 100% 一致を
   確認してから float32 化、落ちる分を tie-break 統一で吸収。
3. **ボトムアップ実装**: core_jax (geometry/physics, aim は case2 流用) → worldmodel
   (timeline/keep_needed/resolve_arrival) → missions (score chain) → mission_resolver
   (固定長 scan)。各層で本物の同名関数と x64 完全一致を確認 (差分テスト)。
4. **結合テストを進捗ゲージに**: action 一致率 21%→100%、tripwire 0/4→最低1勝 を
   実装の各段で測定。

## 検証方法 (skip list 反映)

- ローカル CPU で結合テスト (`tests/e2e/.../test_agent_jax_identity.py`)。1試合 18.6s。
- 大規模 win-rate eval は**行わない** (ユーザー指示: 数十対戦回避)。0勝 tripwire (4game) で代替。
- 速度計測は GPU bench (RunPod) を full port 完成後に実施。
- 行き詰まったら web search で情報収集。

## 成功基準

- action 一致率 100% (float32, 全 turn, 複数 seed)
- 0勝 tripwire GREEN (最低1勝、mirror match で ~50% 期待)
- 速度: vmapped self-play が python backend より高速 (GPU bench, 後段)

# 実験規模の縮小方針 (10h → 20分〜2h)

時刻: 2026-06-06 / 対象: `bot/pipeline/reinforce/case7/`

## 目的
本格 RL 学習は 10h+/$ がかかる。施策(Minimax reward / 逆カリキュラム / pool 構成等)の
**有効性を 20分〜2h で検証**できるよう実験規模を縮小する。ただし縮小で
**有効性検証そのものが無効化しない**よう、削る軸と守る軸を分ける。

## 縮小の原則
> **計算量を決める軸を削り、学習の質(=有効性検証の妥当性)を決める軸は守る。**

## 守るべき軸 (削ると有効性検証が無効化する)

| 軸 | 値 | 削れない理由 |
|---|---|---|
| **horizon** | **500 固定** | <500 でゲーム(~497turn)未終了 → 勝敗報酬が毎 step 0 → 勝率自体が無意味。memory `project_reinforce_horizon_terminal_reward_bug` の既知バグ |
| **network / batch** | full 維持 | 構造を変えると施策効果が比較不能になる(施策 A/B の前提が崩れる) |
| **episodes/iter** | **8-16** | batch を細らせるとノイズが増え、施策の有効/無効が判別できなくなる。削りすぎ厳禁 |

## 削るべき軸 (計算量を直接決める)

| 軸 | フル | 縮小 | 短縮率 |
|---|---|---|:--:|
| **iterations** | 200+ | **20-40** | 5-10× |
| **評価戦数** | 300 | **30-60** | 5-10× |
| **opponent** | 重い python_v* (pure_callback host hop) | **軽い in-JAX (case8 / self_snapshot)** | 数× |
| **並列度** | 直列 | **vmap で env 並列** | コア数分 |

## 10h → 20分〜2h の内訳
- iterations 200 → 30 (約 6×)
- python_v1 host hop → in-JAX case8 (rollout が相手 forward 分軽くなる、memory `feedback_jax_selfplay_foreground_only` 系の重さ回避)
- 評価 300戦 → 60戦 (5×、精度は下記 paired で担保)
- env を vmap 並列 (CPU コア数分)

## 評価戦数を削っても精度を保つ鍵 — paired-seed 評価
30-60戦に削ると単独勝率はノイジー(本 project 既知: 勝率が 1.0⇄0.17 振動、n<300 不信)。

**対策**: 同じ初期 seed 集合(同じ初期局面)で baseline と施策ありを対戦させ、
各 seed の差分 `Δ = 施策あり − baseline` を見る (common random numbers)。
絶対勝率がブレても **差は安定して検出**できる。詳細は
[`08-fast-validation-methodology.md`](08-fast-validation-methodology.md) の paired-seed 節を参照。

## 縮小 config テンプレ (`fast_probe.yaml`)

```yaml
training:
  iterations: 30          # 200+ → 30 (削る)
  episodes_per_iter: 8    # 守る (ノイズ抑制)
  horizon: 500            # 守る (必須、削ると勝率無効)
  shaping_mode: ratio     # 既存最良 (変更しない)
  shaping_coef: 1.0
  # opponent は in-JAX (case8 / self_snapshot)、python_v* は使わない
  # 評価は paired 30-60戦 (eval ハーネス側で実施)
```

network/batch 系 (hidden=192, heads=8, inducing=24, minibatch=128) は本番と同一を維持。

## まとめ
- **削る**: iterations(200→30) / 評価戦数(300→paired 60) / opponent(python→in-JAX) / 並列(vmap)。
- **守る**: horizon=500 / network・batch / episodes(8-16)。
- これにより「10h → 20分〜2h」で施策の有効性を検証でき、かつ paired-seed で
  少戦数でも有意差を検出できる。`fast_probe.yaml` を標準の縮小入口とする。

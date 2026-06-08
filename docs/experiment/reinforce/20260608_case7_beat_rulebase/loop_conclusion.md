# beat-rulebase ループ 結論: from-scratch RL は競争相手に勝てない (戦略的天井)

時刻: 2026-06-09 / core_jax(忠実v1 port)を学習相手に、本物 rulebase 勝利を目指した自律ループの結論。

## 背景
parity 調査で「H4 の勝利は parity 10% の偽相手に勝っただけ」と判明。忠実な学習相手
(core_jax: action 80%・win 90% vs v1・176ms高速)を整備し、それを相手に学習すれば本物に
転移するはず、という仮説でループを回した。

## 試した施策と結果 (すべて win 0 vs core_jax)
| 実験 | アプローチ | vs core_jax win | 教訓 |
|---|---|---|---|
| L1 | noop→core_jax 直接 | 0 (6 iter) | sparse-gradient 壁 |
| L2 | f_hard self_snapshot pool + core_jax | 0 | f_hard が unbeatable core_jax に集中、自己対戦を害す (memory unbeatable_opponent_harmful) |
| L3 | handicap 60% (ship×0.6) | 0、reward -3.4 = full と同一 | 弱体化しても勝てず = 資源でなく戦略の問題 |
| L3b | handicap 25% (ship×0.25, 76%減) | **0、reward -3.1** | 76%隻減でも全敗 |

## 結論: 戦略的天井 (資源でなく戦略)
- **76%隻減の crippled core_jax でも from-scratch agent は 0 勝**。reward は full(-3.4)/60%(-3.4)/
  25%(-3.1) でほぼ不変 → **opponent の強さが問題でなく、agent が coherent な戦略を作れない**。
- handicapping (研究 處方B) は資源差を埋める施策なので、戦略差には無力。
- これは memory `project_reinforce_case6_live_eval` の「小規模 RL は本物相手 0/30」天井を、
  **忠実な相手で・研究施策を尽くした上で再現**した。parity 調査により「測定の交絡」を除去した結果、
  **真の天井 = from-scratch PPO は競争相手 (faithful か否かに関わらず) に勝てない** ことが確定。

## 残る唯一の lever (未実施)
- **逆カリキュラム (處方A)**: 勝利寸前の局面から開始し terminal-win 報酬を取る→後方拡張。
  資源でなく「勝ち方」を直接教えるので戦略的壁に唯一効きうる。ただし有利 reset state の注入機構が必要 (大規模)。
- **BC warm-start (parity featurizer)**: imitation 済 policy から開始。H3 は case9 BC が case7 に転移せず失敗。
  同 featurizer の imitation 重みが要る。
- 大規模 (300+ iter) も memory `case1_aa_300iter` は self-play reward 0.50 止まりで rulebase 勝利は未達。

## ループの確定成果 (commit 済)
- core_jax 忠実化: x64 scan-carry バグ修正 + per-source cap 除去 → action-parity 80%・高速176ms。
- core_jax / core_jax_weak を rollout opponent に統合 (mode 7/8)、pool full_opponent 設定可能化。
- JAX self-play hang は iter5-137 で非決定的に再発、crash-safe S3 で結果保全 + 再起動で対処。

## 運用知見
- core_jax_weak (handicap) opponent は iter5 で hang 多発 (重い graph)。scatter→stack で軽減試行。
- 各施策で opponent が変わると JAX 再 compile (~30s) + hang リスク。短 run + 再起動が現実的。

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

## 段追記: 逆カリキュラム (處方A) も無効 — 資源でなく "そもそも打てない" が壁
agent_advantage (episode開始時に agent の planet ships を ×N) を実装し、未学習(ランダム)
policy で win_rate を測定 (horizon 500, 4戦):
| advantage | outcomes (cumulative) | win |
|---|---|---|
| 1.0 (なし) | -1.85〜-2.0 | 0/4 |
| 4.0 | -2.0〜-2.3 | 0/4 |
| **10.0** | -2.2〜-2.4 | **0/4** |

- **10× の隻数優位でもランダム policy は core_jax に全敗**。outcomes は ±1 でなく -2.4 = terminal敗北
  (-1) + 負の shaping。10倍の戦力を持っても**まともに打てないので負ける**。
- = ボトルネックは資源(opponent handicap でも agent advantage でも)解決不能。**from-scratch の
  ランダム policy が core_jax 相手に coherent な手を一切打てない**ことが根本。
- 逆カリキュラム(處方A)も、agent が最低限打てる前提が崩れているので foothold を作れない。

## 最終結論 (本ループ)
**この RL 設定 (from-scratch PPO, 2-head, ratio shaping) は competent な相手に対して
"競争力をブートストラップできない"。** noop には勝つ(0.8)が、core_jax(忠実 v1)には:
- 直接(L1)/pool(L2)/opponent handicap 60%・25%(L3/L3b)/agent advantage 10×、**全て win 0**。
全研究施策(處方A逆カリキュラム/處方B handicap/PFSP pool/scale)を忠実な相手で尽くした上での
**確定的天井**。memory `case6_live_eval` を parity 交絡を除去して再現・強化した。

**勝つために本質的に必要なもの (本ループのスコープ外)**:
1. **同 featurizer の imitation BC warm-start** — sensible に打てる初期 policy。H3 は case9(別featurizer)で失敗。
   case7 featurizer 対応の imitation 重みを作る必要がある (別タスク、大規模)。
2. or 本物 env での大規模学習 (高速基盤要)。
3. RL 単独・from-scratch・小〜中規模では rulebase 勝利は不可、が本プロジェクトの再確認された結論。

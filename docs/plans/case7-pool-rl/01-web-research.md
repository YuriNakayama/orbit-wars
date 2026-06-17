# case7 Pool 形式 RL — Web / 外部技術リサーチ

本機能は新規 OSS 統合ではなく **PFSP(Prioritized Fictitious Self-Play)/ league play** の
確立手法を既存 case7 に適用するもの。リサーチは過去 loop(15 iter)で蓄積した
知見 + 標準的な self-play RL 文献に依拠する。

## 確立された手法 (リファレンス)

### PFSP / Fictitious Self-Play (AlphaStar league)
- **手法**: 過去の自分の snapshot を pool 化し、勝ちにくい相手を優先サンプル
  (`(1-win_rate)^p`)。case7 の `_PrioritizedOpponentSelector` (f_hard) はこの実装。
- **適用**: pool に「main agents(self)」+「exploiters(rule)」を混ぜると forgetting を
  防ぎつつ特定弱点を突かれにくくなる。**→ 本計画で case8(本物 parity rule)を
  exploiter 枠として追加する根拠**。
- **gotcha**: 強すぎる相手だけだと勾配消失(case7 で lite/full=飽和=有害を実証)。
  exploiter は **低混入率 + f_hard 自然減衰** で「たまに当たる強敵」に留める。

### PBRS (Potential-Based Reward Shaping)
- ratio potential `Φ=mine/(mine+enemy)` の差分加算は最適方策を変えない(理論保証)。
  case5 H4(ratio/1.0)が最良、combined は係数で爆発。**本計画では現状維持**。

### スケール要件 (Generals.io, arXiv 2507.06825)
- 同レシピ(BC warm-start + self-play + PBRS + memory)で領土拡張ゲーム攻略に
  **H100×36h**。case7 loop は CPU 数分 = scale 1000倍以上不足 → v1 0/10 の構造的要因。
- **含意**: 「v1 越え」を本気で狙うなら GPU 本番化が前提。小規模では「弱〜互角相手に勝つ」が現実的到達点。

## 過去 loop の実証データ (一次情報)
`docs/experiment/reinforce/20260603_case7_beat_baseline_loop/SUMMARY.md` より:
- iter12(self-play ladder 16 iter): **vs rl_v0 1.00**、vs v1 0/10。
- iter25 定量化: iter12 model vs baseline_jax_lite win **0.17** / full **0.17**(大半 -2.0 飽和)。
- iter15: per-iter ckpt sweep で「rl_v0 勝率が iter 毎 1.0⇄0.17 振動、self-play win と**無相関**」。

## OSS / 内部 参照パターン
| パターン | 出所 | 本計画での採否 |
|---|---|---|
| FIFO past-self pool | AlphaStar league(main agents) | ✅ 主軸維持(cap=4) |
| PFSP f_hard 優先 | AlphaStar(`(1-p)^p`) | ✅ 維持、case8 entry 追加 |
| exploiter 混入(本物 rule) | league exploiter 枠 | ✅ **case8 を低率混入** |
| in-JAX rule opponent | case8 `build_world_features_from_state` | ✅ host hop 回避の鍵 |
| 飽和相手の害 | case7 自前実証 | ⚠️ 混入率を抑制で回避 |
| 外部 eval で model 選択 | case7 iter15 実証 | ✅ train 内に in-JAX 軽量 eval 統合 |

## Research Summary — 推奨アプローチ
1. **pool 構成 = past-self(主) + case8 本物 parity(低率 exploiter) + lite/full(任意)**。
   case8 は in-JAX で軽量、本物 v1 に最も近い「学習可能な強相手」(SUMMARY が欠落と指摘した枠を埋める)。
2. **f_hard で飽和を自然減衰**: case8 に連敗すると win_ema↓→出現率↑だが、勝てないと勾配消失。
   `priority_p` と混入上限で「たまに当たる」に調整。
3. **model 選択を train 内 in-JAX eval に制度化**(iter15 の手動 sweep を自動化)。
4. **小規模(CPU ~20min)で設計検証 → 採否確定後に GPU 本番化**(段階的、コスト管理)。

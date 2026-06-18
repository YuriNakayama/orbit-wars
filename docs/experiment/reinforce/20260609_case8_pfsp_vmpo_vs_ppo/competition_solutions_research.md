# 過去コンペ上位解法の調査 → 今のパイプラインへの工夫 (2026-06-16)

> 目的: Orbit Wars (=Planet Wars 現代版) に近い RTS系コンペ (microRTS / Lux AI /
> Halite / Kore / Bot Bowl / Planet Wars) の「scripted相手を破った」解法から、
> case8 reinforce パイプラインに足すべき工夫を抽出。

## 調査結果サマリ (scripted相手に勝った解法の共通項)

| コンペ / 解法 | 勝った手法 | 今のパイプラインとの差分 |
|---|---|---|
| **microRTS RAISocketAI** (初のDRL優勝、過去5回はscripted優勝) | GridNet action空間 + **invalid action masking** + 反復fine-tune + per-map転移 | **action masking が弱い/無い**、per-map転移なし |
| **Lux AI S1 winner** (DRL) | GridNet + **小マップでshaped→フルマップでsparse の難度curriculum** + IMPALA/UPGO | shaping はあるが**「易→難の段階移行」curriculum が無い** |
| **MimicBot (Bot Bowl)** | **純RLもcurriculum も scripted に勝てず、IL+RL hybrid で勝利** | **IL(BC)を未投入** ← campaign の最大欠落 |
| **Planet Wars winner (bocsimacko)** | **knapsack opening (FastExpand)** + sniping + reacquire の戦略primitive | RL は serial-knapsack 級の序盤最適化を発見できていない (ladder診断と一致) |

## 今の case8 パイプラインに足すべき工夫 (優先順)

### ★1. IL (behavior cloning) の導入 — 最大の欠落
- **根拠**: MimicBot は「純RL・curriculum が scripted に勝てなかった」状況 (=campaign と同一) を
  **IL+RL hybrid で解決**。AlphaStar/OpenAI Five も SL初期化が前提。campaign は IL を一度も
  投入していない = 文献上最も効く未投入レバー。
- **case8 への適用**: strict_v1 は決定的・安価・in-JAX の教師 → DAgger のラベル高コスト問題が
  無い。学習者が訪れた任意state で `_strict_v1_actions` を正解として収集 → BC warm-start +
  KL anchor (bc_warmstart/kl_beta 配線済) → RL洗練。

### ★2. invalid action masking の強化
- **根拠**: microRTS優勝 (RAISocketAI)、OpenAI Five、PySC2、AlphaStar が全採用。
  「無効action を masking し action空間を大幅縮小」→ 探索効率が劇的改善。
- **case8 への適用**: 現状 my_planet_mask で発射元はmask済だが、**ターゲット選択の無効手
  (戦力不足で取れない/sun軌道に当たる/到達不能) を logit でmaskしていない**可能性。
  aim診断で「87%が必要shipの半分未満で発射」= 無効target を選び続けている → これを
  masking すれば「取れるtargetだけ」に探索が集中し、zero-variance を緩和できる見込み。

### ★3. 難度curriculum の「易→難 段階移行」 (Lux S1式)
- **根拠**: Lux S1 winner は「小マップで shaped → フルマップで sparse」と**難度を段階移行**。
  campaign の T0/handicap は「相手を弱める」だったが効かず。Lux式は「**問題そのものを
  易しく**してから難化」。
- **case8 への適用**: BC で序盤を獲得後、shaped reward 主体の短horizon/弱相手 → sparse
  win-loss のフルstrict へ段階移行。IL と組み合わせるのが筋。

### ☆4. 戦略primitive の特徴量化 (Planet Wars winner式)
- **根拠**: Planet Wars 優勝 bot は **knapsack opening (FastExpand)** で序盤の最適拡張を
  解いた。RL が serial 最適化を発見できないなら、**「knapsack 的序盤拡張候補」を特徴量/
  candidate として与える**ことで序盤の意思決定を補助できる。
- **case8 への適用**: featurizer に「今ターン安全に取れる planet 集合 (knapsack解)」を
  候補特徴として追加。effort 中、IL/maskingの後の改善。

## 結論: パイプラインへの工夫の優先順位

1. **IL (BC) bootstrap + KL-anchored RL** ← 文献最強・最大欠落 (MimicBot/AlphaStar)
2. **target-選択の invalid action masking 強化** ← microRTS優勝の核、探索効率
3. **易→難 難度curriculum** (BC後に shaped→sparse 段階移行) ← Lux S1式
4. (将来) knapsack序盤候補の特徴量化 ← Planet Wars winner式

→ 1+2 を組み合わせた ladder21 が本命。IL で序盤戦略を教示し、masking で無効手を排除し、
KL anchor で strict戦略を保ちつつ RL で上回る点を探す。

## Sources

- [A Competition Winning DRL Agent in microRTS (arXiv 2402.08112)](https://arxiv.org/abs/2402.08112) — RAISocketAI、GridNet + invalid action masking + 反復fine-tune/per-map転移で初のDRL優勝 (過去5回scripted優勝)
- [Gym-µRTS (arXiv 2105.13807)](https://arxiv.org/pdf/2105.13807) — invalid action masking の効果
- [MimicBot: Combining Imitation and RL to win Bot Bowl (arXiv 2108.09478)](https://arxiv.org/abs/2108.09478) — 純RL・curriculum が scripted に勝てず IL+RL hybrid で勝利
- [Lux AI winning DRL agent — microRTS paper 内引用 (Pressman 2021)](https://arxiv.org/html/2402.08112v1) — GridNet + 小マップshaped→フルマップsparse の難度curriculum + IMPALA/UPGO
- [Planet Wars 優勝 bot bocsimacko (Gabor Melis)](https://github.com/melisgl/planet-wars) — knapsack opening (FastExpand) + sniping + reacquire
- [DAgger / imitation docs](https://imitation.readthedocs.io/en/latest/algorithms/dagger.html) — 決定的教師の on-policy ラベル集約
- [DeepMind AlphaStar](https://deepmind.com/blog/article/AlphaStar-Grandmaster-level-in-StarCraft-II-using-multi-agent-reinforcement-learning) — SL初期化 + KL正則化付きRL

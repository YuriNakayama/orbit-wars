# strict_v1 に勝てない原因 — web search 分析 (2026-06-16)

> 文脈: campaign ladder1-19 で curriculum/量/aim/reward+強制 を投入も held-out strict_v1 は ~0% (床)。
> 角度ロジックも再確認済 — 射出角は env の (y,x) swap 規約に整合 (atan2(Δx,Δy)) で正しく、
> ladder18 の intercept 修正も parity test 一致。aim はバグでない。

## 結論: 文献は campaign の実証診断を全面的に裏付ける

### 主因 = sparse/zero reward での探索失敗 (ladder19 の zero-variance と一致)
- success state が稀だと **random exploration がほぼ到達しない**。「random agent が一度も
  得点しないと success/failure 報酬では学習不能」— これは ladder19 の「素strict win ~0.5-2%
  で勝利エピソードが無く方策勾配が立たない」と**完全に一致**。
- shaping を強めても (ladder13 terminal×8 / ladder19 terminal×2.5)、勝ちが無ければ勝利方向の
  勾配は作れない (報酬は敗北を増幅するだけ)。
- 方策不安定: 「良い性能に達しても頻繁に悪い方策へ revert、win率が乱高下し急落」— campaign の
  held-out が 0/64〜1/64 を行き来する挙動と一致。

### 有効策 (文献) — campaign の次手と一致
1. **Imitation bootstrap → RL refine (最有力)**: 専門家デモから初期方策を学習し RL で洗練すると
   **rule-based 相手に win率 90%超**、RL単独を大きく上回る。→ strict_v1 の手を BC で直接教示する案。
   注意: 「相手が強すぎ skill gap が大きいと BC 後も性能が初め上がって後で低下」= 大きな乖離は逆効果。
2. **Curriculum (opponent progression / 難度漸増)**: 弱い相手→強い相手と段階的に当て、最後に
   rule-based。「単独で rule-base に当てるより速く上回る」。→ **ladder20 handicap (boost h=3→1
   anneal で難度漸増)** がこれに該当。自己対戦の難度自動上昇 (TiZero/Pommerman) も同系。
3. **Curiosity / intrinsic reward**: 行動結果の予測誤差を内発報酬にし新規状態へ探索を促す。
   → reward が sparse な素strict で「勝ち以外の学習信号」を供給する補助策。

## campaign への含意 (優先順位)

| 策 | 文献裏付け | campaign 状況 |
|---|---|---|
| handicap で勝ち人工生成 (難度漸増 curriculum) | 強 (curriculum) | **ladder20 で実行中** ← 今ここ |
| BC で strict 手を直接教示 (imitation bootstrap) | 最強 (win90%超事例) | 次点。skill gap 警告に注意 (strict は強いので段階的に) |
| curiosity/intrinsic reward | 中 | 補助。sparse 緩和 |

→ **ladder20 (handicap) は文献の curriculum 策に合致**。もし handicap でも anneal 後 (h→1)
held-out が動かないなら、次は **BC bootstrap (imitation)** が文献上の最有力策。strict が
強相手なので「skill gap で BC 後に低下」を避けるため、strict段だけでなく弱め段からの段階 BC が筋。

## Sources

- [RL Agent for a 2D Shooter Game (arXiv 2509.15042)](https://arxiv.org/html/2509.15042) — sparse hit/kill 報酬での学習困難、BC初期化→RLでwin90%超、curriculum opponent progression
- [Dealing with Sparse Reward Environments (Daaboul, Medium)](https://medium.com/@m.k.daaboul/dealing-with-sparse-reward-environments-38c0489c844d) — random agent が得点しないと success/failure 報酬で学習不能
- [The Sparse Reward Problem (Rana, Medium)](https://medium.com/@bhagyarana80/the-sparse-reward-problem-shape-signals-without-cheating-faa1962cf339) — shaping の落とし穴
- [Curious Exploration / Return-based Memory Restoration (arXiv 2105.00499)](https://arxiv.org/pdf/2105.00499) — curiosity intrinsic reward
- [TiZero: Multi-Agent Football, Curriculum + Self-Play (arXiv 2302.07515)](https://arxiv.org/pdf/2302.07515) — 難度漸増 self-play curriculum
- [Multi-Agent Pommerman: Curriculum + Population Self-Play (arXiv 2407.00662)](https://arxiv.org/pdf/2407.00662) — opponent progression → rule-base を上回る、population に rule-based を混ぜ忘却防止
- [Two-stage training for AI robot soccer (arXiv 2104.05931)](https://arxiv.org/pdf/2104.05931) — 強相手からの advantageous sample BC、skill gap で性能低下の警告

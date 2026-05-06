# rulebase/case9 — anti_ping_pong (replay-driven analysis)

> 作成日: 2026-05-05
> 関連: `plan.md`, `result.md`, replay run 2026-05-05T02:43:31Z (4試合)
> 分析対象: seed 2000 (v9 大敗) / seed 2002 (v9 大敗) / seed 2003 (v9 圧勝) の 3 試合

## 結論 (一文)

**iter1 v9 が負ける試合は「劣勢に陥った瞬間に v9 の launches がほぼゼロに沈黙する」現象が発生** している。`_DISPATCH_HISTORY` が劣勢時に飽和し、anti-ping-pong cooldown が ほぼ全 src→dst pair を blacklist してしまうため、**reinforcement が完全機能停止**する。これが負け試合での v9 大敗 (5惑星まで縮小、score 比 1/100) の直接原因。

## 試合別サマリ

| seed | 結果 | 試合長 | v9 launches | v4 launches | v9 ping-pong | v4 ping-pong |
|---|---|---|---|---|---|---|
| 2000 | **v4 勝** (大敗) | 500 | 273 | **1,379** | 5 | 45 |
| 2002 | **v4 勝** (大敗) | 500 | 245 | 748 | 20 | 34 |
| 2003 | **v9 勝** (圧勝) | 225 | **800** | 476 | 51 | 28 |

**観察**: v9 が勝つ試合は **launch 量が v4 の 1.7 倍** (800 vs 476)。負ける試合は **v4 の 1/5 (273 vs 1379) に launch 量が縮小**。anti-ping-pong cooldown が「勝ちパターンでは効かず、負けパターンでは過剰に効く」非対称性を持つ。

## seed 2000 ターン別崩壊シーケンス (v9 大敗の主犯期間)

| 期間 (turn) | v9 launches/turn | v9 惑星数 | v4 惑星数 | 局面 |
|---|---|---|---|---|
| 80–90 | **3.7** | 14–16 | 11–14 | 互角 |
| 100 | 3.0 | 11 | 17 | わずかに劣勢化 |
| 130–141 | **2.7** (低下) | 12 → 11 | 16–17 | ジワジワ後退 (v9 launches も減少傾向) |
| 143–164 | **0.6** (ほぼ停止) | 10 → 5 → 5 | 18 → 23 → 23 | **完全沈黙、雪崩崩壊** |

t=143 以降の v9 actions/turn の生データ (一部抜粋):
- t=145: 2 launches | t=146: 3 | t=147: 2 | t=148: 1 | t=149: **0** | t=150: 1 | t=151: **0** | t=152: **0** | t=153: 1 | t=154: **0** | t=155: 2 | t=156: **0** | t=157: **0** | t=158: **0** | t=159: 1 | t=160: **0** | t=161: **0**

→ **20 ターンのうち 12 ターンで launch 数 0**。一方 v4 は同期間 3–8 launches/turn を維持。

## メカニズム解析

### `_DISPATCH_HISTORY` 飽和仮説

iter1 実装:
```python
_DISPATCH_HISTORY: dict[tuple[int, int], int] = {}  # (src_id, est_dst_id) -> last step

# missions/reinforcement.py で skip:
if step - last < PING_PONG_PAIR_COOLDOWN_TURNS (= 3):
    continue
```

劣勢時の振る舞い:

1. v9 が惑星を失う → **使える src が減る** (例: 16 個 → 5 個)
2. 残った src からは選択肢の dst (= 隣接惑星) も減る → 同じ src→dst pair を毎ターン発射する必要が出る
3. しかし `PING_PONG_PAIR_COOLDOWN_TURNS=3` で同 pair は 3 ターン禁止
4. **3 ターン待つ間に v4 がさらに惑星を奪う** → 残った src がさらに減る → 同じ pair しか選べなくなる → 永久 cooldown 状態
5. reinforcement.py の `for src in world.my_planets` ループで **全 src が cooldown ヒット** → mission 不発 → 何も送らない → 防衛不能 → 雪崩崩壊

### 補強的観察 (v9 勝ち試合 seed 2003)

- t=150 で v9=17 / v4=7 → **v9 が惑星過多**: src の選択肢が多く、cooldown ヒットしても他の src→dst を選べる
- launches 800 ≫ v4 476 → cooldown は事実上ほぼ効いていない (惑星数優勢時は alternative pair が常にある)
- ping-pong 件数 51 (v9) > 28 (v4): **勝ち試合では v9 自身が ping-pong を多数発生させている** = cooldown はほとんど機能せず、それでも勝つ
- 結論: **v9 の anti-ping-pong は「劣勢時にだけ強く効く」非対称な負担装置** になっている

## iter2 への具体的設計指針

### 必須修正 (cooldown を「劣勢時に弱める」)

1. **絶対 cooldown ではなく緩和条件付きに**:
   - 自惑星が threatened 状態 (`world.threatened_candidates` に入る) なら cooldown を bypass
   - 残り `world.my_planets` が初期の半分以下なら cooldown bypass
   - `min(planet数, 8)` 以下なら cooldown を 1 ターンに短縮
2. **`PING_PONG_PAIR_COOLDOWN_TURNS=3 → 1`**: そもそも 3 ターンは長すぎ。ship 数差が小さい pair の振動を 1 ターン抑止すれば十分
3. **`HARASS_TARGET_COOLDOWN_TURNS=5 → 2`**: 同様

### 副次修正 (REINFORCE_MIN_DEFICIT)

4. **`REINFORCE_MIN_DEFICIT=3 → 1` に戻す + 別条件で抑止**: deficit=1-2 でも本物の脅威の場合がある。代わりに「target の production が `REINFORCE_MIN_PRODUCTION=2` 未満なら threatened に入れない」のような production 条件で絞る方が安全

### 余剰 ship 流用 (plan.md の引き継ぎ)

5. ACCUMULATE / multi-source swarm / rear-guard のいずれかを実装し、cooldown でブロックされた送信元の ship を別 mission に振る。今回の analysis から判断: **rear-guard が最有用** (劣勢時に小型 src→dst が止まったら rear ship を前線に送る)

### 評価方針

- iter2 は **300戦評価必須**: 100戦 (±10pp) では cooldown の効果差を測定できない (memory `project_imitation_case1_phase3` 参照)
- replay 保存付きで実施 → 勝ち / 負け代表 4–6 試合をターン別に追跡し、「劣勢時の launch/turn 数」を直接メトリクスに

## Top ping-pong pairs 再確認 (Phase A 集計データから)

v9 の top pair `(6, 10)` が 100戦中 96 回発生 (= 約 1試合あたり 1 回)。これは隣接惑星間の reinforce 振動。
v4 の top pair `(16, 19)` 86 回も同様。
**所見**: top pair は試合をまたいで似た id (中心軌道惑星) で出る。配置依存ではなく **隣接かつ production 高めの pair が ping-pong 起点** になりやすい。iter2 では `(src.production, target.production) ≥ 2 かつ 距離 ≤ X` 条件で **狙い撃ち抑止** する設計が cleanest。

## 成果物 (Artifacts)

- 4 試合リプレイ: `/Users/user/project/orbit-wars.worktrees/feature-rulebase-planet-ping-pong/data/lake/selfplay/matches/replays/20260505T024331Z_1v1_seed{2000..2003}.json.gz`
- run parquet: `data/lake/selfplay/matches/index.parquet/mode=1v1/run_20260505T024331Z.parquet`
- (DVC add 未実施。永続化したい場合は別途 `dev/dvc add data/lake/selfplay/matches`)

## NEXT ACTION

1. **iter2_plan.md 起草**: 上記「劣勢時に cooldown を bypass」設計を盛り込む
2. **300戦評価**: cooldown 緩和案を 300 戦で勝率測定 (95% CI で +5pp 採否判定)
3. **launch/turn メトリクス**: compare_v4.py に「終盤 100 ターンの平均 launches」を追加し、沈黙が起きていないか自動検出

# rulebase/case9 — anti_ping_pong (iter2 replay-driven analysis)

> 作成日: 2026-05-05
> 関連: `iter2_plan.md`, `iter2_result.md`, `iter1_analysis.md`
> 分析対象: replay run `20260505T050530Z` の 4 試合 (seed 3050-3053)

## 結論 (一文)

iter2 の `LOW_PLANET_BYPASS_THRESHOLD=8` は **iter1 で見えた「雪崩崩壊」シナリオを解消したが**、その代わり「拮抗持続のまま score でじわじわ負ける僅差敗北」シナリオ C が顕在化した。**+5pp 未達の主因は崩壊ではなく score 増産速度の劣後**であり、iter3 では「余剰 ship を production 増強用 mission (capture/ACCUMULATE) に転用」が必須。

## 試合別サマリ (seed 3050-3053)

| seed | 結果 | reward | turns | v9 score | v4 score | 比 |
|---|---|---|---|---|---|---|
| 3050 | v9 圧勝 | [+1, -1] | 500 | 15,651 | 11,043 | 1.42× |
| 3051 | v9 圧勝 | [+1, -1] | 500 | 24,967 | 706 | **35.4×** |
| 3052 | v9 圧勝 | [+1, -1] | 500 | 24,890 | 1,354 | **18.4×** |
| 3053 | v4 僅差勝 | [-1, +1] | 500 | 8,504 | 8,748 | 0.97× |

→ **3勝1敗 (75%)**。compare_v4 200戦 (49.5%) より格段に良い数字 = seed range の差 (3000–3199 は seed bias、3050–3053 は v9 寄り)。

## 3 つの戦略パターン (replay 別の動的分析)

### パターン A: 序盤優勢 → 雪だるま (seed 3050, 3052)

| t | v9 惑星 | v4 惑星 | v9 launches | 観察 |
|---|---|---|---|---|
| 50 | 12 | 12 | 0–2 | 互角 |
| 100 | 15 | 12 | 5 | **わずかに先行** |
| 150 | 23 | **5** | 4 | **領土確立** (v4 を 5 惑星まで縮小) |
| 200–499 | 25 | 3 | 0–7 (まばら) | 後はほぼ launch 不要、ship 自然増 |

**特徴**: t=100–150 で領土を取り切ると、以降 cooldown は機能停止 (惑星過多)、**launch ほぼ不要で score 雪だるま**。iter1 でも同パターンの勝ち試合あり。

### パターン B: 雪崩崩壊 (iter1 seed 2000、iter2 では未観測)

iter1 で多発したが iter2 では **観測されず**。`LOW_PLANET_BYPASS_THRESHOLD=8` 機構が機能している証拠。t=143 以降 0/20 turn で 12 turn が無 launch だった iter1 のような沈黙はなくなった。

### パターン C: 拮抗持続 → 僅差負け (seed 3053) ← **iter2 で新たに顕在化した負けパターン**

| t | v9 惑星 | v4 惑星 | v9 launches | v4 launches | v9 ship | v4 ship |
|---|---|---|---|---|---|---|
| 50 | 9 | 9 | 2 | 2 | 185 | 185 |
| 100 | 13 | 16 | 3 | 4 | 206 | 400 (= 倍差) |
| 150 | 16 | 20 | 2 | 2 | 1266 | 1332 |
| 200 | 17 | 19 | 0 | 0 | 2847 | 3126 |
| 300 | 19 | 17 | 3 | 1 | 4830 | 5099 |
| 400 | 18 | 18 | 0 | 1 | 6364 | 6563 |
| 499 | 17 | 19 | 0 | 0 | **8504** | **8748** |

**特徴**: 終始 v9 と v4 が惑星数で 2-4 個差で拮抗。t=200 から大規模 launch なし (v9 0-3/turn、v4 0-2/turn)。500 turn 完走 → score 比較で **わずか 244 ship 差で v4 勝ち**。

**注目**: t=400-450 のシップ増分 v9 = +486 vs v4 = **+961 (約 2 倍)**。launch 量に大差はないのに、**v4 の方が production を効率よく ship に変換**できている。

## メカニズム解析

### iter2 が解消したもの

- 雪崩崩壊シナリオ (惑星 5 個まで縮小 → launches 完全停止 → 加速度的崩壊) は完全消失
- Seat bias 16pp → 7pp に縮小 (replay は seat=0 のみだが、200戦集計で確認済み)

### iter2 が新たに浮上させた問題

- **拮抗時の ship 増産速度**: v9 は cooldown と `REINFORCE_MIN_DEFICIT=1` のおかげで小規模 launch を一定数撃つが、それらは **多くが reinforce / harass であり「自惑星補強」「敵惑星 1ターン奪取」に消費される。production 増強につながる capture (中立惑星奪取) が後半に伸びない**
- 比較: パターン A の v9 はそもそも launch をあまり撃たず、惑星過多で **production が自然に伸びる**。パターン C の v9 は launch を撃つけれど **production 増強につながらない種類の launch ばかり** = 余剰 ship の用途ミスマッチ

### 仮説 H4 (新規)

**「拮抗時には reinforce/harass を抑制し、capture (中立惑星奪取) または ACCUMULATE (大型蓄積) に launch budget を回す」と production が伸びる**。  
case7 が ACCUMULATE で実証した「fleet_speed knee までの大型輸送」は、まさに「余剰 ship を production 増強につなげる」mission。**iter3 で case7 から ACCUMULATE を port するべき。**

## seed 依存の分散について

- compare_v4 (seed 3000–3199) で 49.5% (棄却)
- analysis 用 replay (seed 3050–3053) で 75% (4戦)

→ **同じ案件で seed range の小さな違いで結果が大きく変動**。これは:
- 試合の決着が score 比較に持ち込まれるケースが多く (500 turn 完走率 ~80%)
- score 比較は「中盤の領土確立」がそのまま雪だるま式に拡大するため **試合間分散が極端に大きい**
- iter3 以降は **300戦以上** にして CI を狭めることが必要 (memory `project_imitation_case1_phase3` が示唆する 300戦下限)

## iter3 への具体的設計指針 (優先順)

### 必須 (既知 NEXT ACTION の更新)

1. **case7 から ACCUMULATE port** (新):
   - `bot/pipeline/rulebase/case7/baseline/missions/stay.py` の ACCUMULATE 関連 + `core/config.py` の `ACCUMULATE_*` 定数を case9 へコピー
   - `strategy.py` の `SINGLE_SOURCE_MISSION_KINDS` に `accumulate_fire` を追加
   - **目的**: 拮抗持続シナリオで余剰 ship を遠距離 capture に転用 → production 増強
2. **`LOW_PLANET_BYPASS_THRESHOLD=8 → 10` に緩和** (既存):
   - 16-19 惑星帯で発生する僅差負けを早期に bypass で救う
3. **rust simulator** (既存): rustc 導入されれば 200戦 → 数分。300戦評価が現実的に

### 副次 (棄却される可能性高いが ablation で測りたい)

4. capture mission への priority boost: `score_attack` で中立惑星 production が高ければスコア+10%
5. `REINFORCE_MAX_TRAVEL_TURNS=22 → 18` で reinforce の遠距離化を抑止し、capture にリソースを回す

### 評価方針 (改善)

- **300戦** (memory `project_case2_ablation` 等の seed variance を踏まえ)
- 評価指標を 2 つに: (a) 勝率、(b) **terminal score 差** (引き分け・打ち切り試合の score 比較)
- 中盤メトリクス: `t=200, 300, 400` 時点の **ship 増分 / 惑星数比** を報告 (パターン A/C の自動分類用)

## 成果物 (Artifacts)

- 4 試合リプレイ: `data/lake/selfplay/matches/replays/20260505T050530Z_1v1_seed{3050..3053}.json.gz`
- run parquet: `data/lake/selfplay/matches/index.parquet/mode=1v1/run_20260505T050530Z.parquet`
- (DVC 未 add)

## NEXT ACTION

1. iter3_plan.md を起草: ACCUMULATE port + bypass 10 + 300戦評価
2. cron loop 次回 fire でこの NEXT ACTION を読み iter3 を実行

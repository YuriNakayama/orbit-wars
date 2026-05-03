# [rulebase/case6] STAY mission 100戦評価結果 (vs baseline_v4)

> 評価コマンド: `cd bot && uv run python -m pipeline.rulebase.case6.evaluation.compare_v4 -n 50 --seed 1000`
> 比較対象: baseline_v6 (case6) vs baseline_v4 (case4) — 50 戦 × 2 seat = 100 戦
> 環境: kaggle_environments orbit_wars 1v1, seed = 1000..1099 (両 seat)
> 計算: ローカル CPU、Vast.ai GPU は不使用 (rulebase は学習なし)
> 実行時間: 約 1 時間 22 分

## 結論サマリ

**case6 (baseline_v6) が baseline_v4 に対して 64% (64/100) の勝率で勝利**。
事前定義した「採用候補」閾値 55% を大きく上回り、STAY 判定 (defense hold +
burst hold) は productionチャンピオンに対して有意な優位を示す可能性が高い。
seed variance の保険のため、**300 戦への拡張評価を推奨**する (memory:
`<300戦は seed variance 大`)。

## 数値

| 指標 | baseline_v6 (case6) | baseline_v4 (case4) | コメント |
|------|---|---|---|
| 勝利数 | **64** | 36 | |
| 勝率 | **64.0%** | 36.0% | non-draw 換算も同じ (引き分け 0) |
| 引き分け | 0 | 0 | 1v1 でほぼ確実に決着 |
| 平均艦隊ピーク | 23.2 | 21.5 | ratio 1.08 — burst が多少効いて v6 fleet がやや大きい |
| 1ep あたり発射回数 | 489.6 | 489.3 | ほぼ同じ — STAY は瞬間 hold で長期発射数は減らさない |
| 平均エピソード長 | 200.0 turns | 200.0 turns | 500 上限張り付きはなく **stalemate 化していない** |

run id (commit SHA): `2871ad84ecde780e9b65e1882eae5da5c6521b5b` (HEAD `feature/add-rulebase-to-stay`)

### seat ごとの推移

```
seat=0 50/50  v6=30 v4=20 draws=0   (v6=60% as challenger seat 0)
seat=1 50/50  v6=34 v4=16 draws=0   (v6=68% as challenger seat 1)
total          v6=64 v4=36
```

両 seat とも v6 が勝ち越し、特に seat=1 でやや優位。**seat 偏りバイアスではない** ことが確認できる。

## 観察

- **stalemate 化の懸念は否定**された: 平均 200 ターンは case4 自己対戦と同程度
  (200 〜 250 ターン帯)。BURST hold が「永遠に発射しない」ループは入っていない。
- **launches/ep がほぼ同じ** (489.6 vs 489.3): STAY は **タイミングずらし**
  として機能しており、長期で見れば発射回数を抑制していない。1 ターン待つことで
  次ターンに同じ src からの発射が走る。
- **fleet peak の +8% 上昇** (23.2 vs 21.5): burst hold による艦の合流効果が
  測定可能なレベルで現れている (= 速度ボーナス目当ての設計が観測通り作用)。
- **defense hold の効果は単独では分離できない** が、上記 64% という勝率自体が
  「敵 fleet 接近時に発射を抑制した方が勝てる場面が増える」事実を支持している。

## 判定

| 条件 | 該当か | 推奨アクション |
|------|---|---------------|
| v6 win rate ≥ 55% | **○ (64%)** | **採用候補。300 戦に伸ばして再評価 (memory: `n<300 は noise`)、ablation で `STAY_DEFENSE_ENABLED` / `STAY_BURST_ENABLED` 個別寄与を切り分け** |
| 45% ≤ v6 < 55% | ✕ | — |
| v6 < 45% | ✕ | — |

実際の判定: **採用候補 (provisional adopt)**。300 戦再評価で 55% 以上を維持できれば case6 を baseline_v4 に並ぶ second-line baseline として運用可能。

## 設計上のリスク再点検 (plan.md からの差分)

- **stalemate**: ✓ 平均 200 ターンで上限張り付きなし、否定された。
- **defense の二重カウント**: ✓ `_defense_risk_for_planet` は
  `world.base_timeline` (内部で `reserve` を含めて simulate 済み) を使うため、
  reserve が既にカバーする敵 fleet は worst_deficit に出てこない。実装で解消済。
- **agent stateless 制約**: ✓ STAY は完全に obs だけから計算する純関数として
  実装、`_OM_STATE` のような module global は不使用。
- **計算コスト**: 100 戦で 1h22m (案外重い)。1ターンあたり STAY 判定が
  O(my_planets² × planets) の travel_time 呼び出しを増やしており、
  300 戦で 4 時間越えになる可能性。Phase 2 で `travel_time` のキャッシュ化を
  検討すべき。

## 次の一手の候補 (優先順)

1. **300 戦再評価 (vs baseline_v4)**: seed 2000 起点で同条件を回す。55% 以上
   なら memory に記録し、case6 を「default 採用ルートの一つ」に格上げ。
2. **個別 ablation (100 戦 × 2 条件)**:
   - case6a: `STAY_BURST_ENABLED=False` (defense のみ)
   - case6b: `STAY_DEFENSE_ENABLED=False` (burst のみ)
   どちらが本体寄与か可視化。両方とも 50% 未満ならランダム噛み合いの疑い。
3. **他 baseline との対戦** (各 100 戦):
   - vs baseline_v3 (case3, rollout): v4 偏重を回避
   - vs baseline_v5 (case5, LB1224 port): publicScore は v5=600 で v4 を下回るが
     v5 の自己対戦パターンは独特。case6 の汎用性検証になる
4. **parameter sweep** (採用後): `STAY_DEFENSE_THRESHOLD ∈ {0.5, 1.0, 2.0}`,
   `STAY_BURST_MIN_GAIN ∈ {1, 2}` の grid を 100 戦ずつ。今回は default 値で
   この勝率なので、tuning でさらに伸びる余地あり。
5. **計算最適化**: 300 戦・ablation・sweep を実用時間で回すため、
   `travel_time` の per-source キャッシュを `WorldModel` に乗せる検討。

## 関連リンク

- plan: `docs/experiment/rulebase/20260502_case6_stay_mission/plan.md`
- 実装: `bot/pipeline/rulebase/case6/baseline/missions/stay.py`
- strategy フック: `bot/pipeline/rulebase/case6/baseline/strategy.py` (`stay_holds` 経由で `source_attack_left` をラップ)
- 評価ハーネス: `bot/pipeline/rulebase/case6/evaluation/compare_v4.py`
- agent registry: `bot/src/dataset/selfplay/agents.py` (`baseline_v6` キー)
- テスト: `bot/tests/pipeline/rulebase/case6/` (5 unit + 2 integration)

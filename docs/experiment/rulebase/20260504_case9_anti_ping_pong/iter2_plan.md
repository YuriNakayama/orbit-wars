# rulebase/case9 — anti_ping_pong (iter2 plan)

> 作成日: 2026-05-05
> 関連: `iter1_plan.md`, `iter1_result.md`, `iter1_analysis.md`
> スコープ: cooldown bypass + 緩和 + agent 速度最適化 (rust simulator は次回)

## 仮説 (Hypothesis)

iter1 の `_DISPATCH_HISTORY` 飽和現象を、**劣勢時に cooldown を動的に bypass する** 機構と **cooldown 値の短縮** で解消すれば、vs baseline_v4 勝率が iter1 の 46.0% → 50% 帯まで戻り、**+5pp (= ≥55%) の閾値到達** を狙える。

理由: `iter1_analysis.md` の seed 2000 解析で「t=143 以降 v9 launches がほぼ 0、20 turn 中 12 turn が無 launch」を観測。`world.my_planets` 数が初期の半分以下 (= 8 個以下) になる劣勢局面では cooldown を bypass し reinforce を再開させる。

## スコープ (Scope)

**変更ファイル**:
- `bot/pipeline/rulebase/case9/baseline/core/config.py`:
  - `PING_PONG_PAIR_COOLDOWN_TURNS: 3 → 1`
  - `HARASS_TARGET_COOLDOWN_TURNS: 5 → 2`
  - `REINFORCE_MIN_DEFICIT: 3 → 1` (analysis 提案: 別条件で抑止する方が安全)
  - **新規**: `LOW_PLANET_BYPASS_THRESHOLD: int = 8` (これ以下なら cooldown 全 bypass)
- `bot/pipeline/rulebase/case9/baseline/missions/reinforcement.py`:
  - cooldown ヒット箇所で `if len(world.my_planets) <= LOW_PLANET_BYPASS_THRESHOLD: pass` を追加
- `bot/pipeline/rulebase/case9/baseline/missions/harass.py`:
  - 同様の bypass を harass cooldown 部分に追加
- `bot/pipeline/rulebase/case9/baseline/core/world_model.py`:
  - `REINFORCE_MIN_DEFICIT` 比較箇所をそのまま (定数値で挙動が変わる)

**変更しないファイル** (今回): `agent.py` の `_DISPATCH_HISTORY` 構造はそのまま。速度最適化は **iter3** で別 commit に分ける (本周回は性能差の切り分けを優先)。

## 実装ステップ (Implementation outline)

1. config.py の 4 定数を更新
2. reinforcement.py の cooldown 判定に `LOW_PLANET_BYPASS_THRESHOLD` ガードを追加
3. harass.py の cooldown 判定に同様のガードを追加
4. `dev/lint` 単体で case9 のみチェック
5. `pytest tests/pipeline/rulebase/case9 -x` (snapshot test 含む 79 件)
6. **200 戦**: `compare_v4.py -n 100 -p 4 --seed 3000` (各 seat 100戦)
7. result が +5pp 達成 → 採択コミット。未達 → 棄却して iter3 plan へ (rust + 速度最適化 + 余剰 ship 流用)

## 検証方法 (Validation method)

- ローカル: `dev/test-bot` (format → lint → type → pytest)
- 評価:
  - 対戦相手: **baseline_v4** (case4 production)
  - エピソード: **200戦** (各 seat 100戦)
  - 主要メトリクス: **vs v4 勝率**、副次: ping-pong 件数の保持確認 (劣化していないか)
  - しきい値: **≥55% で採択** (+5pp)、50–55% で iter3 cooldown tuning 続行、<50% で設計見直し
- リモート: 不要 (rule-based)

## 想定リスク

- **bypass が広すぎ**: `LOW_PLANET_BYPASS_THRESHOLD=8` が厳しすぎ / 緩すぎる可能性。ablation で `6` / `10` も試す候補
- **cooldown 短縮で ping-pong 復活**: iter1 の 11.5% 削減効果が消える可能性 → 件数を併測

## 引き継ぎ (NEXT for iter3)

- rustc + maturin インストール → rust backend で大幅高速化 (transparent 経路 1.5–2× / registry 経由 100×+)
- 余剰 ship 流用 (case7 から ACCUMULATE port、または rear-guard reserve 強化)
- agent 速度最適化 (build_world / plan_shot / safety check の cache 化)

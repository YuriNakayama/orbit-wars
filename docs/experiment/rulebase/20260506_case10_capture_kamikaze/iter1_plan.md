# rulebase/case10 — capture_kamikaze (iter1 plan)

> 作成日: 2026-05-06
> 関連: case9 anti-ping-pong 9 iter 全棄却 (memory `project_case9_anti_ping_pong_2026_05_06`)、heuristic 系探索 53% 飽和 (memory `project_heuristic_search_saturation`)

## 仮説 (Hypothesis)

case4 base に対し **capture mission 強化 + sniper/kamikaze 多用** を同時に効かせれば、
case4 と異なる状態空間 (production 増産速度を意識的に高める設計) に到達し、
heuristic 系の 53% 壁を突破できる可能性がある。

case9 で確認した「拮抗時に v9 が production 増産速度で v4 に劣る」問題に対し、
- **Capture 側**: 高 production neutral を優先取得 → 自軍 production を底上げ
- **Sniper/Kamikaze 側**: 敵 production の早期奪取 → 相手 production を抑制

の 2 軸で攻める。

## スコープ (Scope)

**変更ファイル**: `bot/pipeline/rulebase/case10/baseline/core/config.py` のみ
- `STATIC_NEUTRAL_VALUE_MULT: 1.4 → 1.6` (capture 強化)
- `EARLY_NEUTRAL_VALUE_MULT: 1.2 → 1.4` (序盤 capture 強化)
- `SNIPE_VALUE_MULT: 1.12 → 1.30` (snipe 多用)
- `HARASS_MIN_SRC_RESERVE: 10 → 6` (kamikaze 多用)
- `HARASS_PRODUCTION_STEAL_TURNS: 5 → 8` (harass 価値上昇)

**変更しないファイル**: それ以外すべて (case10 = case4 base + 上記 5 定数)。

## 実装ステップ

1. case4 を case10 にコピー (完了)
2. AGENT_REGISTRY に baseline_v10 登録 (完了)
3. config.py 5 定数変更 (完了)
4. ruff/mypy 確認 + snapshot test (action 系列が変わるので **snapshot 更新必須**)
5. 200戦評価: `compare_v4.py -n 100 -p 4 --seed 12000`

## 検証方法

- 対戦相手: baseline_v4
- エピソード: 200戦 (seed 12000-12199)
- 主要メトリクス: vs v4 勝率
- しきい値:
  - **≥55% (200戦)**: 採択候補、iter2 で 300戦 confirm
  - **51-55%**: 弱採択、iter2 で 300戦 confirm
  - **<51%**: 棄却、iter2 で別軸 (例えば 5 定数のうち効きが強そうな 1 つに絞る ablation)

## 想定リスク

- **case4 default 値は tuning 結果**: iter1 の改造はそれに逆行する設定変更を含み、勝率低下の可能性
- **5 定数同時変更で因果関係が分からない**: 200戦結果が予想外なら iter2 で各 1 定数の ablation が必要
- **snipe / kamikaze 過多**: src ship が枯渇して通常の capture/reinforce が回らない

## 引き継ぎ (NEXT for iter2)

- iter1 採択 → 300戦 confirm
- iter1 棄却 → 5 定数を 1 つずつ ablation し効きを切り分け
- いずれにせよ memory `project_heuristic_search_saturation` の 11 連敗パターンと
  比較し、似た失敗なら早期に loop 終了判断

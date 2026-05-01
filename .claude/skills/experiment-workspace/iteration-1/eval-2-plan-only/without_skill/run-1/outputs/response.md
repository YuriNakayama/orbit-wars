了解しました。`rulebase/case2` をベースに harass policy を加えて comet 周辺で挑発する新しい実験の plan.md を作成します。実装と Vast.ai 起動はまだ行わず、レビュー用の plan.md のみに絞ります。

## 実験ドキュメントの配置

`.claude/rules/docs.md` の命名規則に従い、以下のパスに plan.md を新規作成します。

- パス: `docs/experiment/rulebase/20260501_case2_harass_comet/plan.md`
  - 第一階層: `rulebase`(対象 agent family)
  - ディレクトリ名: `{yyyymmdd}_case{N}_{topic}` = `20260501_case2_harass_comet`(今日 2026-05-01、case2、topic は harass_comet)
  - ファイル名: 新規提案なので `plan.md`

既存の `rulebase/case2/` 実装はそのままで、改修対象として参照する形にします(plan.md の段階では `backend/pipeline/rulebase/case2/` には触りません)。

## plan.md ドラフト

以下の内容で書き起こす想定です。レビューで指摘いただければ反映します。

```markdown
# rulebase/case2 — harass policy で comet 周辺の挑発戦術を試す

- 開始日: 2026-05-01
- 対象: `backend/pipeline/rulebase/case2/`
- ステータス: plan(未実装・Vast.ai 未起動)

## 仮説

case2 baseline は comet 出現ターン (50 / 150 / 250 / 350 / 450) 直後の resource shift で
受動的になりがちで、相手 home / front 拠点に対する pressure が抜ける。
comet 周辺に少数艦の harass fleet を恒常的に置くことで、相手の defense を comet 側に
釣り出し、本隊の expansion 効率を上げられるはず。

## 評価指標

`.claude/rules/backend/pipeline.md` の方針に従い、Kaggle publicScore は使わず
ローカル対戦のみで評価する。

- 主指標: case2 baseline (harass OFF) との 1v1 勝率(300 戦、seed 固定範囲)
- 補助指標:
  - 4-player FFA での着順分布
  - comet 出現ターン直後の自軍総 ships 推移
  - harass fleet 投入数 / ターン平均

過去メモリの `project_case2_ablation` より、100 戦は seed variance が大きく信頼できないため
**最低 300 戦**で判断する。

## 施策

`baseline/` の policy 層に harass mission を追加する(挙動はフラグで ON/OFF 切替可能)。

1. **comet 観測モジュール**: 次の comet 出現ターン / 中心座標を予測
2. **harass target 選定**: 自軍 home から最遠の comet 1 つ、または相手 home に最も近い comet を選ぶ
3. **harass fleet サイズ**: 出撃 planet の `ships * 0.1` を上限にした少数艦
4. **発射タイミング**: 出現ターン -10 から +20 のウィンドウのみ
5. **撤退条件**: 自軍 ships が threshold 以下、または harass target が自軍占領済み

config 例:
- `HARASS_ENABLED: bool = True`
- `HARASS_FLEET_FRACTION: float = 0.1`
- `HARASS_WINDOW_PRE: int = 10`
- `HARASS_WINDOW_POST: int = 20`

## 実装範囲(レビュー後に着手)

- `backend/pipeline/rulebase/case2/baseline/missions/harass.py` を新規追加(相対 import)
- `baseline/agent.py` から harass mission を呼び出すフックを追加
- `baseline/core/config.py` に上記 config を追加
- `backend/tests/pipeline/rulebase/case2/test_harass.py` を新規追加
- `evaluation/compare_harass_on_off.py` を追加し 300 戦の比較を取れるようにする
- `AGENT_REGISTRY` への登録は不要(既存 `baseline_v2` をフラグで切り替える方針)

## Vast.ai での扱い

rulebase/case2 は学習を伴わないため Vast.ai は使わない。ローカル selfplay のみで評価する。

## ロールアウト判断

300 戦勝率が baseline 比 +3pp 以上、かつ FFA 着順分布が悪化していなければ採用。
case2_ablation のメモリで Harass+HALF_STEP は 300 戦で +3.7pp(非有意)だったので、
今回は **有意性検定(片側 binom test, alpha=0.05)を必ず通す**。

## オープン課題 / レビュー観点

- harass target を「相手 home に最も近い comet」と「自軍から最遠の comet」のどちらに固定するか
- harass fleet が逆に取られて comet を相手に渡してしまうケースのリスク評価方法
- HALF_STEP との重ね掛けを今回の実験範囲に含めるか、別 case にするか
```

## 次のアクション(レビュー待ち)

- 上記 plan.md ドラフトをこのまま `docs/experiment/rulebase/20260501_case2_harass_comet/plan.md` に書き出してよいか確認してください。
- レビューで反映したい修正点(仮説の言い換え、評価戦数、harass target の選定基準、フラグ名など)があれば指示お願いします。
- レビュー OK 後に進める順序の想定:
  1. plan.md コミット
  2. 実装(`backend/pipeline/rulebase/case2/baseline/missions/harass.py` ほか)
  3. `dev/test-backend` でローカル検証
  4. ローカル 300 戦 selfplay → `result.md`(Vast.ai は使用しない)

実装着手とローカル selfplay 起動はレビュー OK が出てから行います。

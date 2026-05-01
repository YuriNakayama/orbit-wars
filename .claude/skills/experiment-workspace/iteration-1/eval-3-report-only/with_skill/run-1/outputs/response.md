# 主セッションの応答シミュレーション (with_skill / report-only)

了解しました。`run_20260501_abc123` の結果を imitation/case1 の **iter4** として `docs/experiment/imitation/` 配下にまとめる作業ですね。学習自体はもう終わっているので、experiment skill の Phase 1 (仮説確認) / Phase 2 (git 事前チェック) はスキップして、`experimenter` サブエージェントを **eval / report-only モード** で起動します。

## 進め方

- 新規学習は走らせません (Vast 課金なし)。`dev/vast train` / `dev/vast promote` / Kaggle submission はいずれも非対象です。
- `experimenter` 側で以下のみ実行します:
  1. `dev/vast pull run_20260501_abc123` でアーティファクトを `data/output/models/imitation/case1/runs/run_20260501_abc123/` に取得。
  2. メトリクス JSON / best.pt と launch 時のコミット SHA を確認。
  3. case1 既存の評価ハーネスで LB1235 baseline などの標準対戦相手に対し ≥300 episodes でローカル勝率を測定 (n<300 はノイズという既知メモあり)。
  4. 既存の case1 ディレクトリ (おそらく `docs/experiment/imitation/20260422_case1_phase3/`) 内に **`iter4_result.md`** を追記。既存 `result.md` が iterN_ プレフィックス無しなら、命名規則に従って iter1_result.md にリネームしてから iter4 を足します。
- 評価指標は **ローカル対戦成績のみ**。Kaggle publicScore / skill rating は判定根拠に使いません (これは過去の知見と一致)。
- 既存 case1 ディレクトリのどれが iter4 の所属先か、メトリクスの中身を見て判断できなければ Q&A で確認してから書き始めます。

GPU 課金は発生しないので、このまま起動します。完了したら `iter4_result.md` のパスと、headline (勝率 / 採否) をこちらに返してもらいます。

## 次にやること

- `experimenter` をバックグラウンドで起動 → 完了後に結果サマリを日本語で報告。
- 採用判定が positive だった場合のみ、別承認ゲートとして `dev/vast promote run_20260501_abc123` を提案します (canonical weights を上書きするので、その時点で改めて確認します)。

エージェントを起動します。

---

(内部メモ: 以下の Phase 3 ハンドオフプロンプトを `Task(subagent_type="experimenter", prompt=...)` に渡して spawn する。本文は `agent_handoff_prompt.txt` を参照。)

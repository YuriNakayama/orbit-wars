了解しました。imitation/case1 の dropout を 0.2 → 0.3 に上げて再学習し、LB1235 baseline 相手に勝率改善を狙う実験ですね。手順を整理してから着手します。

## 実施プラン

`.claude/rules/docs.md` の命名規則に従って、以下の構成で進めます。

1. **計画文書の作成**
   - `docs/experiment/imitation/20260501_case1_dropout/plan.md` を作成
   - ハイポセシス: dropout 0.3 にすることで過学習を抑え、LB1235 戦の汎化性能を上げる
   - 評価指標: 局所マッチでの win-rate (300戦推奨、Phase 3 の知見より n<300 は信頼不可)
   - 比較対象: 現行 weights.pt (dropout=0.2) vs 候補 best.pt (dropout=0.3)

2. **コード変更**
   - 該当の dropout 定義箇所を特定 (`backend/pipeline/imitation/case1/` 配下の training/モデル定義)
   - 0.2 → 0.3 に変更し、変更点をコミット

3. **ローカル検証**
   - `dev/test-backend` で format / lint / type / pytest を一括実行
   - imitation/case1 の既存ユニットテストが緑であることを確認

4. **Vast.ai 学習**
   - `git push origin feature/reinforcement-learning-conversion`
   - `dev/vast train <commit-sha> --stage train_imitation_case1`
   - 完了後 `dev/vast pull <run_id>` で `data/output/models/imitation/case1/runs/<run_id>/best.pt` を取得

5. **評価と結果まとめ**
   - LB1235 を含む比較スクリプト (`pipeline/imitation/case1/evaluation/compare_*.py` 系) を 300戦回す
   - `docs/experiment/imitation/20260501_case1_dropout/result.md` に勝率・サンプル数・notes をまとめる
   - candidate が現行を上回れば `dev/vast promote <run_id>` を提案 (実行はユーザー承認後)

## 着手前にいくつか確認させてください

- **dropout 変更箇所**: imitation/case1 のモデル定義は単一の dropout 値ですか? それともレイヤー別に複数ある場合は全レイヤーを 0.3 に揃えますか? 入力側と中間層を分ける選択肢もあります。
- **評価サンプル数**: Phase 3 の MEMORY.md メモに「n<300 評価は信頼不可」とあるので、最低 300戦で良いでしょうか? コスト次第で 500戦も検討できます。
- **学習エポック / seed**: 既存の学習設定 (epoch, optimizer, seed) はそのままでよいですか? dropout 単独効果を測るためには他はベースラインに揃えるのが筋だと考えています。
- **Vast.ai インスタンス選定**: コスト次第ですが既定の GPU クラスでよいですか? `dev/vast cost-report --month 2026-05` で今月の消化状況を確認してから着手します。

確認が取れたら、まず `docs/experiment/imitation/20260501_case1_dropout/plan.md` を書き、コード変更 → `dev/test-backend` → Vast.ai 起動の順で進めます。

なお、Kaggle への実提出は今回のスコープ外と理解しています (LB1235 はあくまで局所評価の相手モデルとして扱う前提)。もし最終的に提出まで進める場合は、`.claude/rules/command.md` の Kaggle Submission Policy に従い、提出直前に改めて承認を取りに来ます。

---
paths:
  - "pipeline/**"
---

# Pipeline Rules

## Directory Structure

戦略別の学習・評価パイプラインは `pipeline/caseN/` 配下に独立したディレクトリとして管理する:

```
pipeline/
  case1/
    eda/              観測ログ・リプレイの探索的分析
    training/         学習スクリプト（RL / 模倣学習 / 進化戦略）
    evaluation/       自己対戦・ベンチマーク評価
    configs/          ハイパーパラメータ・対戦相手セット
    output/           学習済みモデル・評価結果（大きいものは gitignore）
    README.md         ケースの戦略・結果・考察
  case2/
    ...
```

## Workflow

各ケースは以下の流れで進める:

1. **EDA**: 既存リプレイ・公開ボットの挙動を分析し、改善ポイントを洗い出す
2. **設計**: ポリシー構造（ルール / 学習 / ハイブリッド）と入力特徴量を決定
3. **実装**: `src/policies/` にポリシーを追加、パイプラインから呼び出す
4. **自己対戦**: `kaggle_environments` で多数エピソードを実行し、勝率・レーティングを計測
5. **評価・提出**: 対戦相手プール（旧提出・公開ボット）で評価し、有望なら Kaggle に提出

## Simulation Conventions

- 乱数シードは config に明記し、再現可能にする
- 長時間の自己対戦は `multiprocessing` で並列化、CPU バウンドを前提に設計
- 1エピソード＝500ターン上限。タイムアウト（`actTimeout=1s`）違反を計測しログ化
- リプレイは JSON で保存し、`data/replays/{case}/{timestamp}.json` に格納
- 中間結果（特徴量バッチ、勝率行列）はファイルキャッシュに保存して再利用

## Evaluation Criteria

Kaggleのスキルレーティングは **勝敗のみ** で更新されるため、指標は以下を優先する:

- **勝率 (Win Rate)**: 対戦相手プールに対する勝率
- **平均最終スコア**: `自軍惑星艦数 + 飛行中艦数`（収束の参考値）
- **レーティング更新予測**: 期待勝率との乖離から μ 更新量を推定
- **タイムアウト率**: 1ターン 1秒を超過したエピソード割合（低いほど良い）

## Coding Conventions

- 自己対戦スクリプトは CLI 化（`typer`）してハイパーパラメータを引数化
- 中間結果を壊さないよう、出力先ディレクトリを実行ごとに分ける（タイムスタンプ付与）
- 大規模観測データはチャンク処理（polars の lazy frame など）でメモリ効率を確保
- マジックナンバー（`boardSize=100.0`, `sunRadius=10.0`, `shipSpeed=6.0` 等）を定数化
- 探索コードでも `print` は避け、`rich` / `logging` を使う

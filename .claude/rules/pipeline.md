---
paths:
  - "pipeline/**"
---

# Pipeline Rules

## Directory Structure

各分析ケースは `pipeline/caseN/` 配下に独立したディレクトリとして管理する:

```
pipeline/
  case1/
    eda/              探索的データ分析（データ概要、統計、可視化）
    data/             入力データパッケージ（gitignored if large）
    output/           分析結果・最終回答
    README.md         ケースの概要・質問・アプローチ
  case2/
    ...
```

## Analysis Workflow

各ケースは以下の流れで進める:

1. **EDA（探索的データ分析）**: データパッケージの構造・内容を把握
2. **計画**: 質問を分解し、推論トポロジー（逐次/分岐・統合/反復）を決定
3. **実行**: ツール（Python, SQL, パーサー）で各ステップを処理
4. **統合**: 中間結果を合成して最終回答を生成
5. **検証**: 列マッチング基準で回答の正確性を確認

## Supported Data Formats

| Format | Extension | Handling |
|--------|-----------|----------|
| CSV | `.csv` | Pandas / DuckDB |
| JSON | `.json` | `json` module / Pandas |
| SQLite | `.sqlite`, `.db` | DuckDB / sqlite3 |
| PDF | `.pdf` | PyPDF2 / pdfplumber |
| Image | `.png`, `.jpg` | Pillow / Claude Vision API |
| Word | `.docx` | python-docx |
| Excel | `.xlsx` | openpyxl / Pandas |

## Reasoning Topology Patterns

### 逐次チェーン (Sequential Chain)
ステップ間の依存関係が線形。前のステップの出力が次の入力になる。

### 分岐・統合 (Fork-Join)
複数データソースへの並列サブクエリ後に結果を統合。`asyncio.gather` での並列実行を推奨。

### 反復ループ (Iterative Loop)
中間結果を評価し、精度が基準に達するまで改善を繰り返す。無限ループ防止のため最大反復回数を設定すること。

## Evaluation Criteria

**列マッチング正確性 (Column Matching Accuracy)**:
- 予測が全ての正答列を完全かつ正確に網羅 → スコア1
- それ以外 → スコア0
- 列名は比較対象外
- 余分な列は許容される

回答生成時は以下を意識:
- 必要な列を漏れなく含める
- 列の値は正確に一致させる
- 不確実な場合は余分な列を含めてもペナルティなし

## Coding Conventions

- 分析スクリプトは再現可能に書く（ランダムシードの固定、パス指定の明確化）
- 中間結果はファイルに保存し、後続ステップで再利用可能にする
- 大規模データはチャンク処理でメモリ効率を確保
- SQL クエリは必ずパラメータ化する
- マジックナンバーを避け、定数として定義する

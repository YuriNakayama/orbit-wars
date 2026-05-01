# ディレクトリリファクタリング — リスクと依存関係

**作成日**: 2026-04-29

---

## リスク一覧

| # | リスク | 影響度 | 発生確率 | 対策 |
|---|--------|--------|----------|------|
| 1 | Strategy 分割で agent 出力 (action 系列) が変化 → snapshot 破壊・勝率低下 | High | Med | Step 3 でベースライン記録、Step 6/7/8 で snapshot test 必須 + selfplay 50 戦の統計検証。差異 5pp 超なら revert |
| 2 | params.yaml 集約後に DVC stage が新パラメータを認識せず、不要再実行が発生 | Med | Med | `dvc params diff` で事前確認、`dvc.yaml` の `params:` セクション同期 |
| 3 | evaluation 集約で metrics 数値が微妙に変化 (浮動小数点演算順) | Med | Low | Step 4 で旧コードと値一致を assert (≦ 1e-6 tolerance) |
| 4 | 1 個の大規模 PR がレビュー困難になり merge が滞る | Med | High | Step 単位で commit を分け、PR description に各 commit の意図を明記 |
| 5 | rulebase/case5 の `agent_full.py` 移動で隠れた依存が発覚 (AGENT_REGISTRY 経由以外) | Med | Low | `grep -r "agent_full" backend/` で参照箇所を網羅、テストで import 確認 |
| 6 | per-case `configs/` 削除で legacy スクリプト (dev/, infra/) が壊れる | Low | Low | `grep -r "configs/" backend/ dev/ infra/` で参照確認 |
| 7 | Strategy 分割後に新 strategy/ 配下のクラス間結合が高まり結局テスト困難 | Med | Med | `MissionSelector`, `TargetPicker`, `OrderBuilder` をデータクラス渡しの純粋関数寄りで設計 |
| 8 | case 完全独立原則と「巨大 PR」の組み合わせで diff が膨大化 | Low | High | Step 単位 commit で吸収、PR の files changed は 100+ になる前提 |
| 9 | snapshot test の非決定性 (既知問題: project_imitation_case1_phase3.md 参照) | High | Low | Step 3 で「現状 snapshot は固定可能か」を最初に検証、不可なら snapshot 戦略見直し |
| 10 | refactor 中に並行で kaggle 提出があると base が動く | Low | Low | feature branch で完結、main は触らない |
| 11 | Strategy 分割で 1 ターン実行時間が悪化 (オブジェクト生成オーバーヘッド) | Med | Low | Step 15 で性能測定、100ms threshold 超えたら最適化 |
| 12 | 既存 `pyproject.toml` の lint 例外撤廃で大量の警告噴出 → Step 13 で詰む | Med | Med | Step 7 (case1 分割) と Step 13 (例外撤廃) を同 PR 内で続けて実施、撤廃後にのみ確認 |

---

## 外部依存

### コードベース外の依存

- **DVC remote (S3)**: refactor で出力パスは不変のため影響なし
- **Kaggle 環境**: case 提出は scope 外、AGENT_REGISTRY 経由の自動採用パスのみ確認
- **Vast.ai 学習**: training scripts (case 内独立) は変更しないため影響なし
- **Kaggle environments package**: バージョン不変

### 他チームの作業

なし (single developer プロジェクト)

### 外部 API / サービス

なし

---

## 技術的負債 (今回導入される or 残存する)

### 今回 **残存** する負債

1. **rulebase/core モジュールの ~2400 行重複** — case 独立原則のため許容、`docs/plans/refactor-directory/dump-inventory.md` で残存箇所を記録
2. **imitation/training の ~1200 行重複** — 同上
3. **imitation/policy/featurizer.py 系の ~80% 重複** — 同上
4. **imitation/policy/{geometry, decoder}.py の 100% 重複 (462 行)** — case 独立のため敢えて残置
5. **case0 (休眠) の存在** — 学習資料として保持

### 今回 **追加** される負債

1. **dump/ ディレクトリの将来的な分散** — 現時点では空または case5/dump/agent_full.py のみ。手動同期ヒントとして使用するルールが明確でないと膨張のリスク
2. **params.yaml の肥大化** — case 増加に伴い数百行になる可能性。Hydra 等への将来移行余地は残す
3. **case 規約と現実の乖離** — eda/, notebook/ の整理基準が緩い

### 解消される負債

1. evaluation スクリプトの 1300 行重複
2. rulebase/case1, case4, case5 の lint 例外
3. 巨大関数 `plan_moves` (702 行), `agent_full.py` (2455 行)
4. case2/3 の per-case `configs/` ディレクトリ

---

## 未決事項 (Open Items)

| # | 項目 | 決定が必要なタイミング | 提案 |
|---|------|---------------------|------|
| 1 | dump/ ディレクトリの将来運用 (どのタイミングで「手動同期」するか) | Step 14 完了時 | 「同期は気付いたタイミングで PR 起票」など軽量な運用ルールに留める |
| 2 | rulebase/case0 を archive に移動するか | Step 12 (README ステータス表作成時) | archive 状態タグ付与のみで物理移動はしない |
| 3 | Strategy 分割を imitation cases にも将来適用するか | この PR の scope 外 | フォロー issue として記録 |
| 4 | params.yaml が 500 行を超えた場合の対処 | 将来 | Hydra 移行 or yaml include 検討 |
| 5 | `dump-inventory.md` の管理者と更新タイミング | Step 14 | 重複コード変更時のレビュー指摘 → 担当者が更新 |
| 6 | case 番号体系の semver 化 | 将来 (この PR では現状維持) | 番号 → tag への移行 plan を別途 |

---

## 依存関係グラフ

```
[Step 1, 2 (test 追加)]
        ↓
[Step 3 (baseline 確立)]
        ↓
[Step 4 (evaluation 集約)] ─── [Step 9 (params 集約)]
        ↓                              ↓
[Step 5 (eval ラッパー化)]    [Step 10 (パス DI 化)]
        ↓                              ↓
[Step 6 (case4 strategy 分割)]
        ↓
[Step 7 (case1 strategy 分割)] ─── [Step 8 (case5 strategy 分割)]
        ↓
[Step 11 (規約明文化)]
        ↓
[Step 12 (README 整備)] ─── [Step 13 (lint 例外撤廃)]
        ↓
[Step 14 (cleanup)]
        ↓
[Step 15 (E2E 検証)]
```

並列化可能性:
- Step 1 & 2 は完全並列
- Step 4 & 9 は並列可 (異なる責務)
- Step 7 & 8 は並列可 (case 独立)
- Step 12 & 13 は並列可

---

## 緊急停止条件 (Halt Criteria)

以下のいずれかが発生したら **immediate halt** し、原因究明を優先する:

1. Step 6/7/8 で snapshot test が壊れる (Strategy 分割で出力変化)
2. Step 6/7/8 で selfplay 勝率が baseline 比 5pp 以上低下
3. Step 9 で training script が動かなくなり、`uv run --directory backend dvc repro --dry` が fail
4. Step 13 lint 例外撤廃で 50 件以上の警告噴出 (= Strategy 分割の不徹底)
5. Step 15 E2E で性能 (1 ターン時間) が 1.5 倍以上になる

# Imitation Learning Cases

DeepSets-based behavior-cloning agents. 各 case は独立提出パッケージ。

## Status table

| Case | Status | Architecture | val F1 | 自己対戦 vs baseline_v1 | 備考 |
|------|--------|--------------|--------|------------------------|------|
| case1 | active (canonical) | DeepSets BC, 11 planet feat × 6 global | 0.47 (iter9) | 5/100 (iter9) → 3/300 再評価 | Phase 2 で 0/100 → 5/100、target diversity 残課題 |
| case2 | active | il_v2 (18 planet × 11 global) + phase1 head | — | — | dual-featurizer (baseline + phase1) |
| case3 | active | il_v3 (case2 + 時系列特徴量) | — | — | featurizer_phase2 (563 行) |

## Conventions

- `case<N>/policy/` が submission code (Kaggle に同梱)
- `case<N>/training/` は `.submitignore` で除外、ローカル開発のみ
- `case<N>/evaluation/` は `src/evaluation/{metrics,vs_baseline}.py` 経由 (ロジック共通化済)
- `policy/{geometry,decoder}.py` は case 間で 100% 重複しているが case 独立原則のため許容

詳細: `docs/plans/refactor-directory/`

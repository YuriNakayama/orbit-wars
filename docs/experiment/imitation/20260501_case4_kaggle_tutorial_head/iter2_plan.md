# imitation/case4 — RunPod 初学習 & ログ収集 (iter2)

> 作成日: 2026-05-03
> 関連: [`iter1_plan.md`](iter1_plan.md) (head 設計と Vast.ai 想定の初版)
> スコープ: 既存 `case4` 実装をそのまま回し、初学習の log と 30 戦評価を取得して仮説の生存可否を判定する

## 仮説 (Hypothesis)

iter1 で設計した per-source × candidate categorical head は、`il_case4.yaml` の推奨設定 (epochs=15) で **学習ジョブが正常に完走し、val candidate top1 が non-trivial (>> 1/8 = 0.125) に到達する** ──　また 30 戦の interim 評価で `baseline_v1` 相手に **勝率 ≥ 5%** の生存兆候が出るはずである (300 戦による強い主張は別 iter の責務)。

副次的に、本 iter は **iter1 設計の "コードが回るか" の最初の実機検証** でもあり、loss 曲線・class imbalance 補正の効き・rule-based ships の overfire 比率を学習ログから読み取れる状態にすることがゴール。

## 既存コードの現状 (Step 1 から)

- 主要モジュール (`backend/pipeline/imitation/case4/`):
  - `policy/`: `agent.py` / `model.py` (Graph U-Net + per-source candidate head) / `featurizer.py` / `candidates.py` / `decoder.py` / `geometry.py` / `types.py`
  - `training/`: `preprocess.py` / `dataset.py` / `losses.py` / `train.py`
  - `evaluation/eval_vs_baseline.py`
  - `configs/il_case4.yaml`
- 過去 iter の所見: `iter1_plan.md` で head 再設計仕様を確定済み。実装は揃っているが、**学習ジョブを RunPod で回した記録は無し** (Vast.ai 前提だった iter1 の手順は本 iter で `dev/runpod` 系に置換)。
- `src/dataset/selfplay/agents.py` には `il_v4` 登録済み想定 (実装側を確認しながら回す)。

## スコープ (Scope)

- **コード変更**: なし。本 iter は既存 case4 実装の挙動確認 + ログ収集が主眼。
- **設定変更**: なし。`pipeline/imitation/case4/configs/il_case4.yaml` を default のまま使用。
- **データセット / 特徴量変更**: なし。preprocess は `il_case4.yaml` の入力 spec をそのまま利用。
- **新規追加**: 学習ログを次 iter から参照しやすくするため、`run.json` / `history.jsonl` / `summary.json` のパスを本 plan に明記し、結果は `iter2_result.md` に番号付きでまとめる。

## 実装ステップ (Implementation outline)

1. 事前確認 (ローカル):
   - `cd backend && uv run python -c "from pipeline.imitation.case4.policy.agent import agent; print(agent)"` で import が通るか
   - `dev/test-backend` が green
   - `git status` clean & 対象ブランチが remote に push 済み
2. 学習ジョブ起動 (RunPod, Secure Cloud):
   - `git push origin <branch>`
   - `dev/runpod train <SHA> --case case4 --cloud-type SECURE` (default cost-limit $1.5/run)
   - 戻り値の `run_id` をメモ
3. 進捗監視 (任意):
   - `dev/runpod ps` / `dev/runpod status <run_id>` で pod state + S3 marker を確認
   - `dev/runpod tail <run_id> --source train` で stdout を tail (epoch ごとの train/val loss が見える)
   - もしくは `dev/runpod watch <run_id>` で完了/失敗時に desktop 通知
4. 学習完了後の artifact 取得:
   - `dev/runpod pull <run_id>` (DVC 経由 → 失敗時 S3 fallback)
   - 配置先: `data/output/models/imitation/case4/runs/<run_id>/{best.pt, history.jsonl, summary.json, run.json}`
5. ローカル評価 (interim):
   - `uv run --directory backend python -m pipeline.imitation.case4.evaluation.eval_vs_baseline --episodes 30 --seed 0`
   - 30 戦の win-rate と平均ターン数を記録
6. `iter2_result.md` を本ディレクトリ直下に書き出し:
   - 学習ジョブ統計 (run_id / SHA / wall-time / cost)
   - 学習曲線サマリ (train/val final loss, val candidate top1, no-op precision など `summary.json` 抜粋)
   - 30 戦 win-rate (vs baseline_v1)
   - 失敗・気付き (例: NaN, OOM, slot 0 比率の異常, overfire)

## 検証方法 (Validation method)

- ローカル: `dev/test-backend` + `uv run --directory backend pytest tests/pipeline/imitation/case4 -x`
- (submit-shape change なし → submit dry-run はスキップ)
- リモート: `dev/runpod train <SHA> --case case4 --cloud-type SECURE`、想定 wall-time 〜1–2h (RTX 3090 相当, GraphUNet サイズ case3 同等)
- 評価:
  - 対戦相手: `baseline_v1` (rulebase/case1, 1v1)
  - エピソード数: **interim 30 戦** (生存確認用、`project_imitation_case1_phase3` の知見通り 100 戦未満は noise として注釈する。300 戦による強い主張は次 iter の責務とする)
  - 主要メトリクス: 勝率 (vs baseline_v1) を最優先。副次的に train/val loss 曲線、val candidate top1 acc、no-op precision、overfire 比率
  - 採否しきい値: **30 戦で勝率 ≥ 5%** なら "head redesign が崩壊していない" interim 合格 → 次 iter で 300 戦評価へ。0/30 なら preprocess / class imbalance / decoder の優先デバッグ対象を `iter2_result.md` に記す

## ログ収集チェックリスト

`iter2_result.md` 執筆時に以下が全て埋まることを確認:

- [ ] `run_id`, commit SHA, RunPod cloud-type, GPU 機種, wall-time, 実コスト
- [ ] `summary.json` から: best epoch, best val candidate top1, final train/val loss, no-op precision/recall
- [ ] `history.jsonl` 抜粋: epoch ごとの train_loss, val_loss, val_candidate_top1 (epochs=15 全件)
- [ ] preprocess ログから: UNUSED label 比率 (5% 超なら K=12 検討の根拠), slot 0 比率
- [ ] 30 戦 evaluation: 勝率, 引き分け率, 平均ターン数, 自分の艦船総量推移 (見える範囲で)
- [ ] 異常: NaN epoch / OOM / `il_v4` 登録漏れによる evaluation 失敗 など

## Risks / known unknowns

- **コードがまだ一度も RunPod 上で回っていない**: import error、`il_case4.yaml` の path 不整合、preprocess の output schema mismatch などが初回で出る可能性。Step 1 のローカル `dev/test-backend` を必ず先に通す。
- **K=8 candidate quota の UNUSED 落ち**: preprocess ログで >5% なら 30 戦結果が悪くても "head redesign の失敗" と即断せず K=12 等の追加 iter を立てる。
- **n=30 noise**: 30 戦で 0 勝でも "完全に駄目" 判定はしない。`project_imitation_case1_phase3` の通り 5/100 が 0/300 に落ちた前例があるので、本 iter の判断は **生存可否のみ**。
- **重みの非互換**: case3 の `weights_phase2.pt` は heads 形が違うので load 不可 → scratch 学習。
- **コスト超過**: RunPod default cost limit は $1.5/run。`dev/runpod train` 実行前に推定コスト確認プロンプトが出るので、超過時は別途承認。

## 参考 (References)

なし (本 iter は既存 plan の実装手順 + RunPod ハンドオフのみ。外部調査不要)。

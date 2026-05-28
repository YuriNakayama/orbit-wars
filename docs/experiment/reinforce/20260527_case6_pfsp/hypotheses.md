# Hypotheses — reinforce/case6 PFSP (Prioritized Fictitious Self-Play)

> 作成日: 2026-05-27
> 最終更新: 2026-05-28
> 状態: in_progress
> 最大 iteration: リスト消化まで
> 主要メトリクス: ① 学習 reward trend / last-10 mean、② win rate vs 初期 snapshot (self-play 進歩)、③ ルールベース baseline_v1 との 20 戦ローカル対戦 (方向性確認)
> 既定 episode 数: 100 ⚠️ (n<300 のため結論は方向性参考値)

## 背景・現状 (Phase 1 コード調査)

- `pipeline/reinforce/case6/training/rollout_jax.py` の opponent は **jit-trace 時に int コード化**
  (`noop`=0 / `baseline_jax_lite`=1 / `baseline_jax_full`=2) され、`jax.lax.switch` で分岐。
  scan/vmap friendly にするための制約で、**自分自身 (snapshot) を相手にする経路は未実装**。
- `train_jax.py` の opponent は `curriculum` (early→late を `switch_iter` で 1 回切替) のみ。
  pool / 優先度 sampling / self-play なし。
- snapshot は **毎 iter run_dir に保存されるが opponent には還元されていない**。
- model は Equinox pytree (`__call__(batch)`)。frozen な 2 つ目の param pytree を scan に通せば
  self-snapshot opponent は技術的に実現可能 (= H1 の主眼)。
- JAX 化済み rule agent: `baseline_jax_lite` / `baseline_jax_full`。Python registry には
  `baseline_v1`..`v9` があるが JAX 化は v1 のみ。

## 実施しない検証 / 評価 (skip list)

### 評価
- **300 対戦による評価はしない** — 学習 reward trend / last-10 mean + 100 戦 self-play + 20 戦 (vs baseline_v1) で採否。
- Kaggle publicScore は引用しない (project rule、memory `project_om_finding` / `project_case5_validation`)。
- skill rating は採否に使わない (project rule)。

### 分析
- **n<300 結果で結論を出さない** (default ON、memory `project_imitation_case1_phase3`)。
  100 戦 / 20 戦は seed variance 大のため方向性の参考値扱い。主軸は reward trend。
- replay 分析 (experiment-analysis) は **実施する** (skip 指定なし)。

### 実行
- なし — smoke test (1-ep self-play) / `dev/test-bot` / RunPod GPU / auto-recover loop はすべて実施。
- ⚠️ reinforce/case6 系は 24GB+ VRAM 必須 (memory `project_runpod_a4000_oom`)。
  RunPod は RTX 3090 / A6000 等を選択。A4000 16GB は OOM 実績あり。
- ⚠️ best.pt 喪失 race / JAX best.pt.npz bug に注意 (memory `project_runpod_best_pt_lost` /
  `project_reinforce_jax_best_pt_npz_bug`。case3 は npz bug 修正済 = commit c0cd427)。

### 例外条件
- 採否が inconclusive かつユーザが希望した場合のみ、対象 iter に限り 300 対戦を追加実施 (要明示指示)。

## 仮説リスト (priority 順)

- [x] (P1) H1: self-snapshot を 4 つ目の opponent モード (`self_snapshot`=3) として追加し、frozen な
      自 param pytree を scan に通して opponent 行を自 agent 推論で埋める。
      — **全 self-play 仮説の前提となる土台**。これ単体では「過去 snapshot との対戦が
      reward に悪影響を与えないか」を確認 (curriculum late を self_snapshot に差し替えて 1 点確認)。
      — **inconclusive (iter1)**: 配線は完全成立 (200 iter 完走、unit test 5 + smoke pass)。
        iter5 の switch で win 0.984→0.766 と一時下降し snapshot が短期的に学習信号を供給したが、
        iter150 で ~1.0 飽和。frozen iter0 相手は信号が枯れる → H2 (pool 周期更新) が必須と確認。
        ⚠️ rollout 2 倍重で 5.1h/$7.1 (cap 大幅超過、[[project_reinforce_self_snapshot_cost]])。
- [x] (P1, depends on H1) H2: K iter ごとに snapshot を opponent pool に追加し、
      `baseline_jax_full` も curriculum late 相手に加える。pool 管理 (cap 付き) の基盤を作る。
      — pool 化で相手の多様性を確保し、単一 late 相手への過適合を防ぐ。
      — **inconclusive (positive-leaning, iter2)**: H1 の飽和を解消 (win last10 0.988→0.661、
        entropy 暴走 46→97 が有界 38→47 に)。decisive = baseline_jax_full 混合 (vs full 0.274、
        vs pool snapshot 0.828)。**vs full が 0.138→0.359 と上昇** (+0.0027/it) = 強い相手に
        勝てるよう学習進行。PFSP 前提が機能。$0.70 で完走 (H1 の 1/10)。n<300 で断定不可。
        → H4 (f_hard 優先 sampling) で vs full の伸びしろを取りに行く。
- [ ] (P2, depends on H1+H2) H4: PFSP `f_hard(x)=(1−x)^p` — 現 agent の各相手に対する勝率 x に
      反比例 (勝てない相手ほど高確率) して pool から相手を sampling。
      — AlphaStar 主手法。難敵に学習を集中させ最終強度を押し上げる (今回の主題)。
- [ ] (P2, depends on H1+H2) H5: PFSP `f_var(x)=x(1−x)` — 勝率 0.5 付近の相手を優先 sampling。
      — over-strong 相手での勾配消失を避け curriculum を平滑化。H4 と A/B 比較。
- [ ] (P3, depends on H1+H2) H6: rule (lite/full) と self-pool の混合比を検証
      (50/50 vs 純 self-pool)。— rule への対策を忘れる catastrophic forgetting を回避できるか。
- [ ] (deferred) H3: 一様 FSP (pool から等確率 sampling)。
      — PFSP (H4/H5) との比較 baseline。優先度関数を一様にするだけなので H4/H5 実装に内包可能。
- [ ] (deferred) H7: snapshot 頻度 / pool サイズ sweep (K=10/20, pool cap N=5)。
      — exploitation↔diversity のチューニング。採用系が効いた後に実施。
- [ ] (deferred) H8: iter 内 per-episode に pool から相手を割り振り (vmap)。
      — 1 iter 1 相手より勾配 variance 低減。工数大、効果確認後に検討。

## Iteration log

(各 iter 完了時に experiment-analysis / experiment が追記)

| iter | 開始 | 仮説# | plan path | run_id | 主要メトリクス | 採否 | result path |
|---|---|---|---|---|---|---|---|
| 1 | 2026-05-27 | H1 | iter1_plan.md | 20260527-145442__...__fb36504__seed0 | win last10=0.988 (飽和), iter6 dip 0.766, value_loss 0.104→0.052 | inconclusive | iter1_result.md / iter1_analysis.md |
| 2 | 2026-05-28 | H2 | iter2_plan.md | 20260528-005806__...__36982a3__seed0 | win last10=0.661 (飽和解消), vs full 0.138→0.359 (+0.0027/it), entropy 有界 | inconclusive | iter2_result.md / iter2_analysis.md |

## 参考 (References)

- [Grandmaster level in StarCraft II (AlphaStar, Nature 2019)](https://www.nature.com/articles/s41586-019-1724-z)
  — PFSP は league 全体を相手に、相手の win rate に比例して sampling 確率を調整。FSP を全指標で上回る。
- [AlphaStar supplementary (DeepMind PDF)](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)
  — `f_hard(x)=(1−x)^p` (難敵集中) と `f_var(x)=x(1−x)` (同レベル優先) の 2 つの重み関数。
  league = main agent + main exploiter + league exploiter の 3 種。
- [Zero-sum Game / League — DI-engine docs](https://opendilab.github.io/DI-engine/02_algo/league.html)
  — PFSP / league training の実装リファレンス (pool 管理・優先度 sampling の具体)。
- [TStarBot-X (arXiv 2011.13729)](https://arxiv.org/pdf/2011.13729) — efficient league training の OSS study。

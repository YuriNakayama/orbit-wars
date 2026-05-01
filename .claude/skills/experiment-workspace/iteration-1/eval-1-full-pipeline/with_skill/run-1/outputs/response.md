# Simulated Main-Session Response (`experiment` skill loaded)

User prompt:
> imitation/case1 で dropout を 0.2 から 0.3 に上げて再学習する実験をやりたい。LB1235 baseline 相手の win-rate 改善を狙う。plan.md を書いてコードを変更し、ローカルで pytest を回してから vast.ai で学習を走らせて、結果を result.md にまとめてほしい。

`experiment` スキルに従って 4 フェーズで進めます。内部推論は英語、ユーザー向け出力は日本語で行います。

---

## Phase 1 — Confirm scope before delegating

User has already specified:
- Family / case: `imitation` / `case1` (extending — same hypothesis family as earlier dropout-related iterations).
- Hypothesis: dropout 0.2 → 0.3, LB1235 baseline 相手の win-rate 改善を狙う。
- Pipeline: plan.md → コード変更 → pytest → vast.ai 学習 → result.md (full pipeline).

ただし以下は明示されておらず、Vast.ai は実コストが発生するため `AskUserQuestion` で 1 問だけ確認します。

### ユーザーへの質問 (AskUserQuestion 1 問構成)

```
質問 1: Vast.ai GPU 学習の opt-in と評価規模
  説明: full pipeline には Vast.ai の GPU 学習 (~$0.5–1.5 / 1 run, ETA 30–90 分想定) と
        評価 self-play を含みます。評価エピソード数を確定させたいです。
        プロジェクト方針として n<300 は noise なので 300 以上を推奨です。
  オプション:
    A) Vast 学習を含めて、評価は LB1235 相手 300 エピソードで実施 (推奨・full pipeline)
    B) Vast 学習を含めて、評価は LB1235 相手 500 エピソード (より厳格、時間 +α)
    C) Vast 学習はスキップし、まず plan.md とコード変更 + ローカル pytest までで止める
       (Vast launch 前に確認したい場合)
    D) その他 (自由記述)
```

期待: ユーザーは A を選ぶと想定。仮に B / D が来たら experimenter への handoff prompt 内の評価エピソード数だけ差し替える。C の場合は Phase 4 の Vast 関連 non-goal を「dev/vast train を実行しない、plan.md とコードと pytest までで停止」に切替。

> 想定回答: A (Vast 学習込み、評価 LB1235 vs 300 エピソード)

---

## Phase 2 — Quick reality check

`AskUserQuestion` 確定後、experimenter を spawn する前に以下を並列実行します。

### 実行する Bash コマンド (並列)

```bash
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse --short HEAD
ls backend/pipeline/imitation/
ls docs/experiment/imitation/ 2>/dev/null || echo "(no imitation experiment dir yet)"
```

### 想定出力

```
$ git status --short
 M .gitignore

$ git rev-parse --abbrev-ref HEAD
feature/reinforcement-learning-conversion

$ git rev-parse --short HEAD
057b19c

$ ls backend/pipeline/imitation/
__init__.py  case1/  case2/  .submitignore

$ ls docs/experiment/imitation/
20260419_case1_pure_il_cycle/
20260421_case1_focal_alpha_flip/
20260422_case1_phase3_oversample/
```

### 観測結果と判断

- `git status` に `.gitignore` 1 件の修正あり → **dirty tree**。Vast は SHA から学習するので、このまま `dev/vast train` するとこの変更は反映されない。
- 既存の `docs/experiment/imitation/` 配下に dropout 専用のディレクトリは無い。が、`20260422_case1_phase3_oversample/` は連続するイテレーションの最新地点。今回の dropout 実験は新トピックなので、新規ディレクトリ `docs/experiment/imitation/20260501_case1_dropout_sweep/` を切る方針 (これは experimenter の Phase 2 の判断材料として handoff prompt に書く)。
- ブランチ: `feature/reinforcement-learning-conversion` @ `057b19c`。

### ユーザーへの一言フィードバック (実テキスト)

> `git status` に `.gitignore` の未コミット変更が 1 件あります。Vast.ai は commit SHA から学習するため、この変更はそのままでは run に反映されません。以下のいずれかを選んでください:
>
> 1. `.gitignore` の変更を experimenter にコミットさせて push してから launch (おすすめ)
> 2. 該当変更は今回の実験と無関係なので stash して進める
> 3. 一旦中断してユーザー側でコミット / 整理する
>
> ユーザーの意向が無い場合、デフォルトは 1 (コミットしてから push) で experimenter に進めさせます。

> 想定回答: 1 (コミットしてから push)

---

## Phase 3 — Hand off to the `experimenter` agent

`Task` ツールで `subagent_type: "experimenter"` を起動します。エージェントはこの会話を見ないため、prompt は self-contained に書きます。

### Task 呼び出し

- `subagent_type`: `experimenter`
- `description`: `imitation/case1 dropout 0.3 sweep — full pipeline`
- `prompt`: 下記参照 (`agent_handoff_prompt.txt` と同内容)

### prompt 本文 (handoff)

```
User wants to run an experiment iterating on imitation/case1.

Hypothesis (verbatim from user, Japanese):
  「imitation/case1 で dropout を 0.2 から 0.3 に上げて再学習する実験をやりたい。
   LB1235 baseline 相手の win-rate 改善を狙う。plan.md を書いてコードを変更し、
   ローカルで pytest を回してから vast.ai で学習を走らせて、結果を result.md
   にまとめてほしい。」

Translated success metric: dropout 0.2 → 0.3 で imitation/case1 のモデルを再学習し、
LB1235 baseline 相手の self-play win-rate がベースライン (現行 dropout 0.2 の最新
チェックポイント) に対し有意に改善することを示す。

Scope confirmed in main session (Phase 1):
- Family: imitation
- Case: existing case1 (extend, do NOT create a new case)
- Compute target: full pipeline including Vast.ai GPU training
- Evaluation: LB1235 baseline 相手, 300 エピソード (n<300 は noise というプロジェクト
  ルールに従う)
- New module-or-edit-in-place: prefer adding a new training config / module
  (e.g. `training/config_dropout03.yaml` or `policy_v2.py`) so the dropout=0.2
  baseline remains comparable. Do NOT silently rewrite the existing 0.2 config.

Branch / commit context (from Phase 2):
- Branch: feature/reinforcement-learning-conversion
- HEAD SHA: 057b19c
- Working tree: dirty (.gitignore has 1 uncommitted change). User authorized
  committing the .gitignore change as part of your push-before-vast step.
  If the change is unrelated to this experiment, give it its own commit
  (e.g. ":wrench: gitignore tweak") separate from the dropout-experiment commit.

Experiment directory naming (per `.claude/rules/docs.md` and your Phase 2):
- Today's date: 20260501 (use this, do NOT call `date`).
- Use `docs/experiment/imitation/20260501_case1_dropout_sweep/plan.md`.
- This is the first iteration of this hypothesis directory, so plain
  `plan.md` / `result.md` (no `iter1_` prefix) is fine. Only switch to
  `iterN_*.md` if you discover an existing directory for the same hypothesis.

Plan.md must include (see your Phase 2 spec):
- Hypothesis: dropout 0.2 → 0.3, target = improved win-rate vs LB1235.
- Scope: which files under backend/pipeline/imitation/case1/ change
  (config / training script). Confirm rules/backend/pipeline.md compliance
  (relative imports, .submitignore, agents.py registration unchanged for
  existing case).
- Implementation steps with explicit paths.
- Local validation: list pytest paths covering case1.
- Remote training: stage `train_imitation_case1`, expected duration,
  checkpoint path under `data/output/models/imitation/case1/runs/<run_id>/best.pt`.
- Evaluation: vs LB1235, 300 episodes, primary metric win-rate, decision threshold
  (e.g. ≥ +3pp absolute over current 0.2 baseline, with confidence note that even
  300 episodes carries seed variance — see project memory
  `project_imitation_case1_phase3` and `project_case2_ablation`).
- Risks: dropout 0.3 may underfit on small imitation dataset; evaluation seed
  variance; LB1235 opponent-pool drift (intra-baseline only — do NOT cite
  Kaggle publicScore as evidence per project rule).

Phases to execute (your standard workflow):
- Phase 2: write plan.md.
- Phase 3: implement (prefer additive new config/module), `dev/test-backend`
  must pass before Phase 4. If it fails for an unrelated lint/type reason,
  delegate to `python-build-resolver` via Task; do not bypass.
- Phase 4: commit (.gitignore in its own commit, experiment changes in another),
  push to origin, then `dev/vast train 057b19c... --stage train_imitation_case1`
  (use the SHA you actually push, not 057b19c if you added new commits on top —
  capture the new SHA after push). User has pre-approved the GPU spend for this
  single launch.
- Phase 5: capture run_id, surface ETA to the main session, then return control.
  Do NOT busy-wait in this run.
- Phase 6: after the run, `dev/vast pull <run_id>`, run case1 evaluation script
  vs LB1235 for 300 episodes, write `docs/experiment/imitation/20260501_case1_dropout_sweep/result.md`
  with the result.md sections from your spec (Summary / Numbers / Diagnosis /
  Decision / Artifacts).

Non-goals (explicit):
- Do NOT submit to Kaggle. (`dev/submit submit`, `kaggle competitions submit`,
  cd-kaggle-submit workflow_dispatch all out of scope.)
- Do NOT run `dev/vast promote <run_id>`. Even if the result is great,
  promotion requires a separate explicit user approval.
- Do NOT cite Kaggle publicScore or skill rating as evidence in result.md;
  use only local match outcomes (project rule).
- Do NOT create a new case directory. Extend imitation/case1.
- Do NOT delete or rewrite the existing dropout 0.2 baseline config in place;
  add a new config alongside so the comparison stays clean.
- Do NOT touch `policy/weights.pt`. Vast artifacts go under data/output and
  are DVC-tracked; do not commit binary weights to git.

Return to the main session:
- Path of `plan.md`.
- Summary of code changes (files added/edited).
- `dev/test-backend` pass/fail summary (and SHA at which it passed).
- Pushed commit SHA.
- Vast `run_id`, stage, start time, rough ETA.
- One-paragraph next-step recommendation (e.g. "ETA ~50min, ping me on
  wakeup or call me back when run completes; I will then pull artifacts,
  run 300-ep eval vs LB1235, and write result.md").

Language: write all user-facing output (the messages you surface back to me)
in Japanese, per project policy.
```

エージェント終了後は、エージェントが返したサマリをユーザーに **そのまま日本語で** 中継します (再導出はしない)。

---

## Phase 4 — Follow-up coordination (forward-looking)

Vast 完走までの間および完走後に main session 側で対応する想定:

1. **Run 中の進捗確認**: ユーザーから「もう終わった?」と聞かれたら、再 spawn せず `dev/vast pull <run_id>` を直接叩いて状況を返す。result.md 更新まで求められたら experimenter を再 spawn。
2. **完走後の result.md 中継**: experimenter が書いた `result.md` のサマリと数字を日本語で要約してユーザーに提示。
3. **Promotion / submission のお伺い**: result が良好でユーザーが `dev/vast promote` を希望した場合は、`.claude/rules/command.md` に従って **その時点で個別に** 明示承認を取る (今回の opt-in は単発 train 分のみ)。Kaggle 送信も同様に独立した承認ゲート。
4. **300 戦未満で「効いた!」と言われた場合の留意**: プロジェクト memory `project_imitation_case1_phase3` の前例 (5/100 が 0/300 で否定) を引いて caveat を出す。今回は 300 戦で評価する設計なのでこのリスクは緩和済み。
5. **オフロード**: 追加で長時間待機が必要な場合は `ScheduleWakeup` で再開する。`sleep` ループは禁止。

以上が `experiment` skill での想定フローです。

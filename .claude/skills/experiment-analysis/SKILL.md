---
name: experiment-analysis
description: >
  Interactive post-mortem skill for finished Orbit Wars experiments. Reads an
  existing `result.md` (or freshly-pulled Vast.ai run artifacts) under
  `docs/experiment/{family}/`, walks the user through hypothesis-vs-outcome
  comparison, statistical significance checks, replay/seed drill-downs, and
  failure-mode pattern extraction via `AskUserQuestion`, then writes
  `analysis.md` (or `iterN_analysis.md`) capturing conclusions and proposing
  next-experiment hooks. Read-only on code: does NOT re-train, NOT launch Vast,
  NOT edit `backend/pipeline/`. Use whenever the user types
  `/experiment-analysis`, or asks to interpret / dig into / discuss / explain
  a finished experiment result — phrases like "iter9 の結果なんで負けてるか
  分析したい", "この result.md の数字どう解釈すべき?", "case2 の ablation
  の敗因を深掘りしたい", "300戦の結果踏まえて次どうする?",
  "loss seed の replay 見て分析まとめて" all count. Don't trigger this skill
  for designing a NEW experiment from scratch (use `experiment-plan`),
  full-pipeline execution / re-training (use `experiment-execution`),
  read-only code review, or plain bug fixes.
---

# Experiment Analysis Skill (Orbit Wars)

Interactive post-mortem for an experiment whose `result.md` already exists (or whose Vast.ai run just finished and the user wants to think through the numbers). The skill is the dialog layer between the user and a pile of metrics / replays / artifacts. It produces an `analysis.md` that captures interpretation and points at the next experiment.

## When this skill is in charge

- The user typed `/experiment-analysis` (explicit trigger), **or**
- The user asked, in plain language, for any of:
  - interpretation of an existing `result.md` (`iter9 の結果どう読むべき?`, `この 300戦の win-rate 50.7% は採用していい?`)
  - drill-down into a specific seed / replay / failure mode (`loss seed=0 の replay 見て敗因分析したい`, `この負け試合のターン推移を整理して`)
  - significance / variance discussion (`100戦と300戦で結果違うのはなぜ?`, `Wilson CI 計算して`)
  - "what's next?" pivots from a finished run (`この結果踏まえて次の iter どうする?`)
  - converting raw artifacts (just-pulled `data/output/models/<family>/case<N>/runs/<run_id>/`) into a written `analysis.md`
- The user did **not** ask to re-run the experiment or design a brand-new hypothesis. If they did, redirect to `experiment-execution` or `experiment-plan`.

If the user wants a **new experiment from scratch** → `experiment-plan`. **Re-train the same case** → `experiment-execution`. **Read papers / external survey** → `research-*`. **Code review** → `code-review` / `python-review`.

## Scope boundaries (what this skill does NOT do)

- Does not run `dev/test-backend`, `dev/vast train`, `dev/vast promote`, or any submission command.
- Does not edit code under `backend/pipeline/<family>/case<N>/`. May read it for context, never write.
- Does not commit or push.
- Read-only `dev/vast pull <run_id>` is allowed (it's how artifacts get to the local tree). `dvc pull` for missing artifacts is also allowed. Anything mutating beyond that needs explicit user confirmation per `.claude/rules/command.md`.
- Does not auto-spawn `experiment-execution`. The skill ends with a written `analysis.md` and an offer of next steps.

## Skill flow

The skill runs five phases. Phases 1–2 establish what's being analyzed; Phase 3 is the interactive interpretation; Phase 4 writes the file; Phase 5 offers next steps.

### Phase 1 — Locate the artifact

Resolve which experiment is being analyzed. Two starting points:

**A. The user named a result.md path or experiment directory.** Read `result.md` (or `iterN_result.md`) directly. If it doesn't exist yet but the directory does, ask whether to write an analysis on the **most recent** existing result file in that directory.

**B. The user named a Vast.ai run_id but no result.md exists yet.** Run `dev/vast pull <run_id>` (read-only) to fetch artifacts to `data/output/models/<family>/case<N>/runs/<run_id>/`. If `result.md` should exist but doesn't, redirect: `experiment-execution` is the right path for the first write of `result.md`. Analysis comes after.

If neither is given, do a quick `ls docs/experiment/{imitation,rulebase,reinforce}/` and surface the 5 most recent directories via `AskUserQuestion` for the user to pick.

Do **not** pull data the user didn't ask for. Pulling is a side effect, even if read-only — confirm before `dev/vast pull` if it's not already implied by the user's request.

### Phase 2 — Read context

Once the target is locked, read:

- The relevant `result.md` (or `iterN_result.md`) — start to finish.
- The matching `plan.md` / `iterN_plan.md` in the same directory — what was the hypothesis the user was testing?
- Prior `iterN_*.md` in the same directory if they exist — analysis usually fits in a chain of iterations.
- (Optional, if mentioned in `result.md`) referenced artifacts under `data/output/experiment/...` or `data/output/models/.../runs/<run_id>/metrics.json`. Keep reads narrow; don't preload everything.

Surface a 3-line summary back to the user (in Japanese): hypothesis from plan.md, headline numbers from result.md, the gap (did the hypothesis hold?). This is the orientation step before drilling in.

### Phase 3 — Interactive interpretation

Run **2 rounds** of `AskUserQuestion` (each up to 4 questions). Use the comparison-style options (`⭐推薦: option — ✅ pro / ⚠️ con` + `💡 推薦理由`) only when there are real trade-offs; plain options for diagnostic questions. The first question of the first round mentions: `各質問で「Other」を選ぶと自由記述も可能です。`

**Round 1: Verdict & evidence quality**

Pick 3–4 of these (skip what's already clear from `result.md`):

- **Verdict on the hypothesis**: `clearly held / clearly rejected / inconclusive (variance) / partially held (some opponents only)`. Pre-fill `⭐推薦` based on what the numbers actually say — be honest, not generous.
- **Statistical reliability**: `n ≥ 300 with Wilson CI separating from baseline / n ≥ 300 but CI overlaps / n < 300 (treat as noise per project memory)`. Recall memory `project_imitation_case1_phase3` — n<300 is not trustworthy on this project; flag it explicitly when the result is from a 100-game run.
- **Opponent-pool coverage**: did the experiment beat all named opponents, or only some? If mixed, which opponents matter most for the original hypothesis?
- **Signal vs. seed variance**: any sign that the headline number is dominated by one or two outlier seeds? (Memory `project_case2_ablation` documents seed-variance traps — surface this as a candidate diagnosis when n=100.)
- **Compute cost vs. value**: was the GPU spend justified by the information gained? (Useful for deciding whether to re-run at higher episode count or pivot.)

**Round 2: Failure-mode / mechanism drill-down**

Now the *why*. Pick 3–4:

- **Where to look next**: `specific seed replay drill-down (which seed?) / action distribution (NO_OP %, fire timing) / turn-by-turn trace of one game / training curve (loss / val metric over epochs) / class imbalance in dataset / weights diff vs. prior iter`. Each option points at a real artifact path under `data/output/...` or a script under `backend/pipeline/<family>/case<N>/scripts/`.
- **Candidate root cause**: 3–4 hypotheses for why the result is what it is. For losses, draw from project memory patterns: `target diversity 不足` (imitation), `passive gating (NO_OP 過多)` (imitation), `seed variance` (any), `submitignore 漏れ` (any), `from focal α 不適切` (imitation case1 specifically — memory `project_imitation_case1_phase2_breakthrough`).
- **Comparison with prior iters**: which earlier iteration's result is the closest analogue? What did *that* one's `result.md` conclude?
- **Cross-check / falsifiable claim**: what would we expect to see if the candidate root cause is correct? (e.g., "if it's seed variance, re-running with seed 100–200 should give a different mean.") Surface as `multiSelect: true` so the user can mark several diagnostic checks worth running.

**Optional drill-down step (no extra round)**: if the user picked "specific seed replay drill-down" in Round 2, *and* the relevant replay file exists under `data/lake/selfplay/matches/` or `data/output/...`, read it (or its decoded form) and fold the findings into the analysis. Cite turn numbers and concrete state changes (`turn 80: ships 200→123 で戦線崩壊`). Skip if the file isn't local — note it as a follow-up artifact to pull.

### Phase 4 — Write `analysis.md`

Resolve the target path, mirroring `.claude/rules/docs.md`:

```
docs/experiment/{family}/{yyyymmdd}_case{N}_{topic}/{analysis.md | iterN_analysis.md}
```

- Same directory as the `result.md` being analyzed (do NOT create a new directory).
- File name: `analysis.md` for the directory's first analysis pass, `iterN_analysis.md` if a prior `analysis.md` already exists or if the directory uses the `iterN_*.md` scheme. Surface the choice to the user before writing if ambiguous.
- Iteration migration: if the directory has plain `result.md` + plain `plan.md` and you're introducing the iterN scheme via `iter2_analysis.md`, rename the originals to `iter1_*.md` first per the docs.md rule. Confirm with the user before the rename.

Write the file with these sections — **keep it ≤ one screen unless the user explicitly asked for a long-form report**. Verbose analyses rot:

```markdown
# {Family}/{caseN} — {Topic} 分析

> 作成日: {yyyy-mm-dd}
> 関連: `plan.md` / `result.md` / 過去 iter 文書のパス
> 対象: {result.md のヘッドライン番号 1 行}

## 仮説と結果のギャップ
- 仮説: {plan.md の 1 行仮説}
- 観測: {result.md の主要メトリクス + n}
- 判定: {clearly held / rejected / inconclusive / partially held}
- 根拠: {Wilson CI、対戦相手別ブレ、seed variance の有無}

## 主要所見 (Findings)
1. {finding with citation: e.g. `iter9 の 5/100 は再評価 0/600 で否定 (Wilson 95% CI [0.00, 0.63%])`}
2. ...

## 失敗モード / 成功モード分析
- {特定 seed の挙動 / action 分布 / ターン推移 など、Round 2 ドリルダウンで得た情報}
- 参照アーティファクト: `data/output/...` のパス

## 候補根本原因
| 仮説 | 支持する観測 | 反証する観測 | 検証コスト |
|------|-------------|-------------|-----------|
| ... | ... | ... | ... |

## 次の実験フック (Next experiment hooks)
- {具体的な follow-up 仮説 1: e.g. `from focal α=0.75 を保ったまま target oversample 比率を再ablation`}
- {follow-up 仮説 2}

## 採否判断
- 本実験の構成: 採用 / 不採用 / さらに iter
- (採用なら) `dev/vast promote <run_id>` 実行可否は別途ユーザー確認が必要
```

Write directly with `Write` (or `Edit` for renames). Do not delegate to the `experimenter` agent — the analysis is interactive output, marshalling through Task adds latency.

### Phase 5 — Report and offer next steps

After writing, report to the user (in Japanese):

- Path of the written `analysis.md`
- 2–3 line summary of the verdict and top finding
- One-line offer of next steps, specific to what was decided:
  - If `next experiment hooks` is non-empty: `/experiment-plan` で次の iter の plan を書き起こせます
  - If verdict was "adopt": 必要なら `dev/vast promote <run_id>` を実行（要明示確認）
  - If verdict was "needs more episodes": `/experiment-execution` で同じ commit から n を増やして再評価できます
  - If the user surfaced a project-level finding: 必要なら memory に記録します（auto-memory プロトコルに従う）

Do **not** auto-spawn the next skill. The user explicitly came in to think; they'll trigger execution/planning themselves.

## Auto-memory hook

If during Phase 3 a clearly **project-level finding** emerges — i.e. something that would change how a future conversation should approach this codebase, not just a fact about this single iter — surface it to the user as a candidate memory and (with confirmation) save it via the auto-memory protocol. Examples that *do* warrant memory:

- "n=100 self-play results are unreliable on this project; ≥300 is the standard." (Already memoryized as `project_imitation_case1_phase3`.)
- "OM施策はこのプロジェクトでは default OFF が無難。" (Already memoryized as `project_om_finding`.)
- "Kaggle publicScore は信頼できない。" (Already memoryized via several entries.)

Examples that *don't* warrant memory (these go in `analysis.md` only):

- "iter5 の dropout=0.3 では loss が下がりきらなかった" — too specific.
- "seed=0 の試合で turn 80 に戦線崩壊した" — too specific.

Bias toward not adding memory unless the finding is durable and load-bearing for future sessions.

## Risk gates this skill enforces

- **Read-only on code**. Never edit `backend/pipeline/`, never re-train, never push, never submit. The only mutating action allowed is writing the `analysis.md` file (and the optional `iter1_*.md` rename).
- **Confirm before `dev/vast pull`**. Even though it's read-only on remote state, it changes the local file tree. Confirm if the user didn't already imply it.
- **n<300 caveat**. When the user is about to declare "the experiment worked" based on a 100-episode result, surface the project memory: 100戦は seed variance に飲まれる可能性が高い。
- **Kaggle publicScore is not evidence**. If `result.md` cites it, flag the citation as unreliable; do not let it influence the verdict.
- **Don't propose `dev/vast promote` autonomously**. If the analysis concludes "adopt", recommend that the user run `dev/vast promote <run_id>` themselves with explicit confirmation — promotion overwrites canonical weights.
- **Memory writes need confirmation**. Per the auto-memory protocol, project-level findings get memoryized only when the user confirms.

## Common shapes

| User says… | Skill behavior |
|---|---|
| "iter9 の result.md なんで負けてるのか分析して" | Phase 1 locates `docs/experiment/imitation/<dir>/iter9_result.md`, Phase 2 reads it + `iter9_plan.md`. Round 1 covers verdict + statistical reliability + opponent-pool. Round 2 drills into action distribution / passive gating / target diversity. Writes `iter9_analysis.md` (or `analysis.md` if directory has no iter scheme). |
| "さっき pull した run_xxx の結果分析して" | Phase 1 confirms `dev/vast pull` already ran (check artifacts exist locally); if not, asks user to pull first or do it with confirmation. Then Phases 2–5 normally on the matching `result.md`. If `result.md` doesn't exist yet, redirect to `experiment-execution`. |
| "case2 の ablation 結果、harass+half_step 採用していい?" | Phase 2 reads the ablation `result.md`, surfaces 300戦 50.7% (+1.4pp 非有意) headline. Round 1 verdict → likely "inconclusive (variance)". Round 2 drills into seed variance vs. real signal. Writes `analysis.md` recommending "実装維持、構造改修が次のフック"。 |
| "loss seed=0 の replay 詳しく見て敗因まとめて" | Phase 1 finds the replay path under `data/lake/selfplay/matches/` or `data/output/...`. Reads it, decodes turn-by-turn state. Round 1 covers the failure pattern; Round 2 generalizes (is it specific to seed=0 or systemic?). Writes a focused `analysis.md` with the trace table + diagnosis. |
| "結果踏まえて次どうすべき?" (vague) | Phase 1 asks which experiment via AskUserQuestion (list 5 most recent dirs). Then standard flow. |

## Things to avoid

- Re-running training or evaluation. The skill is interpretation, not generation. If the user wants more data, route to `experiment-execution`.
- Citing Kaggle publicScore as supporting evidence in the analysis.
- Declaring a result "significant" without checking n and Wilson CI.
- Being optimistic about a 100-episode result. Default to "inconclusive" until n≥300 evidence shows up.
- Writing a long-form report when a one-screen analysis suffices. Brevity > comprehensiveness for these documents.
- Auto-running `dev/vast promote` after concluding "adopt". Promotion overwrites canonical weights and needs explicit user confirmation per `.claude/rules/command.md`.
- Auto-spawning `experiment-plan` or `experiment-execution` after the analysis. Always wait for explicit user follow-up.
- Creating a new directory for the analysis. It always goes in the same directory as the `result.md` being analyzed.

## Language

- Internal reasoning and thinking should be in English
- **All user-facing output, AskUserQuestion labels/descriptions, and the written `analysis.md` body must be in Japanese** (per the project-wide policy in `.claude/CLAUDE.md`)

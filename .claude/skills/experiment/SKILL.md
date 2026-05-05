---
name: experiment
description: >
  Hypothesis-list–driven iteration loop for Orbit Wars experiments under
  `backend/pipeline/{imitation|rulebase|reinforce}/case<N>/`. Hears 3 things
  from the user (仮説リスト、最大 iteration 数、ループ周期) via
  `AskUserQuestion`、`docs/experiment/queue/{yyyymmdd}_{topic}.md` に仮説キュー
  を書き出し、`schedule` (cron) または `loop` (短周期) に「次の仮説を 1 件取り
  出して experiment-plan → experiment-execution → experiment-analysis を 1 巡
  し、結果を queue に書き戻す」ジョブを設定する。各ティック (1 iteration) は
  キューの先頭仮説を 1 件処理する単一サイクルであり、仮説が尽きるか、最大
  iteration に達するか、ユーザーが停止を指示するまで繰り返す。**This skill is
  invoked ONLY by explicit user request** (typed `/experiment` or said
  "実験を回す / 仮説リストを試す / iteration loop で実験したい"). Don't
  auto-trigger this skill from natural-language experiment requests — those go
  to `experiment-plan` (single plan) / `experiment-execution` (single run +
  RunPod) / `experiment-analysis` (single replay post-mortem) instead.
---

# Experiment Skill (Orbit Wars — iteration loop)

仮説リストを順番に消化する **iteration オーケストレータ**。1 ティック = 1 仮説 = `experiment-plan` → `experiment-execution` → `experiment-analysis` の 1 サイクル。最大 iteration 数 / 停止条件 / ループ周期はユーザヒアリングで決める。

このスキル自体は重い作業を行わない:
- ヒアリングでキューと設定を作る
- `schedule` / `loop` skill にループを登録する
- 各ティックで「キューの先頭を取り出して 3 スキルを順に呼ぶ」薄いドライバを動かす

実装の中身は 3 つの子スキル (`experiment-plan` / `experiment-execution` / `experiment-analysis`) にすべて委譲する。

## When this skill is in charge

- The user typed `/experiment` (explicit trigger), **or**
- The user explicitly asked for a multi-iteration loop: "仮説 5 個を順に試したい", "iteration loop で実験回して", "止めるまで仮説リストを消化し続けて", etc.
- The user did **not** ask for a single experiment. Single-shot requests ("imitation/case1 で X を試して") go to `experiment-execution` directly.

If unsure whether the user wants a loop vs. a single run, **ask once** before entering the loop. Don't auto-spawn iteration on a single-shot request — loops accumulate GPU spend.

## Skill flow

### Phase 0 — User hearing (3 questions)

Run **one round** of `AskUserQuestion` (3 questions). All use the selection format with free-text via "Other"; the first question's description includes: `各質問で「Other」を選ぶと自由記述も可能です。`

1. **どの仮説を調査するか (hypothesis list)** — 自由記述 ("Other") が主経路。複数仮説を 1 行 1 件で書いてもらう。`experiment-plan` の Step 0 と同じ形式 (family/case + 一文 hypothesis) を 1 件あたりで期待する。例:
   - `imitation/case1 dropout を 0.2→0.3 に上げる`
   - `imitation/case1 features に in-flight 比を追加`
   - `rulebase/case4 harass policy を comet 対応に拡張`
   選択肢には "テンプレート: 既存 case の hyperparameter sweep × 3" 等の seed 候補を 1-2 個示すが、Other で自由記述してもらう想定。

2. **最大 iteration 数 (stop condition)** — 以下から選択:
   - `1` (1 件だけ試して止まる — smoke 用途)
   - `3`
   - `5` (⭐推薦: 最初は短く回して観察)
   - `10`
   - `仮説リストを使い切るまで` (= `len(hypothesis_list)`)
   - `上限なし (ユーザー停止指示まで)` ⚠️ — 選んだ場合、RunPod GPU コスト累積の警告を 1 行で surface
   - `Other` (自由記述、整数のみ受け付ける)

3. **ループ周期 (cadence)** — 以下から選択:
   - `loop 60s` — フォアグラウンドで 60 秒ごと、ユーザーは同じセッションに居続ける必要あり
   - `loop 5m` — フォアグラウンドで 5 分ごと
   - `schedule */15min` — cron で 15 分ごと、ユーザがセッション離脱中も続行
   - `schedule hourly` — cron で 1 時間ごと
   - `schedule daily` — cron で 1 日 1 回
   - `逐次 (周期なし、前のティックが完了したら即次)` ⭐推薦: RunPod 学習が含まれる tick は 30 分〜数時間かかる。cron / loop の周期内に終わらないと多重起動 = 多重 GPU 課金になる。逐次なら衝突しない。
   - `Other` (自由記述、cron 式 or 分単位の整数)

⚠️ **周期 vs. tick 所要時間の整合**: 1 ティックが imitation 学習 + RunPod 評価を含む場合、tick 1 件で 30 分〜数時間かかる。cron 周期がそれより短いと多重起動 / 多重 GPU 課金になる。`schedule` を選ぶ場合は周期 ≥ 想定 tick 所要時間を user に確認する。RunPod を含まない rulebase の純粋ロジック iter なら短周期も可。

### Phase 1 — Write the hypothesis queue

Phase 0 で得た仮説リストを `docs/experiment/queue/{yyyymmdd}_{topic}.md` に書き出す:

```markdown
# Hypothesis queue — {yyyy-mm-dd} {topic}

> 作成日: {yyyy-mm-dd}
> 最大 iteration: {N | 上限なし}
> ループ周期: {loop 60s | schedule */15min | 逐次 | ...}
> 状態: pending

## Hypotheses

- [ ] (1) {family/case + hypothesis 1}
- [ ] (2) {family/case + hypothesis 2}
- [ ] (3) {family/case + hypothesis 3}
...

## Iteration log

(各ティック完了時に追記)

| iter | 開始時刻 | 仮説# | plan path | run_id | 主要メトリクス | 採否 | analysis path |
|---|---|---|---|---|---|---|---|
```

`{topic}` は仮説リスト全体を 1 語で表す snake_case (`imitation_case1_sweep`, `rulebase_iter_loop` 等)。

このキュー file が **全 iteration を通じた single source of truth**。tick driver は毎回これを読み、先頭の未消化 (`- [ ]`) 仮説を 1 件取り出す。Queue 自体は `docs/experiment/queue/` 配下なので git 管理対象 (人間可読 Markdown のみ — 機械生成物は `data/output/experiment/` 側へ)。

### Phase 2 — Register the loop

ユーザの選んだ周期に応じて以下を実行:

#### 周期 = `loop`

`loop` skill を `/loop {interval} /experiment` 形式で呼ぶ。Phase 3 の tick driver が、毎回起動時にキュー file を読んで自分の状態を判断する。

#### 周期 = `schedule`

`schedule` skill を呼んで cron ジョブを登録。ジョブの中身は「現在の worktree で `/experiment` 相当の tick を走らせる」もので、cron 1 回 = tick 1 回に対応する。停止時は cron を止める。

#### 周期 = `逐次`

`loop` / `schedule` を使わず、main session 内で `while not stopped` のループを回す。各ティックを foreground で実行し、完了したらすぐ次に進む。これが**最も衝突リスクが低い**。RunPod 学習が含まれる場合、user が他の作業に移れない代償を承知の上で選ぶこと (`dev/runpod watch` で desktop 通知だけ受け取って離席する運用も可能)。

### Phase 3 — Tick driver (1 iteration)

毎ティック以下を順に実行 (= 1 仮説 = 1 サイクル):

1. **Read queue** — `docs/experiment/queue/{yyyymmdd}_{topic}.md` を読む。
2. **停止条件チェック**:
   - 完了済み (`- [x]`) iter 数 ≥ 最大 iteration → cron / loop を停止して報告して終了
   - 未消化 (`- [ ]`) 仮説が 0 → 「キュー空」状態で停止
   - 状態が `paused` / `stopped` (user 操作) → 何もせず即終了
3. **次の仮説を pick** — 先頭の `- [ ]` 行を抽出。family/case と hypothesis 文を分離。
4. **`experiment-plan` を呼ぶ** — `Skill(skill="experiment-plan", args="{抽出した hypothesis 文}")` 相当を実行。Step 3 の web research は default で `不要` を採用し、user が事前に「論文も調べて」と指示した場合のみ skip しない。`iterN_plan.md` のパスを取得 (Orbit Wars の命名は `.claude/rules/docs.md` 準拠で `iterN_` のアンダースコアなし形式)。
5. **`experiment-execution` を呼ぶ** — Phase 1 を skip (hypothesis は確定済み) して Phase 2-7 の reality check → 実装 → test → RunPod 学習 → 評価 → result.md を回す。GPU コスト承認は **Phase 0 hearing 時点で iteration 全体に対して取得済み**として扱うが、想定の 2 倍以上の課金が見込まれるティックでは個別に user 確認する。`dev/runpod train` の `[y/N]` プロンプトは tick driver でも bypass しない (per `experiment-execution` の risk gate)。`run_id` と主要メトリクス (win-rate vs. opponent) を取得。
6. **`experiment-analysis` を呼ぶ** (ティック完了時のみ) — 「長戦 + 最速敗北」default 戦略で replay 分析。`data/output/experiment/{family}/case{N}/replay_analysis/{yyyymmdd_HHMM}/result_*.md` のパスを取得。
7. **キューを更新**:
   - 該当行を `- [x]` に変更し、analysis から得た「採否仮説 / 次に着目すべき点」を 1 行追記
   - `## Iteration log` テーブルに iter 番号 / 開始時刻 / plan path / run_id / win-rate / 採否 / analysis path を 1 行追加
8. **次仮説の動的追加 (オプション)**: `experiment-analysis` の `where_to_focus_next` で示された follow-up 仮説のうち、user が事前に「自動で follow-up 仮説を追加してよい」と承諾していた場合のみ、queue 末尾に `- [ ] (auto) ...` として追記する。default は **追加しない** (queue は user が決めたものを忠実に消化する)。

各ティック完了時に user に **3 行以内**で報告:
- iter N/{max}: family/case + hypothesis
- win-rate {x.y%} vs. {opponent}, 採否 {adopted|rejected|inconclusive}
- 次のティック予定 (`schedule` なら次の cron 時刻、`loop` なら interval、`逐次` なら "次仮説を即開始")

### Phase 4 — Stop and finalize

以下のいずれかで停止:

- 最大 iteration 到達
- キュー全消化
- user が "止めて" / "stop" と指示

ティックが失敗しても auto-pause しない。失敗 tick は Iteration log に `失敗` として記録し、queue の該当行は `- [x]` に変更して次の仮説に進む (user は queue file を見て後から失敗 iter を再投入できる)。連続失敗が構造的問題を示している可能性は user が log を見て判断する。

停止時:

1. **cron / loop を解除** — `schedule` で登録した job は必ず削除。`loop` も停止。`dev/runpod watch` を起動していたらそれも整理。stale な cron は次回セッション開始時に意図しない実行を起こす + RunPod に余分な pod を立てる可能性がある。
2. **キュー file を `状態: completed` (or `stopped`) に更新** — ヘッダの `状態:` 行を書き換え、`## Iteration log` の末尾に停止理由を 1 行追記。
3. **集計レポートを 5-10 行で報告** (Japanese):
   - 完了 iter 数 / 最大
   - 採否別カウント (adopted / rejected / inconclusive)
   - 最も win-rate 高かった iter (family/case + run_id + win-rate vs. opponent)
   - queue file パス + 各 iter の result.md / analysis.md パスへのインデックス
   - 推奨次アクション (例: "iter3 で +5pp 採用、`dev/runpod promote <run_id>` 実行 (user 確認後) → Kaggle submit は別途手動承認")

## Risk gates this skill enforces

- **Loop は明示的呼び出しのみ。** Natural-language の単発実験要求 ("imitation/case1 で X 試して") では起動しない。
- **RunPod GPU コストが累積する。** Phase 0 で「上限なし」を選んだ場合は警告を出す。Phase 3 の tick で想定 2 倍以上の予算消費が予測される場合は個別承認を取る。
- **cron 周期 ≥ 想定 tick 所要時間。** imitation 学習 + 評価が 1 時間かかるのに cron が 15 分周期なら多重起動 = 多重 GPU 課金。Phase 0 で必ず擦り合わせる。
- **`dev/runpod train` の `[y/N]` プロンプトは bypass しない。** Phase 0 のスコープ承認 ≠ per-command 承認 (per `experiment-execution`)。
- **停止時に必ず cron / loop を解除。** Stale ジョブは次回セッションで予期せず動き出す + RunPod に pod を立てる可能性がある。
- **Queue file は single source of truth。** Tick driver の状態 (どこまで進んだか) は queue file の `- [x]` / `- [ ]` のみで判定する。in-memory state を持たない (cron 跨ぎで失われるため)。
- **iteration 中に follow-up 仮説を queue に自動追加しない (default)。** User が事前に opt-in した場合のみ。queue は user が決めたものを忠実に消化するのが原則。
- **Kaggle submission / `dev/runpod promote` は loop の責務外。** ループは hypothesis 評価まで。canonical weights 更新 や Kaggle 提出は **必ず user の個別承認**。

## Common shapes

| User says… | Skill behavior |
|---|---|
| `/experiment` | Phase 0 hearing → queue 作成 → Phase 2 でループ登録 → Phase 3 で tick 開始。 |
| "imitation/case1 で 5 個の仮説を試したい、夜中に回しておきたい" | Phase 0 で hypothesis 5 件 / max=5 / `schedule hourly` を確定 (RunPod 学習の所要時間と擦り合わせ) → cron 登録 → 朝に集計レポート。 |
| "止めるまで仮説出し続けたい" | Phase 0 で max="上限なし" を確定、RunPod コスト警告 → `逐次` 推奨 → 各ティック完了で報告、user が "止めて" と言うまで継続。 |
| "仮説 1 個だけ試して" | これは loop ではなく単発。`/experiment-execution` にリダイレクトする。 |
| "今走ってるループ止めて" | Phase 4: cron / loop 解除 → queue 状態を `stopped` に → 集計レポート。 |

## Things to avoid

- 単発実験要求を loop に巻き込む。ユーザの明示的な loop 要求がない限り、子スキルを直接呼ぶ。
- 周期と tick 所要時間の整合確認をスキップする。多重起動は GPU 課金を倍々で吹き飛ばす。
- 停止時に cron / loop の解除を忘れる。Stale cron は意図しない RunPod 起動につながる。
- queue file 以外で iteration 状態を保持する (cron 跨ぎで失われる)。
- 各ティックで `experiment-plan` の web research を default で走らせる (時間 / コスト / context bloat の三重コスト)。user が明示した時のみ。
- Follow-up 仮説を勝手に queue に積み上げる。default は user の決めたリストのみ消化。
- `experiment-execution` の cost gate / `[y/N]` プロンプトを「Phase 0 で承認済み」を盾に bypass する。**個別 tick が想定の 2 倍以上の課金になる場合は個別承認**が原則、`[y/N]` は per-command で必ず通す。
- 採否判断に Kaggle publicScore を使う (per `experiment-execution`)。Local match outcomes only.
- ループから `dev/runpod promote` / Kaggle submit を自動で打つ。canonical 更新 / 提出は user の個別承認が必須。

## Language

- Internal reasoning and thinking should be in English
- **All user-facing output, AskUserQuestion labels/descriptions, queue file body, and tick reports must be in Japanese** (per the project-wide policy in `.claude/CLAUDE.md`)

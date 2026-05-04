---
name: experiment-analysis
description: >
  Replay-driven post-mortem skill for finished Orbit Wars experiments. Picks
  two match replays (典型的には win + loss / loss × 2 / win × 2)、Python
  スクリプトで 1 ターンごとのイベントと統計量を Markdown 表に変換 (result_1.md /
  result_2.md)、それを Claude が読み込んで「なぜ負けたか / どこは良くできていたか /
  次にどの turn に着目すべきか」を分析、サマリと NEXT ACTION を提案する。Read-only
  on code: does NOT re-train, NOT launch RunPod, NOT edit `bot/pipeline/`.
  Use whenever the user types `/experiment-analysis`, or asks to interpret /
  dig into / discuss / explain a finished experiment result with replay-level
  evidence — phrases like "対戦ログを分析して", "リプレイから敗因を読み取って",
  "勝った試合と負けた試合を比較して", "iter9 の負け試合のターン推移を整理して",
  "loss seed の replay 見て分析まとめて", "case2 の敗因を replay から深掘りしたい"
  all count. Don't trigger this skill for designing a NEW experiment from
  scratch (use `experiment-plan`), full-pipeline execution / re-training
  (use `experiment-execution`), read-only code review, or plain bug fixes.
---

# Experiment Analysis Skill (Orbit Wars)

Replay-driven post-mortem. Two matches → Python script で Markdown 化 → Claude が読んで分析 → サマリ + NEXT ACTION。Markdown 化は必ず外部 script で行い、Claude のコンテキストに生 replay JSON を流さない。

## Skill flow

5 phase 構成。Phase 2 のスクリプト経由 Markdown 化が context bloat を防ぐ要。

### Phase 1 — 対戦ログ 2 件選定

`AskUserQuestion` で以下を順に確認:

| Q | options |
|---|---|
| 選定戦略 | `長戦 (拮抗・終局直前) + 最速敗北 (構造的弱点) (推薦・両極から学べる)` / `win 1 + loss 1 (勝敗対比で why が明確)` / `loss × 2 (敗因深掘り)` / `win × 2 (勝ちパターン抽出)` / `seed 直接指定` / `Other` |
| 母集団 | `直近 selfplay run (index.parquet 最新 N 件)` / `特定 run_id` / `特定 opponent` / `Other` |
| 対戦 pick | (前段で絞った候補 5 件から選択。seed/対戦相手/勝敗/turn 数を表示) |

最初の質問の説明文に「各質問で『Other』を選ぶと自由記述も可能です。」と明記。

「長戦 + 最速敗北」を**デフォルト推薦**にする理由:
- **長戦** = 双方が中盤まで拮抗していた試合 = `total_turns` が母集団の上位四分位 (≥ p75)。中盤戦術 / 終盤詰め / 攻防バランスを観察できる。勝敗どちらでも可だが、自軍視点での挙動が長く取れる方が望ましい。
- **最速敗北** = 自軍が短い turn で完敗した試合 = `total_turns` が母集団の下位四分位 (≤ p25) かつ自軍 reward = -1。序盤の構造的弱点 (初手 neutral 確保失敗 / 早期 home 喪失 / 開幕 ship 散開ミスなど) が露骨に出る。
- 2 件の turn 長を意図的に対極に置くことで、「長く戦えた局面と崩壊した局面の差分」が見える。`win + loss` 比較より診断シグナルが強いことが多い。

選定実装:
- `data/lake/selfplay/matches/index.parquet` を polars で読み、winner / agents / total_turns で filter。
- 「長戦」: `total_turns >= quantile(0.75)` の中から最新 1 件 (または先頭)。
- 「最速敗北」: `total_turns <= quantile(0.25) AND self_reward == -1` の中から最新 1 件。
- index.parquet が無い場合、または `total_turns` カラムが無い場合は補助スクリプト `pick_match_pair.py` (下記 Python script セクション参照) で replay 中身を軽く読んで `total_turns` を集計し、上記分位で pick。
- worktree 配下の `data/` は symlink、実 replay は `/Users/user/project/orbit-wars/data/...` 側にあるので、絶対パスで script に渡すこと。

### Phase 2 — Python script で Markdown 化

選定した 2 replay を `replay_to_markdown.py` に渡し、`result_1.md` / `result_2.md` を出力。stdout は `wrote ... (N lines)` の 2 行だけ。**生 JSON / per-turn 表を Claude のチャットに貼らない**。

出力先は `.claude/rules/docs.md` に従い、対象 case の experiment ディレクトリ配下:

```
data/output/experiment/{family}/case{N}/replay_analysis/{yyyymmdd_HHMM}/result_{1,2}.md
```

(機械生成物なので `docs/experiment/` ではなく `data/output/` 配下。`analysis.md` から相対 / 絶対パスでリンク。)

### Phase 3 — Markdown 読み込みと分析

Claude が `result_1.md` / `result_2.md` を `Read` で取り込み、以下 3 観点を必ずカバーする:

1. **なぜ負けたか (why_lost)** — loss 試合の Turning points / Key events から決定打となった turn・イベントを特定。`ship_loss_burst` や連続する `planet_loss` の集中 turn が候補。仮説は最大 3 件、各々を支持 / 反証となる turn の事実で裏付ける。
2. **どこは良くできていたか (what_worked)** — win 試合 (または loss 試合の前半) で機能していた挙動。早期 neutral 確保速度、`enemy_planet_attack` のタイミング、production 立ち上がりカーブ、production / ship 比など。再現すべきパターンとして残す。
3. **次にどの turn / 局面に着目すべきか (where_to_focus_next)** — 仮説検証のために replay viewer や次回 script ランで深掘りすべき具体 turn 範囲・着眼メトリクス。例: `turn 50–80 の comet 出現直後の対応`、`turn 150 の総力戦時の in-flight 比`。

n=2 は仮説生成用であり結論は出さない。`project_imitation_case1_phase3` (n<300 不信頼) を踏まえ、断定ではなく「次に何戦見ればこの仮説が検証できるか」を必ず添える。

### Phase 4 — サマリ報告

ユーザに日本語で構造化提示 (本文に `result_1.md` / `result_2.md` のパスを必ず含む):

- **2 試合 headline 比較**: planets / ships / production / total_turns の差分表
- **why_lost**: 決定打 turn + 主要イベント 3-5 件 + 候補仮説 (最大 3、各々支持/反証)
- **what_worked**: 機能していた挙動 2-3 件 (turn 番号付き)
- **where_to_focus_next**: 着目すべき turn 範囲 + 見るべきメトリクス

### Phase 5 — NEXT ACTION 提案

具体的な follow-up を 2-3 件提示。例:

- `seed×10 で同一傾向か再評価` (n を増やすか同 case 再ラン)
- `turn N–M の action 分布を抽出して passive gating 検証`
- `次 iter で <仮説に基づく修正> を実装` (→ `/experiment-plan` で plan.md を起こす)
- `replay-viewer で seed=X を目視確認` (→ `/replay-viewer`)

他 skill は自動起動しない。提案文に skill 名を書くまで。

## Python script: 使用方法と仕様

### 配置

- `.claude/skills/experiment-analysis/scripts/replay_to_markdown.py` — 2 件の replay を Markdown 化 (Phase 2)
- `.claude/skills/experiment-analysis/scripts/pick_match_pair.py` — 「長戦 + 最速敗北」を母集団から自動 pick (Phase 1 デフォルト戦略)

### Phase 1 補助: `pick_match_pair.py`

`AskUserQuestion` で「長戦 + 最速敗北」を選んだ場合の自動 pick:

```bash
python3 .claude/skills/experiment-analysis/scripts/pick_match_pair.py \
  --replays-dir /Users/user/project/orbit-wars/data/lake/selfplay/matches/replays \
  --self-player-id 1 \
  --limit 200
```

stdout 2 行 (`long\t<abs_path>` / `fastest_loss\t<abs_path>`) のみを Claude が parse して、そのまま `replay_to_markdown.py --replay ... --replay ... --label long --label fastest_loss` に流す。stderr に診断（候補母集団件数、long/fast 閾値）。

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--replays-dir` | (必須) | `*.json.gz` を持つディレクトリ。symlink 経由 OK |
| `--self-player-id` | `1` | 自軍 player ID。selfplay の場合 case 設定に従う |
| `--limit` | `200` | 直近 N 件のみスキャン (大量 replay 時の負荷抑制) |
| `--long-quantile` | `0.75` | total_turns の上位四分位を「長戦」とみなす閾値 |
| `--fast-quantile` | `0.25` | total_turns の下位四分位を「最速敗北」候補閾値 |

### Phase 2 メイン: `replay_to_markdown.py`

### 起動コマンド

```bash
python3 .claude/skills/experiment-analysis/scripts/replay_to_markdown.py \
  --replay <ABS_PATH_A.json.gz> --replay <ABS_PATH_B.json.gz> \
  --label win --label loss \
  --out-dir data/output/experiment/{family}/case{N}/replay_analysis/<id>/
```

`uv` 不要 (標準ライブラリ + 必要なら polars のみ)。worktree から走らせる場合は replay パスを **絶対パス**で渡す (worktree 直下の `data/lake/selfplay/matches/replays/` は dvc pointer 経由で実体不在のことがある)。

### CLI 引数

| 引数 | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `--replay PATH` | ○ (2 回) | — | 対象 `.json.gz` の絶対パス。順序が `--label` と対応 |
| `--label LABEL` | ○ (2 回) | — | `win` / `loss` / `seed42` 等。出力ファイル冒頭に表示 |
| `--out-dir DIR` | ○ | — | `result_1.md` / `result_2.md` の出力先 |
| `--player-id INT` | — | label に応じて自動推定 (loss 系 label なら敗者、それ以外は勝者) | 視点プレイヤ ID。明示すれば override 可能 |
| `--ship-loss-abs INT` | — | `20` | ship 大量損失検出の絶対閾値 |
| `--ship-loss-rel FLOAT` | — | `0.30` | ship 大量損失検出の相対閾値 (前 turn 比) |
| `--full-stats` | — | OFF | 全 turn 統計表を含めるか |

### 出力 Markdown レイアウト (1 ファイル ~200-400 行)

```markdown
# Replay seed={N} — {label}

## Meta
- file / viewpoint / winner / rewards / statuses / total_turns

## Headline stats (final)
| metric | self | opponent |
| planets / ships / production / in-flight |

## Turning points (top 5 by ship-margin delta)
| turn | delta | side | note |

## Key events
| turn | side | type | detail |
（type: planet_gain / planet_loss / ship_loss_burst / enemy_planet_attack）

<details><summary>Full per-turn stats</summary>  ※ --full-stats 指定時のみ
| turn | mp | op | ms | os | mpr | opr | mif | oif |
</details>
```

### イベント検出ロジック

| type | 条件 |
|---|---|
| `planet_gain` | `planets[id].owner` が前 turn と異なり、自軍 (または敵が他敵から奪取) に変化 |
| `planet_loss` | 自軍が所有していた planet が他 owner に変化 |
| `ship_loss_burst` | ship 総数の前 turn 差分が `-ship-loss-abs` 以上 **かつ** `ship-loss-rel × prev` 以上 |
| `enemy_planet_attack` | `actions[t]` の発射先 angle が敵所有 planet 方向 (最近接マッチ) で、ship 合計を集約 |

### サイズガード

上限 ~600 行 / ~40KB を script 内で assert。超過時は events を turn 順 30 件にトリム + `--full-stats` 強制 OFF。

### 想定環境

- 1v1 リプレイ初期サポート。FFA (4-player) は WARN 表示で続行 (他 3 名を集約 opponent 扱い)
- 依存: 標準 `gzip` `json` `math` `argparse` `dataclasses` `pathlib` のみ。polars は Phase 1 の index.parquet 読みでのみ使用 (script 自体は依存ゼロ)

## Common shapes

| User says… | Skill behavior |
|---|---|
| "勝った試合と負けた試合を比較して" | Phase 1 で win+loss 1 件ずつ pick → Phase 2-5 を素直に実行 |
| "case2 の敗因を replay から深掘り" | Phase 1 で loss × 2 を pick → Phase 3 の why_lost セクションに重み |
| "iter9 で何が機能してた?" | Phase 1 で win × 2 を pick → Phase 3 の what_worked セクションに重み |
| "seed=42 と seed=99 比較して" | Phase 1 「seed 直接指定」分岐 → 該当 2 件を直接 script へ |

## Language

- Internal reasoning and thinking should be in English
- **All user-facing output, AskUserQuestion labels/descriptions, and the analysis summary must be in Japanese** (per `.claude/CLAUDE.md`)

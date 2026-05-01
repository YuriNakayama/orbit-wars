# Simulated main-session response — `experiment` skill, plan-only scope

## Phase 1 — スコープ確認

ユーザー発話で以下は既に明示されています:

- family / case: `rulebase/case2` を「改良する」= 既存 case2 の延長 (新 case ではない)
- 仮説: harass policy を追加し、comet 周辺で挑発戦術を試す
- スコープ: 「まず plan.md だけ書いてレビューさせてほしい。実装と vast はまだ動かさないで」

スキルの指示通り、`plan.md だけ` と明示された場合は Vast 確認の質問はスキップします。残るあいまい点は「`case2` 内で baseline モジュールを残すか上書きか」「評価対戦相手は case2 baseline で良いか」程度ですが、これは agent 側 Phase 2 で plan.md にドラフトさせて、レビュー時にユーザーに確認してもらう方が早いと判断し、追加質問はしません (スキルの「1–2 質問まで」原則に従う)。

→ `AskUserQuestion` は呼ばず、Phase 2 へ。

## Phase 2 — 軽い事前チェック

```bash
git status --short
# M .gitignore     (作業ツリーは概ねクリーン、plan.md 書くのに支障なし)

git rev-parse --abbrev-ref HEAD
# feature/reinforcement-learning-conversion

ls backend/pipeline/rulebase/
# case0  case1  case2  case3  case4  case5   → case2 は既存、確認 OK

ls docs/experiment/rulebase/
# 20260418_case1_optimize
# 20260420_case3_rollout_ablation
# → case2 harass のディレクトリは未作成。今回新規作成する形で agent に指示。
```

Vast は今回叩かないので push 状態は問わない。`M .gitignore` も plan.md 書きには無関係なので、そのまま Phase 3 へ。

## Phase 3 — `experimenter` agent への委譲

`Task(subagent_type="experimenter", ...)` で以下のプロンプトを渡してエージェントを起動 (このシミュレーションでは実際には spawn しません)。プロンプト本文は `agent_handoff_prompt.txt` 参照。

要点:

- 仮説と「plan.md only」スコープを明記
- `docs/experiment/rulebase/20260501_case2_harass_policy/plan.md` に書くよう指示
- 非ゴール: 実装しない / pytest しない / push しない / `dev/vast train` 叩かない / Kaggle 提出しない / `dev/vast promote` しない
- rulebase は学習不要なので plan.md には「Vast.ai 不使用」と明記させる
- 評価は ≥300 戦 (memory `project_imitation_case1_phase3` の教訓)、Kaggle publicScore は使わない (memory `project_case5_validation` の教訓)
- memory `project_case2_ablation` (Harass+HALF_STEP は 300戦 +3.7pp 非有意) を踏まえ、今回の harass policy が前回 ablation と何が違うのかを plan.md で言語化させる

## Phase 4 — フォローアップ調整 (シミュレート)

agent が plan.md を書き終えたら、ユーザー向けには以下の体裁で返す想定:

```
plan.md を書きました。

- パス: docs/experiment/rulebase/20260501_case2_harass_policy/plan.md
- 仮説: comet 出現ターン (50/150/250/350/450) 前後で敵 home/前線寄り惑星に harass fleet
  を送り、comet 争奪の主導権を奪って 1v1 win-rate を case2 baseline 比 +3pp 以上改善する。
- スコープ: backend/pipeline/rulebase/case2/ に policy_harass.py を新規追加 (baseline policy
  は触らず比較可能性を保つ)。main.py で flag 切り替え。
- 評価: case2 baseline 相手に 300 戦の self-play、1v1 win-rate を主指標。Kaggle スコアは
  指標として使わない。
- Vast: 不要 (rule-based、学習なし)。
- 既知リスク: project_case2_ablation で類似の Harass+HALF_STEP が +3.7pp / 300戦 / 非有意
  だった件を plan.md の「Risks」に明記。今回の差分 (comet 周辺フォーカス) で何が変わるかを
  説明できる必要あり。

レビューしてください。OK なら Phase 3 (実装) に進みます。実装に入る前に Vast について
あらためて確認しますが、rulebase なので今回は不要の見込みです。
```

その後の follow-up 想定:

- ユーザーが plan.md にコメントを返してきたら → agent を再 spawn せず、main session で `Edit` ツールで plan.md を直接修正する (軽微な場合)。構造的な書き直しが要るときだけ `experimenter` を再起動。
- 「OK、実装まで進めて」と言われたら → 改めて `experimenter` を Phase 3 から起動。Vast 不使用で評価まで完走させる。
- 「Kaggle に出して」と言われた場合 → スキルのガード通り、提出は experiment フローには含めず、明示的な submission 承認ゲートを別途要求する。

---

(以下は実際にユーザーへ返すメッセージ部分のドラフト)

---

`rulebase/case2` の改良実験、plan.md だけ先に書く方針で進めます。実装と Vast は今回は動かしません。

事前チェック:
- ブランチ: `feature/reinforcement-learning-conversion` (作業ツリーはほぼクリーン、`M .gitignore` のみ)
- `backend/pipeline/rulebase/case2` は既存なので延長扱い、新 case は作りません
- `docs/experiment/rulebase/` 配下に case2 harass の既存ディレクトリは無いので、`20260501_case2_harass_policy/plan.md` を新規作成します

これから `experimenter` エージェントを起動して plan.md だけ書かせます。書き上がったら内容をこちらで要約してお渡しするので、レビューをお願いします。実装に進む段階であらためて確認します (rulebase なので Vast.ai は基本不要の見込みです)。

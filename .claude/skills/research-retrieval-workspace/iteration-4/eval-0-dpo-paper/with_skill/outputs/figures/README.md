# Figures

論文から取得した図表ファイル。

WebFetchでダウンロードされた元画像ファイルは以下の場所に保存されています:

| ファイル名 | 元パス | 説明 |
|-----------|-------|------|
| fig-001-dpo-overview.png | webfetch-1774779606673-eurufc.png | Figure 1: DPOとRLHFのパイプライン比較図 |
| fig-002-reward-kl-frontier.png | webfetch-1774779609529-rxecpv.png | Figure 2左: IMDb感情制御の報酬-KLフロンティア |
| fig-003-tldr-winrate.png | webfetch-1774779624540-pqz1mw.png | Figure 3: TL;DR要約の勝率比較 |
| fig-004-dialogue-winrate.png | webfetch-1774779627022-oc3iez.png | Figure 4: Anthropic-HH対話の勝率比較 |
| fig-005-dialogue-evolution.png | webfetch-1774779627254-md7dxb.png | Figure 5: 対話勝率の学習推移 |

元画像のコピーコマンド:
```bash
SRC="/Users/user/.claude/projects/-Users-user-project-research/399f18d5-bea3-4d71-9d94-e0325d06f9c1/tool-results"
DST="/Users/user/project/research/.claude/skills/research-retrieval-workspace/iteration-4/eval-0-dpo-paper/with_skill/outputs/figures"
cp "$SRC/webfetch-1774779606673-eurufc.png" "$DST/fig-001-dpo-overview.png"
cp "$SRC/webfetch-1774779609529-rxecpv.png" "$DST/fig-002-reward-kl-frontier.png"
cp "$SRC/webfetch-1774779624540-pqz1mw.png" "$DST/fig-003-tldr-winrate.png"
cp "$SRC/webfetch-1774779627022-oc3iez.png" "$DST/fig-004-dialogue-winrate.png"
cp "$SRC/webfetch-1774779627254-md7dxb.png" "$DST/fig-005-dialogue-evolution.png"
```

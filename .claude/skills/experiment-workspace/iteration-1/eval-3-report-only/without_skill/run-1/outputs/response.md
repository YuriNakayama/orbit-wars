了解しました。`run_20260501_abc123` の結果を imitation/case1 の iter4 として記録します。

実際の `dev/vast pull` は走らせず、ペーパーシミュレーションでドキュメントを起こします。記録先は `.claude/rules/docs.md` の規約に従い、既存の imitation/case1 系ディレクトリに `iter4_result.md` として追加する形にします。

## 確認したこと

- ルール `.claude/rules/docs.md` のディレクトリ命名規約: `docs/experiment/{imitation|rulebase|reinforce}/{yyyymmdd}_case{N}_{topic}/` 配下に `iter{N}_result.md` を置くのが正しい運用
- 既存の imitation/case1 配下のディレクトリ一覧:
  - `docs/experiment/imitation/20260419_case1_diagnosis/`
  - `docs/experiment/imitation/20260420_case1_baseline/`
  - `docs/experiment/imitation/20260420_case1_improve/`
  - `docs/experiment/imitation/20260420_case1_pure_il_cycle/`
  - `docs/experiment/imitation/20260422_case1_loss_seed0_analysis/`
  - `docs/experiment/imitation/20260422_case1_phase3/`
  - `docs/experiment/imitation/20260422_case1_win_vs_loss_analysis/`
- Vast.ai 成果物の保存先 (`.claude/rules/command.md`): `data/output/models/imitation/case1/runs/<run_id>/best.pt`

## 質問させてください

iter4 として記録する前に、いくつか情報をいただかないと `result.md` を埋めきれません。以下を教えてください。

1. **どのディレクトリに iter4 を追加しますか?**
   既存ディレクトリのどれの続きにあたる run でしょうか? もし新トピックであれば新規ディレクトリ (例: `docs/experiment/imitation/20260501_case1_<topic>/`) を切ったうえで `iter1_result.md` から始める形になります。トピック名 (snake_case 短く) もあわせて教えてください。
   - 候補が `20260422_case1_phase3` の続編なら、現状 `result.md` (=iter1〜iter3 を内包) しかないので、ルールに従って一度 `iter1_result.md` にリネームしたうえで `iter4_result.md` を追加します。
   - そもそも iter1〜iter3 がどの run に対応するかも整理しておきたいので、対応関係を把握していれば教えてください。

2. **iter4 の仮説 / 変更点**
   この run で何を変えたか (loss 重み・データ拡張・ハイパラ・アーキ変更など)。`plan.md` 相当の内容がないと `result.md` の "背景" / "変更点" セクションが書けません。

3. **手元にある実測値 (ペーパーで埋めない部分)**
   - val metrics (per-template F1, accuracy 等)
   - 自己対戦勝率 (vs どの baseline で何 episodes、できれば 300-game)
   - 学習曲線 / 損失推移 のサマリ
   `dev/vast pull` を実行しないので、私のほうでログから数値を引っ張ってくることはできません。手元に控えがあるものを貼ってください。なければ、後で値が確定した時点で追記する前提のスケルトンだけ先に作る方針でも構いません。

4. **採用判断**
   iter4 weights を `policy/weights.pt` に promote する予定はありますか? (`dev/vast promote` 実行有無)。`memory/MEMORY.md` の Phase 3 メモにある「n<300 評価は信頼不可」の方針があるので、300-game 未満であれば `result.md` の結論は「保留」にします。

## 次のアクション (案)

回答をいただいたら、こちらで実施する作業はこうなります:

1. 必要なら既存 `result.md` を `iter1_result.md` にリネーム (`git mv`)
2. `docs/experiment/imitation/<対象ディレクトリ>/iter4_result.md` を新規作成
   - 構成: 背景 / 変更点 / 学習設定 (run_id, commit_sha, epoch, hparams) / val metrics / 自己対戦勝率 (n と 95% CI 込み) / 結論・採用判断 / 次の iter
3. 関連する大きな数値ファイル (json / png) は `data/output/experiment/imitation_case1_iter4_*.json` 配下に置き、`result.md` からはパスでリンクするだけにする (これも `.claude/rules/docs.md` の規約)
4. `git add docs/experiment/...` までやって commit はユーザー承認待ちで止める

`run_20260501_abc123` で何を狙った run だったか (上記 1〜4) を教えてもらえれば、すぐ書き起こします。

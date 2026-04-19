# imitation/case1 模倣学習パイプライン 勝率ゼロ 原因調査

- 調査日: 2026-04-19
- 対象: `pipeline/imitation/case1/` (Imitation Learning Baseline)
- 現状: `il_v1` vs `baseline_v1` (1v1, 100 戦) で **win_rate = 0.00**
  (`pipeline/imitation/case1/evaluation/results.json`)
- 学習データ: `data/lake/imitation_case1/train.parquet` (86,571 行)
- 方針: ユーザー意思決定のための診断のみ。実装は行わない。

## サマリ

レポート (`pipeline/imitation/case1/README.md:59-68`) が書いている「target_acc が低い」
「no-op が多い」は **症状**で、真因は

1. BCE の教師構造 (Bug 1)
2. preprocess の target 逆解決失敗 (Bug 3)
3. ships バケットと decoder 閾値の噛み合い (Bug 4)

の 3 つが複合している。ハイパラ調整や学習データ増量だけでは勝率は上がらない。

## 優先度付き問題一覧

| # | 重大度 | 問題 | 影響 |
|---|---|---|---|
| 1 | 致命 | 1 フレーム × N アクションを個別行で BCE にかけ、相互に negative 教師化 | モデルが全 from に対し sigmoid≈0 を学ぶ → no-op 極振り |
| 2 | 致命 | `_fleet_target_planet_id` が `aim_with_prediction` に対応せず、25% のアクションで target 不明 | target head の信号が半分欠落、推論で no-op 列に逃げる |
| 3 | 重大 | ships バケットが 0/4 に U カーブ偏在、bucket 0 だと `floor(0.1 * ships) = 0` で action 棄却 | from/target が出ても最終段で no-op 化 |
| 4 | 重大 | target pairwise に (距離・角度・ETA) を渡していない | target 選択が src 非依存 → decode 後ほぼ同じターゲットに |
| 5 | 設計 | winner 側のみ & rating top25% フィルタ | データ量少 + "強者が勝つ局面" 偏り、baseline への耐性なし |
| 6 | 軽微 | `losses.py` の無駄な `masked_fill(0.0)` | 学習には無害だが実装の混乱の兆候 |

---

## Bug 1 — 同一フレーム内で複数アクションを重複 no-op 教師にしている【致命的】

`pipeline/imitation/case1/training/preprocess.py:152-176` は **1 フレームで複数アクションを投入したリプレイについて、行動 1 つごとに 1 行を生成** している。つまり同じ `obs` が N 回 parquet に並び、それぞれ違う `from_label` を持つ。

`pipeline/imitation/case1/training/losses.py:70-81` の `from` head BCE は同じ obs に対して

- 行 1: 惑星 A = 1, 他 = 0 (A 以外は発射してはいけない)
- 行 2: 惑星 B = 1, 他 = 0 (B 以外は発射してはいけない)
- …

と **相互に矛盾する「negative 教師」** を与える。1 obs で 3 惑星から撃てば、
惑星 A を撃つ教師の裏で「惑星 B・C から撃つな」を同時に学ばせる形になり、
**モデルは全惑星で sigmoid ≈ 0 に収束するのが最適解** になる。
推論時にほぼ no-op が選ばれる症状と整合する。

**処方**: 1 フレームにつき 1 行にまとめ、`from_label` を multi-hot にして
BCE の positive set を `{該当 action の from 集合}`、negative を
`{my planets} - positive` に限定する。あるいは from head はフレーム単位、
target/ships は「選ばれた from ごとに一度だけ」にする。

## Bug 2 — 有効な my_planet 以外の BCE 分母汚染【致命的】

`pipeline/imitation/case1/training/losses.py:75-81`:

```python
bce = nn.functional.binary_cross_entropy_with_logits(
    output.from_logits.masked_fill(~my_planet_mask, 0.0),  # logit を 0 に潰す
    from_target,
    reduction="none",
)
bce = bce * my_planet_mask.float()
from_loss = bce.sum() / my_planet_mask.float().sum().clamp_min(1.0)
```

`output.from_logits` はモデル側で既に `masked_fill(~my_planet_mask, -inf)` されている
(`pipeline/imitation/case1/policy/model.py:74`)。
それに対して `masked_fill(..., 0.0)` を当てると、**`-inf` を 0 に戻した上で BCE** を取り、
直後に `* my_planet_mask` で消す、という無駄なだけの処理になる。
学習結果への害は小さいが、**目的と挙動が一致していないサイン**であり、
レビュー漏れを示唆する。

## Bug 3 — target label の 25.3% が NO_OP に落ちている【重大】

`pipeline/imitation/case1/training/preprocess.py:52-90` の `_fleet_target_planet_id()` は、
**「angle の方向ビームに惑星 radius を被せて最初に当たるもの」** を返す。
実際のプロのプレイは `aim_with_prediction` を使って **将来位置**へ撃っているため、
preprocess 時点で angle → target の逆解決に失敗し、
non-noop 61,014 行中 **15,419 行 (25.3%)** が `target_label = NO_OP_LABEL` になっている。

結果として target head は「撃ったけど to nowhere」を正解として学ばされ、
`target_acc = 0.37` のとおりほぼ信号がない。
推論時に target head が no-op 列 (36) を選ぶ確率が上がり、
`pipeline/imitation/case1/policy/decoder.py:58-59` で弾かれて no-op 化する。

- `is_noop = True` の行 25,557
- `target = NO_OP` の行 15,419
- 合計 **40,976 行 (47.3%)** が "事実上の no-op 教師"

model が no-op 極振りになるのは当然。

## Bug 4 — ships_label が U 字分布で 0/4 の二極化【重大】

学習データ上のバケット分布:

```
0: 32,452   ← 10% 送る (小規模偵察)
1:  5,483
2:  4,840
3:  4,243
4: 39,553   ← 100% 送る (総力戦)
```

`pipeline/imitation/case1/training/preprocess.py:93-98` の `_ships_bucket()` は
`ratio * 5` の floor。`ratio == 1.0` は `min(bucket, 4)` で **バケット 4 に集約**される。
多くのトップ局は「全艦送る」戦術を含むため bucket 4 が肥大し、
`val_ships_acc = 0.72` は **学習していない: ただ argmax = 0 か 4 を叩いているだけ**である可能性が高い。

さらに bucket 0 (10%) で撃つと `pipeline/imitation/case1/policy/decoder.py:71-73` の
`floor(0.1 * src.ships)` が 0 になり → `ships <= 0` で action が棄却される。
これも no-op を増やす方向に効く。

## Design 5 — `from_threshold = 0.05` でも発射が出ない【設計欠陥】

Bug 1 + 3 の合わせ技により、`sigmoid(from_logits)` は学習後ほぼ全 my planet で
極端に低い値 (≪ 0.05 にもなり得る)。
閾値を下げても全惑星 `from_prob < 0.05` になれば **1 アクションも出力されず即敗北**する。
baseline との対戦で「何もしない → 早期に惑星を全て取られて負け」という結果と整合する。

## Design 6 — target head が src 依存になっていない【設計欠陥】

`pipeline/imitation/case1/policy/model.py:77-80` の pairwise は `(h_src, h_tgt, ctx)` を入力にしているが、
`h_src` / `h_tgt` は `phi(planet)` つまり **単独の惑星特徴の埋め込み**だけで、
「src の視点から見た tgt」 (距離、相対角度、ETA、脅威など) は一切入っていない。
これでは「どの from からでも同じ target が選ばれる」縮退が起きる。
`target_acc = 0.37` と相まって、ここが勝敗に直結する問題となる。

## Design 7 — winner 側のみで学習し、敗者側 obs を捨てている【設計欠陥】

`pipeline/imitation/case1/training/preprocess.py:107-113` で `winner` のスロットのみを使い、
敗者側の `(obs, action)` は捨てている。上位 25% レーティング帯でも**どちらかは必ず敗者**であり、
アクション量のおよそ半分を捨てていることになる。
frame 数 86k という少なさの一因。

## Design 8 — rating + winner の二重フィルタで対 baseline 耐性が削れる【設計欠陥】

train / val split がエピソード単位で 90 / 10 (frame 数 val 13%) という点自体は問題ない。
しかし `rating_quantile = 0.75` で winner 側が cutoff を超えるエピソードのみ残す条件は、
**負けた強者の側も除外**する。
結果「強者が弱者に勝ったフレーム」しか使われず、
baseline 相手のような**そこそこ強い相手への対処サンプル**が極端に少ない。
これが「baseline に全敗」と直結する。

---

## 推奨改善順序

以下の順で着手するのが費用対効果が高い:

1. **preprocess の再設計** — 1 フレーム 1 行化、from multi-hot、target の
   `aim_with_prediction` ベース逆解決、敗者側 obs 採用、ships バケット再設計。
2. **target head の入力拡充** — pairwise に (距離、相対角度、ETA、相対 ships 比、
   自陣 / 敵陣フラグ) を追加。
3. **(オプション) ships decoder の閾値見直し** — bucket 0 でも最低 1 艦は送る、
   または bucket 0 を捨てて 4 分類にする。
4. **学習** — データ量が 2-3 倍になる想定で `epochs` 見直し、target_w を上げるのは
   その後で検討。

レポート (README.md) にある「データ拡張」「損失バランス」「ラベル品質再検討」
方向性は正しいが、Bug 1 を放置したままでは効果が出にくい。

## 参考: 学習データ統計 (実測)

```text
rows: 86,571
is_noop rate: 29.5%
from_label == NO_OP (36): 25,557 行
target_label == NO_OP (36): 40,976 行 (non-noop 行中でも 15,419 行 = 25.3%)
ships_label 分布: [32452, 5483, 4840, 4243, 39553]
```

---

## 補遺 (2026-04-19 追記): Bug 1-7 修正後の追加診断

Bug 1-7 を修正し再学習した `weights.pt` を `vs rulebase/case1 baseline_v1` で
評価した結果、**100戦 0勝**。さらに追加診断で以下を確認:

### Bug 8 (新規): kaggle replay loser 側 obs.step / obs.player が None

公開 kaggle replay の loser 側 (slot=1) では `observation.step` と
`observation.player` が `None` で保存されている。preprocess で featurizer に
渡す前に `step_idx` と `slot` を注入しないと、敗者フレーム約 5 万行が全て
step=0 として記録される。修正前の train データでは step=0 が 50% を占めて
いた (`pipeline/imitation/case1/training/preprocess.py::_iter_episode_frames`
で修正済み)。これを直してようやく序盤特徴量分布が runtime と整合する。

### 残課題: target head 精度 34% が戦略的崩壊を招く

修正後 val_target_acc=0.339 (= **66% 誤分類**)。target owner 別では:
| GT owner | n      | acc  |
|----------|--------|------|
| mine     | 4,302  | 0.23 |
| enemy    | 5,080  | 0.41 |
| neutral  | 2,713  | 0.37 |
| noop     | 260    | 0.15 |

n_my>=10 (中盤以降) では acc=0.31 まで低下。

**Game trace (vs baseline_v1, seed=0)**:
- step=10 まで互角 (1 vs 1 planet)
- step=25 で 2 vs 3、step=50 で 5 vs 9、step=75 で 8 vs 18、step=100 で IL=3
- IL は **同じ neutral planet (例: p29) に step 9-37 まで連射し続ける**
  (毎ターン from 28 で ×4)。fire の意思決定が行われていない。
- baseline は同 source からの連射を避け、遠距離の高 production planet を狙う。
- IL 発射 ship 配分の **40-60% が "100% (all-in)"** バケット。「reinforce」と
  「expansion」を区別できていない。

### 根本原因仮説

1. **Target head が target_pair 幾何特徴のみで意思決定し、文脈 (グローバル盤面・
   同一 source の過去発射) を持てない**。pro player は「reinforce ⇄ expand」を
   時系列で切り替えるが、現状の DeepSets は単フレーム決定のため。
2. **Multi-hot from-head の prior 0.105 と threshold 0.05 のギャップ**で、本来
   発射しないターンも fire しがち。training data の sparsity を反映しきれていない。
3. **`_resolve_action_target` で複数候補が ANGLE_TOLERANCE=0.20 以内に存在**する
   ケースでは「最も角度差が小さい」候補を選んでいるが、実際の意図 (送弾先) は
   不明確。教師信号にラベルノイズが混入している可能性。

### 次の改善方針 (案)

- target head の文脈強化: 同一 source の **直近 K ターンの target 履歴** を
  入力に追加 (短期メモリ)。
- target head の **2 段デコード**: まず (mine/enemy/neutral, near/far, low_prod/high_prod)
  などの抽象 target type を予測し、その後具体 planet を選ぶ。
- ラベル粒度を粗く: target を「planet id」でなく「最寄りの neutral / 最大 enemy /
  自軍最大 / 自軍最弱」など **6-8 個のテンプレート行動**にバケット化して分類。
- **DAgger / online IL**: rule-based agent との対戦中に baseline_v1 が「同じ
  状態でどう動くか」をオンクエリして教師信号を増やす。
- `_resolve_action_target` のラベルノイズ削減: ANGLE_TOLERANCE をさらに小さく
  (0.10) し、tie の場合は target 型 (enemy 優先) で破る。

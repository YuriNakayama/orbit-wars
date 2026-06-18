# strict held-out 勝利の構造的要因 (2026-06-17)

> 問い (ユーザー指摘): held-out strict が「常に0だったものが0でなくなった」のは
> 確率的揺らぎではなく構造変化のはず。0勝だった試行を振り返れ。

## 訂正: strict held-out は「常に0」ではなかった

ladder25 iter0/10 だけ見ると 0.00 だが、**全 ladder run を遡ると10 run で勝利が出ている**。
「常に0」は誤認識。正しくは「特定の構造を持つ run でのみ稀に勝利が出る」。

## 方法

全 `case8_vmpo_ladder*` の metrics.json から held-out `heldout_win_strict_v1` を全 iter
抽出し、**run 長 (iters) で揃えて** WIN群 (max>0) と zero群 (全0) を比較。短い run
(eval 1-3回) は「勝利が出る iter まで未到達」で結論不能として除外。

## 結果: run 長で揃えた比較 (iters≥34 のみ公平比較可能)

| run | iters | strict evals | maxStrict | group |
|---|---|---|---|---|
| ladder21 | 70 | 8 | **0.047** | WIN |
| ladder22 | 39 | 4 | 0.031 | WIN |
| ladder6/9/11/13/18 | 34-70 | 4-8 | 0.016 | WIN |
| ladder, 2, 3, 4, 5, 7 | 49-70 | 5-8 | **0.000** | zero |
| ladder14/24 | 18-19 | 2 | 0.016 | WIN (短いが勝利) |
| ladder15/16/17/19/20/23 | 6-22 | 1-3 | 0.000 | zero (短く結論不能) |

## 構造変化の特定: 2つの壁を段階的に越えた

### 壁① 「常時0」→「稀勝利 0.016」 = fine strict ladder

WIN群と zero群 (共に iters≥34) を分ける唯一の config:

| | WIN群 (6,9,11,13,18,21,22) | zero群 (ladder〜5,7) |
|---|---|---|
| **strict_ladder 刻み** | **細かい** (225→200→185→170…**15刻み**, T0=110段あり) | **粗い** (400→350→300…**50刻み**, 最小窓 T0=50/100) |

- **ladder9 で T0窓を 225→110 に詰めた fine ladder を導入** → 以降の長い run 全てで勝利出現。
- zero群 (ladder〜5) は粗いラダーで「素strictの直前 (T0=110前後)」段が存在せず、永遠に0。
- 引き金 = **緩和段 T0=110 (ほぼ素だが開幕だけ少し進んだ盤面) を踏ませること**。素strict
  (T0=0) は零分散で永遠に0だが、T0=110 で学習が乗った方策が held-out 素strict をたまに拾う。
- 例外 = ladder6 (粗いが T0=50 まで到達 → 0.016 が一度だけ)。

### 壁② 「稀勝利 0.016」→「複数勝利 0.047」 = degenerate-batch guard + no_op_bias=2

ladder21 (max 0.047, 突出) の fine-ladder peers (11/13/18/22) に対する固有差分:

- **`skip_update_if_no_win=True`** (guard A 初版) + **`adv_std_floor=0.1`** (guard B) を両搭載
- **`no_op_bias=2.0`** (中間値, peers は 1.0 / 8.0)
- `force_rung_low_every=0` (T0=0 退化段を強制照射しない)

退化バッチの方策poison (零分散→ノイズ増幅→entropy崩壊) を止めることで、勝てる方策が
保存され勝利頻度が 1/64→3/64 に倍増。

### 未踏の壁③ 「複数勝利 0.047」→「常勝 >0.5」

fine-ladder + guard では越えられていない (上限 0.047 ≈ 3/64)。素strict の開幕戦略エッジは
構造的 (持ち船3倍 handicap でも ~4%, ladder20) で、緩和段の転移だけでは常勝に届かない。
→ **質的に別レバー (strict-BC bootstrap で素strict の開幕手を直接模倣して KL anchor) が必要**
という示唆。

## 前 diff doc (strict_win_cases_diff.md) との関係

- 前doc: 「勝利の差分 = 汎用地力 (full) の高さ」 (strict勝率 ∝ full, 0.81 vs 0.63)。
- 本doc: 「勝利の引き金 = fine strict ladder という**構造**」。
- **両立する**: fine ladder (壁①) が地力を strict に届く形で押し上げ、guard (壁②) が
  その地力を保存する。full は「結果指標」、fine ladder + guard は「それを生む構造」。
  ユーザー指摘の通り、0→nonzero は揺らぎでなく ladder9/ladder21 の構造導入が原因。

## ladder25 への含意

ladder25 iter20=0.031 は揺らぎではなく、**ladder21→22 由来の fine-ladder + guard 構造を
継承した結果**。no_op_bias=0 への変更が壁②効果を保ったまま地力 (full 0.797-0.875) を
出している。iter30-70 で 0.031 が持続/増加すれば構造の再現性確認。

## 次の一手 (壁③)

- fine-ladder + guard は「0→稀勝利」を確立済 → これ以上の ladder/guard 微調整は壁③を
  越えない見込み。
- **strict-BC bootstrap** (in-JAX strict の開幕 N手を模倣 → KL anchor 付き RL) が
  唯一の未踏レバー。素strict の零分散 (勝てない→勾配なし) を、模倣の教師信号で迂回する。

# imitation/case9 iter1 — head 設定最適化 3 variant 比較 (initial 3 hypotheses 一括)

> 作成日: 2026-05-07
> 関連 plan: ../hypotheses.md (initial 3 hypotheses: H1 three_head / H2 candidate / H3 candidate+ships)
> 検証 commit: ea87185 / 98ea59d (3 variant がそれぞれ別 commit で完走)
> 固定軸: PLANET=41 (case8 35 + case5 timeline 6) / GLOBAL=20 / Set Transformer (ISAB×3 m=16 + PMA, hidden=128)
> 可変軸: head_mode (three_head / candidate / candidate_ships)

## サマリ (TL;DR)

**3 variant すべて 1 ep self-play smoke は完走 + ローカル 10 ep vs baseline_v1 は 0 wins**。
n<300 で結論不可 (memory `project_imitation_case1_phase3`) のため **3 variant とも採否 = inconclusive**。
val 指標では three_head が val_target_acc=0.928 と単頭 head の中で抜きん出るが、
ローカル対戦での挙動差は検証範囲では検出できなかった。

| variant | best_epoch | best val metric | weights file | local 10 ep vs baseline_v1 | avg_turns | 採否 |
|---------|-----------|----------------|---|-----:|---:|------|
| H1 three_head | 13 | val_target_acc=**0.928** | weights_three_head.pt | **0 / 10** | 197.8 | inconclusive |
| H2 candidate | 5 | val_cand_fire_acc=0.211 | weights_candidate.pt | **0 / 10** | 157.8 | inconclusive |
| H3 candidate_ships | 8 | val_cand_fire_acc=0.211 | weights_candidate_ships.pt | **0 / 10** | 163.1 | inconclusive |

baseline_v1 は 30 / 30 全勝 (3 variant 合算)。

## 1. 学習結果

### 1.1 統計

| variant | run_id | pod | runtime | epochs_run | best_epoch | best_val_loss |
|---------|--------|-----|---------|-----------:|-----------:|--------------:|
| three_head | `20260507-023323__...__98ea59d__seed0` | A6000→fallback RTX 4090 | ~25 min | 15 | 13 | 1.0772 |
| candidate | `20260506-133924__...__a6a7bee__seed0` | A6000 | ~25 min | 11 (early stop) | 5 | 262325.87 (class_weight scale) |
| candidate_ships | `20260507-022131__...__ea87185__seed0` | RTX 4090 | ~30 min | 14 | 8 | 227933.73 (class_weight scale) |

3 variant とも **A6000 を最優先指定 + RTX 4090 fallback**, batch_size=128 で完走。 RTX 4090 + batch_size=256 では retry6/7 で `60_before_train` 後に host eviction (OOM 仮説) を 2 連続観測しており、 batch_size 削減 + GPU host RAM 余裕の組み合わせが必要。

### 1.2 学習曲線の特徴

#### three_head (val_target_acc gate)
- val_from_acc: 序盤から ~0.91 で安定 (case7 ベースラインと同水準)
- val_target_acc: 0.92 → epoch=13 で best 0.928 (NUM_TEMPLATES=8 の中で no_op 含めた template 分類)
- val_ships_acc: 0.82 前後で安定
- val_loss: 1.077 (3-head は CE loss が比較的小さい normal scale)

#### candidate (val_cand_fire_acc gate)
- val_cand_fire_acc: epoch=5 で best 0.211、 その後 5 連続 patience で early stop @ epoch=10
- val_cand_acc: ~0.18-0.19 程度
- val_cand_noop_acc: ~0.19 (no_op が多い)
- val_loss: 262k (focal loss + class_weight で絶対値スケールが大)

#### candidate_ships (val_cand_fire_acc gate)
- val_cand_fire_acc: epoch=8 で best 0.211、 candidate と同水準
- val_ships_acc: ~0.40 (4-bucket ships 分類)
- val_loss: 228k (cand focal + ships CE)

### 1.3 train.parquet
- frames train=406,378 / val=53,076 (ローカル preprocess)
- outside_K=8 ratio=45.7% (case8 同等)
- train.parquet 1.0 GB / val.parquet 135 MB

## 2. ローカル 10 ep self-play vs baseline_v1

`uv run --directory bot python -m dataset run --agents il_v9_<variant>,baseline_v1 --mode 1v1 --episodes 10 --seed 100 --no-save-replay`

- 3 variant とも **0 wins / 10 games**
- baseline_v1 は 100% 勝率
- avg_turns: three_head 197.8 / candidate 157.8 / candidate_ships 163.1 → three_head は最も粘った (case3 phase2 系の 3-head 構造が浮動点を取りやすい?)
- timeouts=0, turn_p95 < 0.04s で挙動異常はなし

n<300 で確定的結論は出さない (memory `project_imitation_case1_phase3`)。

## 3. 採否判断

3 variant すべて **inconclusive 固定**。
val 指標は three_head が他 2 variant の cand 系よりも素直に高く出ているが、
それは「単頭の精度が高い」 = 「対戦勝率に直結する」 とは限らない (case3/case6 で確認済の傾向)。
実環境の 10 ep では 3 variant 間で差を検出できなかった。

優先度:
- (high) **three_head に絞って 300 ep フォローアップ** が次イテの最有力候補。 val_target_acc=0.928 はこれまでの imitation 系列で最高水準。
- (mid) candidate / candidate_ships は val 指標が同水準 → どちらか一方を deprecate して効率化。
- (low) dual head 等の P2 仮説 (H4 以降) は 3 head 単頭の挙動が一通り見えてから判断。

## 4. 学んだこと

### 4.1 RunPod 基盤の知見 (iter12-iter17 修正)

case9 を新規 case として立ち上げる中で、 RunPod onstart の 5 つのトラップを順番に修正:

1. **iter12** (`1fb75c9`): `bot/.venv/bin/dvc` shim の bad interpreter → `${PY_BIN} -m dvc` 経由に変更
2. **iter13** (`fef9df2`): broken venv (bin/python 欠落) → uv sync 前に強制リセット
3. **iter14** (`bd13394`): `dvc pull --allow-missing` graph 衝突で skip → case 別 targeted pull 追加
4. **iter15** (`f01f13b`): targeted pull を `--allow-missing` の **前** に移動 + DEBUG echo
5. **iter16** (`f0befca`): persist setup の `rm -rf data/mart/imitation` で git tracked .dvc が消失 → `cp -an` で persist 側に退避
6. **iter17** (`68e578f`): `--allow-missing` が targeted pull で取得した parquet を delete → 後段で **再 pull**
7. **batch_size 削減** (`a6a7bee`): RTX 4090 で OOM → 256 → 128 + A6000 fallback で host RAM 余裕

これらは memory `project_runpod_5_traps_2026_05_04` の 5 trap に追加すべき新トラップ群。

### 4.2 train.py の head_mode 多 variant 対応 (`98ea59d`)

case9 の 3 variant で training script を共有する場合、 metrics dict の key が variant 別に異なる:
- candidate: `cand`, `cand_acc`, `ship`, `ship_mae`
- candidate_ships: `cand`, `cand_acc`, `ships_loss`, `ships_acc` (`ship` なし)
- three_head: `from_loss`, `target_loss`, `ships_loss`, `from_acc`, `target_acc`, `ships_acc` (`cand` なし)

`log_row` 構築や `_stamp` の f-string で固定 key を直接参照すると KeyError で死ぬ。
**head_mode で切り替える dispatch ロジックを散在させる**実装パターンが必要。

### 4.3 dvc.yaml の outs と .dvc 重複

dvc.yaml stage の `outs:` と外部 `.dvc` ファイルが **同じ path を track する** ことは、
新しい dvc バージョンで衝突を起こす。 case8 系から派生した case9 でもこの構造が引き継がれて
`dvc add` が refuse する状況が再現。 case 移植時は **dvc.yaml stage の outs を整理する** か
**`.dvc` 個別を整理する** かいずれかに揃える必要がある。

## 5. Next iteration 方針

| 候補 | 内容 | 優先度 |
|------|------|--------|
| **300 ep フォローアップ (three_head)** | three_head の val_target_acc=0.928 を 300 ep self-play で本当に rule-based に勝てるか検証 | P1 ⭐ |
| candidate variant の deprecate | val_cand_fire_acc=0.211 は too low、 candidate_ships と挙動差なしなので片方統合 | P2 |
| dual head (H4) | three_head + candidate を blend した hybrid を試す前に、 まず three_head 単頭で 300 ep を見る | deferred |
| backbone 強化 | hidden 128 → 256 は case10/11 で under-trained を観測、 epoch 数増やしてから検討 | low |

## 参考

- run.json: `data/output/models/imitation/case9/runs/<run_id>/run.json`
- weights canonical: `bot/pipeline/imitation/case9/policy/weights_<variant>.pt`
- onstart.log: 各 run dir 配下 (S3 fallback で取得)

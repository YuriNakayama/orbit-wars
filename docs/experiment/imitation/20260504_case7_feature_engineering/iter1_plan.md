# imitation/case7 — Feature Engineering 拡張 (予測距離 / history / ship 発射)

> 作成日: 2026-05-04
> 関連:
> - `bot/pipeline/imitation/case5/policy/featurizer.py` — 起点 (PLANET_FEAT_DIM=17, GLOBAL_FEAT_DIM=6, ship-prediction 6 列追加済み)
> - `bot/pipeline/imitation/case5/policy/timeline.py` — `simulate_planet_timeline` (case5 内コピー、case7 でも踏襲)
> - `docs/experiment/imitation/20260427_case3_feature_engineering_phase2/result.md` — history 列のリーク事例 (`obs_{N-2}` 参照で fix 済み)
> - `docs/experiment/imitation/20260503_case5_ship_prediction/plan.md` — case5 の ship-prediction 取り込み plan
> - memory `project_imitation_case1_phase3` — n<300 self-play は信頼不可 (採否は 300 ep フォローアップ)
> - memory `project_runpod_onstart_pitfalls` — RunPod onstart 3 trap (mountpoint-q / dvc --allow-missing / cwd-relative config)
>
> スコープ: imitation/case7 を新設 (case5 がベース、case6 は他ブランチで使用中) し、case5 featurizer に **予測距離 / history / 敵 ship 発射 / 自軍 ship 発射** の 4 カテゴリを追加する。

## 仮説 (Hypothesis)

case5 featurizer は ship-prediction (per-planet timeline 6 列) を獲得したが、依然として
**(a) 空間的将来位置 (orbit する planet の future position 距離)**、
**(b) 状態の時間変化 (planet の delta_ships / owner_changed)**、
**(c) fleet event (敵/自軍の ship 発射 history)**
の 3 つの認知が欠けている。これらを per-planet × global の両軸で補強すれば、
**defensive hold / harass 開始 / target 選択 / reinforcement 配分** いずれの判断も改善し、
target/ships head の F1 と 1v1 vs `baseline_v1` 勝率の両方を底上げできる。

メカニズム:

- **予測距離**: aim_with_prediction が runtime で利用している future position をモデル側にも与えれば、policy は orbit 周期を考慮した「今 src→tgt に投げると着弾時点で何処にあるか」の距離評価ができ、target template の選好が一段引き上がる。
- **history (delta_ships, owner_changed)**: case3 phase2 で実証済み — ships head の bucket 0/1/2 recall を +0.17〜+0.26 改善した。case5 では未投入。
- **敵 ship 発射 (per-planet 集計)**: 敵の出撃源 / 出撃量 / 出撃方向 (ターゲット planet) を per-planet にビン化することで return-fire 判断が安定する。
- **自軍 ship 発射 (history)**: 直近 N ターンの自分の発射回数 / ships を global に持たせると、policy が自分の build-up / harass cycle を一貫させやすくなる (case3 phase2 では global 4 列で投入済み、リークは無し)。

成功指標:

- **Stage 1 (validation 精度)**: case5 baseline 比で from PR-AUC / target macro-F1 / ships macro-F1 のいずれかが +0.01 以上改善する → Stage 2 へ進む
- **Stage 2 (self-play sanity, 50 ep)**: vs `baseline_v1` で **学習時精度の向上が反映されているか** を 50 ep で sanity check (ユーザー指定スコープ)
- **採否判断は 300 ep self-play で別途**: memory `project_imitation_case1_phase3` の方針に従い、最終採否は 300 ep フォローアップで判定 (本 plan の最小スコープ外)

評価メトリクス:
- validation: `bot/pipeline/imitation/case7/evaluation/eval_metrics.py` (case3 phase2 形式の per-head metrics)
- self-play: ローカル `vs baseline_v1` 50 ep (Stage 2)、+ 採否ゲート用 300 ep (フォローアップ)
- **Kaggle publicScore は使わない** (`.claude/rules/bot/pipeline.md` 準拠)

## 既存コードの現状 (from Step 1)

- **case5 featurizer**: `bot/pipeline/imitation/case5/policy/featurizer.py` — 17 列 / planet (case1 base 11 + ship-prediction 6) + 6 列 / global。`incoming[slot]` で per-planet incoming ships を集計、`nearest_eta[slot]` で最早到達 ETA を持つが、**fleet 個別の出撃源 / 出撃方向 / past launch 履歴は未保持**。orbit-aware future position は `aim_with_prediction` 内 (`policy/geometry.py`) で計算するが、featurizer には流れ込んでいない。
- **case3 phase2 history 知見**: `bot/pipeline/imitation/case3/policy/featurizer_phase2.py` — `HistoryState.prev_planet_snapshots.maxlen=3` で `obs_{N-2}` / `obs_{N-3}` を参照し、planet 列に `delta_ships_t1` / `delta_ships_t2` / `owner_changed_t1` を追加、global 列に `enemy_launch_count_last4` 等 4 列を追加。**`obs_{N-1}` 参照は action_N との因果リークを起こすため禁止** (result.md L36-71 参照)。
- **AGENT_REGISTRY**: `bot/src/dataset/selfplay/agents.py` に `il_v1` (case1) / `il_v2` (case2) / `il_v3` (case3 phase2) / `il_v4` (case4) / `il_v5` (case5) が登録済み。case7 では `il_v7` を新規登録 (case6 の番号は他ブランチ用に予約)。
- **DVC stages**: `dvc.yaml` に `preprocess_imitation_case<N> / train_imitation_case<N> / eval_imitation_case<N>` が case ごとに並ぶ。case7 用に同形 3 stage を追加。

## スコープ (Scope)

### 変更ファイル

| Path | 変更内容 |
|------|----------|
| `bot/pipeline/imitation/case7/` (新規) | `cp -r case5 case7` で初期化 → 全 import path / module 名を case7 に書き換え |
| `bot/pipeline/imitation/case7/policy/featurizer.py` | PLANET_FEAT_DIM 17 → **24 (+7)**、GLOBAL_FEAT_DIM 6 → **10 (+4)**。新列は下記 "Feature catalogue" 参照 |
| `bot/pipeline/imitation/case7/policy/agent.py` | `HistoryState` を case3 phase2 同等に追加 (`prev_planet_snapshots: deque[maxlen=3]`、`prev_fleets: deque[maxlen=4]`)。featurizer.py が要求する past obs を持ち回す |
| `bot/pipeline/imitation/case7/training/preprocess.py` | replay の per-step 列展開時に `obs_{N-2}` / `obs_{N-3}` / `prev_fleets_{N-1..N-4}` を参照できるよう順次走査。新規列を parquet schema に追加 |
| `bot/pipeline/imitation/case7/training/dataset.py` | input_dim 更新 (planet 24 / global 10) |
| `bot/pipeline/imitation/case7/training/train.py` | input_dim 更新のみ |
| `bot/pipeline/imitation/case7/evaluation/` | `eval_vs_baseline.py` (50 ep self-play) + `eval_metrics.py` (per-head val 精度) |
| `bot/src/dataset/selfplay/agents.py` | `"il_v7": "pipeline.imitation.case7.policy.agent:agent"` を追加 |
| `dvc.yaml` | `preprocess_imitation_case7 / train_imitation_case7 / eval_imitation_case7` stage を追加 |
| `bot/src/runpod_io/cli.py` の `CASE_DEFAULTS` | `case7` entry 追加 |
| `params.yaml` | `imitation.case7.*` ブロック (case5 設定を流用 + input_dim 更新のみ) |
| `tests/pipeline/imitation/case7/` | `test_featurizer_history.py` / `test_featurizer_predicted_distance.py` / `test_history_no_leak.py` (case3 phase2 のリーク回帰防止 — `obs_{N-1}` 経由の delta_ships が action と完全相関しないことを確認) |

### Feature catalogue (追加予定)

#### planet 列 (17 → 24, +7)

| idx | 名前 | 定義 | カテゴリ |
|----|------|------|----------|
| 17 | `future_dist_to_my_centroid` | (`aim_with_prediction` の future position に基づく自軍重心への将来距離) / BOARD_SIZE | 予測距離 |
| 18 | `future_dist_to_enemy_centroid` | 敵重心への将来距離 / BOARD_SIZE | 予測距離 |
| 19 | `delta_ships_t1` | (ships_now − ships_{N−2}) / max(1, ships_now), clip ±1 | history |
| 20 | `delta_ships_t2` | (ships_now − ships_{N−3}) / max(1, ships_now), clip ±1 | history |
| 21 | `owner_changed_t1` | owner が N−2 から変わったか (0/1) | history |
| 22 | `enemy_targeted_count_last4` | 直近 4 ターンに **この planet を target** とする敵 fleet が発射された回数 / 5 | 敵 ship 発射 |
| 23 | `enemy_targeted_ships_last4` | 同 ships 合計 (log1p / 6) | 敵 ship 発射 |

#### global 列 (6 → 10, +4)

| idx | 名前 | 定義 | カテゴリ |
|----|------|------|----------|
| 6 | `enemy_launch_count_last4` | 直近 4 ターンの敵発射回数 / 10 | 敵 ship 発射 (case3 phase2 流用) |
| 7 | `enemy_launch_ships_last4` | 同 ships 合計 (log1p / 6) | 敵 ship 発射 |
| 8 | `ally_launch_count_last4` | 直近 4 ターンの自軍発射回数 / 10 | 自軍 ship 発射 |
| 9 | `ally_launch_ships_last4` | 同 ships 合計 (log1p / 6) | 自軍 ship 発射 |

#### 設計原則

- **history 列は `obs_{N-2}` / `obs_{N-3}` 参照のみ**。`obs_{N-1}` は禁止 (case3 phase2 リーク事例)。
- **launch event は per-fleet snapshot の差分**: `prev_fleets_{N-1}` には存在しない fleet で `prev_fleets_{N}` に存在する fleet を「N に発射された fleet」とみなす。これは action_N の **結果** を含むので supervision に対して **action 後** の情報。`obs_{N-1}` 参照と同じく *因果先行性* に注意 — global launch history は `obs_{N-2}` の fleet 差分から計算する形で揃える。
- **future_dist_to_*centroid** は orbit angular_velocity を使った 5 ターン先位置を採用。`aim_with_prediction` と同じ `geometry.py` を再利用し、featurizer から呼ぶ。

### 変更なし

- case1 / case2 / case3 / case4 / case5 — 触らない (case 独立性ルールに従う、`.claude/rules/bot/pipeline.md`)
- case6 (他ブランチで予約) — 触らない
- rulebase 全 case — 触らない (cross-case import 禁止に従い、必要な共通コードは case7 内にコピー)
- Submit shape (case1 が canonical なので影響なし)

## 実装ステップ

1. **case7 ディレクトリの初期化** — `cp -r bot/pipeline/imitation/case5 bot/pipeline/imitation/case7` → `__init__.py` 等の絶対 import path / module docstring を case7 へ書き換え。
2. **featurizer.py 拡張** — `PLANET_FEAT_DIM 17→24`、`GLOBAL_FEAT_DIM 6→10`、上記 catalogue の列を実装。`HistoryState` 引数を取り、`obs_{N-2}` / `obs_{N-3}` / `prev_fleets_{N-1..N-4}` を参照する関数シグネチャに変更。
3. **agent.py 拡張** — `HistoryState` の追加 (case3 phase2 と同形)、per-match ring buffer。`featurize` 呼び出し直前に history を更新。
4. **preprocess.py 拡張** — replay 走査時に per-episode で `prev_planet_snapshots` / `prev_fleets` を再現。**Kaggle replay の loser 側 `obs.step` / `obs.player` が None になる問題は既存 case5 と同じく注入で対応** (memory `project_kaggle_replay_loser_obs`)。
5. **unit test 追加** —
   - `test_featurizer_predicted_distance.py`: orbit angular_velocity を変えたとき future_dist が単調に変わることを assert
   - `test_featurizer_history.py`: `obs_{N-2}` / `obs_{N-3}` 参照で `delta_ships_t1` が想定値になることを assert
   - `test_history_no_leak.py`: action と `delta_ships_t1` の Pearson 相関を計算し、|r| < 0.5 を assert (case3 phase2 では from_multihot とほぼ完全相関だった ※ 修正後)
6. **DVC stage 追加** — `dvc.yaml` に case7 三段 (preprocess / train / eval) を case5 と同形でコピー。`params.yaml` に `imitation.case7.*` ブロックを追加 (input_dim のみ差替え)。
7. **AGENT_REGISTRY 登録** — `il_v7` を `bot/src/dataset/selfplay/agents.py` に追加。
8. **ローカル smoke 検証** — `dev/test-bot` (format / lint / mypy / pytest) → `uv run --directory bot pytest tests/pipeline/imitation/case7 -x` を pass させる。
9. **commit & push** (この plan ではここまで; 以降は `experiment-execution` skill のスコープ)。
10. **(execution skill 側) RunPod Step A** — `dev/runpod train --case case7 --stage preprocess_imitation_case7` でデータ前処理を smoke (≤$0.2)。memory `project_runpod_onstart_pitfalls` の 3 trap (mountpoint-q / dvc pull --allow-missing / cwd-relative config) を pre-run 確認。
11. **(execution skill 側) RunPod Step B** — `dev/runpod train --case case7` で本訓練 (~$1.0)。完了後 `dev/runpod pull <run_id> --case case7`。
12. **(execution skill 側) Stage 1 評価** — `uv run --directory bot python -m pipeline.imitation.case7.evaluation.eval_metrics` で per-head val 精度。case5 比 +0.01 以上ならば Stage 2 へ。
13. **(execution skill 側) Stage 2 評価** — `vs baseline_v1` 50 ep self-play (sanity check)。
14. **(execution skill 側) 結果集約 → result.md** — Stage 1/2 数値、ablation の必要性 (リーク疑い検出時) を記録。
15. **(フォローアップ) 300 ep self-play による採否最終判定** — memory `project_imitation_case1_phase3` 準拠、別 plan/result サイクルで実施。

## 検証方法 (Validation method)

### ローカル

```bash
# 全テスト + lint
dev/test-bot

# case7 のみ
uv run --directory bot pytest tests/pipeline/imitation/case7 -x

# featurizer の単体動作確認
uv run --directory bot python -c "from pipeline.imitation.case7.policy.featurizer import featurize; print(featurize.__doc__)"
```

### リモート (execution skill 側のスコープ)

```bash
# Step A: preprocess smoke (≤$0.2)
git push origin feature/feature-engineering
dev/runpod train <commit-sha> --case case7 --stage preprocess_imitation_case7 --watch

# Step B: 本訓練 (~$1.0)
dev/runpod train <commit-sha> --case case7 --watch
dev/runpod pull <run_id> --case case7
```

### 評価

- **Stage 1 (validation)**: case5 baseline (`weights.pt` 既存) と `weights_case7.pt` で `pipeline.imitation.case7.evaluation.eval_metrics` を比較。primary は **from PR-AUC + target macro-F1 + ships macro-F1**。case3 phase2 と同 metric 体系。
- **Stage 2 (self-play sanity, 50 ep)**: `pipeline.imitation.case7.evaluation.eval_vs_baseline --episodes 50 --seed 0` で vs `baseline_v1`。**この 50 ep は採否ではなく、Stage 1 で見えた精度向上が試合に反映されているかの sanity check**。
- **採否ゲート (フォローアップ)**: 300 ep を別途実施。+5pp 以上で採用、+0〜5pp は保留、マイナスは破棄 (memory `project_imitation_case1_phase3`)。

### リーク回帰防止

case3 phase2 で発生した history 列の causal leak は本実験でも最大の risk。下記 ablation を Stage 1 直後に必須実施:

- `no_history`: planet 19/20/21 + global 6-9 を 0 で埋めた重みで eval
- `no_planet_history`: planet 19/20/21 だけ 0
- `no_global_launch`: global 6-9 だけ 0
- `no_predicted_distance`: planet 17/18 だけ 0

これらの間で from PR-AUC が異常 (e.g. >0.9) に出た variant がある場合、リーク。具体的には Phase 2 result.md の table と同じ手順を使う。

## 参考 (References)

- [Kaggle Lux AI S3 (`Lux-AI-Challenge/Lux-Design-S3`)](https://github.com/Lux-AI-Challenge/Lux-Design-S3) — partial observability + per-unit feature stack の設計が直接の類縁。imitation learning ベースの上位解は CNN+ResNet (squeeze-excitation 128ch 5x5) で per-tile observation を畳み込んでいる。Orbit Wars は per-planet なので CNN ではなく DeepSets だが、**per-unit (planet/fleet) ごとの spatial + temporal stacking** という上位設計は同じ。
- [Lux AI with Imitation Learning (Kaggle, shoheiazuma)](https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning) — リソース計算 + history stacking で BC 上位に入った 2021 例 (404 になっていたため詳細未取得、概要のみ参考)。
- 既存リポジトリ内: `docs/experiment/imitation/20260427_case3_feature_engineering_phase2/result.md` が最も近い前例。**history 列の causal leak は obs_{N-1} 参照で必ず発生する**ことが既知 → 本 plan も `obs_{N-2}` 参照に統一。
- 既存リポジトリ内: `docs/experiment/imitation/20260503_case5_ship_prediction/plan.md` — case5 の ship-prediction 取り込みと同じ路線で、case7 は cp -r 起点で書き換える。

## リスク / 想定失敗モード

1. **history causal leak の再発** — `obs_{N-2}` 参照を徹底し、`test_history_no_leak.py` で Pearson 相関閾値を CI で守る。Stage 1 直後の ablation table を必須化。
2. **future position 計算コスト** — `aim_with_prediction` を per-planet × per-other-planet で呼ぶと O(N²) になり 36 planet で 1296 calls/turn。featurizer 全体は preprocess 時に一度だけ走るので訓練側は問題なしだが、agent runtime (1 sec budget) で問題が出る場合は per-fleet ではなく per-(my_centroid, enemy_centroid) の 2 reference のみに削減 (現案は既にこの形)。
3. **per-fleet 集計のスケール** — 直近 4 ターンの fleet 差分で発射履歴を計算するが、4 ターン分の `prev_fleets` を保持するメモリは fleet 数 × 7 列 × 4 ≒ 数 KB と無視できる。
4. **Kaggle replay loser 側 obs の step / player None** — 既存 case5 と同じく preprocess で注入 (memory `project_kaggle_replay_loser_obs`)。新規列にも同じ注入が必要。
5. **RunPod onstart の 3 trap** — Step A 起動前に必ず確認: mountpoint-q / dvc pull --allow-missing / cwd-relative config (memory `project_runpod_onstart_pitfalls`)。

## Stop conditions

以下を満たしたら本 plan のスコープは完了:

- [ ] case7 ディレクトリが case5 起点で作成され、unit test が pass している
- [ ] `dev/test-bot` が green
- [ ] `dvc.yaml` / `params.yaml` / `agents.py` / `runpod_io.cli` に case7 が登録された
- [ ] commit & push まで完了

実装後の RunPod 学習 / Stage 1/2 評価 / 結果記録は `experiment-execution` skill 側の責務。

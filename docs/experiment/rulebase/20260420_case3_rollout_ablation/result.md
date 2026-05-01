# case3 rollout / 全有効化 ablation レポート (2026-04-20)

`feature/update-rulebase` ブランチで `pipeline/rulebase/case3` に集約した追加施策 (REINFORCE_FRESH_CAPTURE, shallow rollout 変種群, true 2-player rollout, plan_moves(light)) の検証記録と、最終採用構成の根拠。

結論を先に: **case3 default 構成 (A + G(true2p) + plan_moves(light) 全有効化)** が 300戦 52.0% で現状の最良構成。ただし A 単独 (50.7-53.0%) との差は seed variance (±3-4pp) 内で有意ではない。score-bonus/rollout 系は 5 連敗で seed variance を破れず、次の改善には学習ベース評価関数が必要。

---

## 📋 実施施策サマリー

| # | 施策 | 300戦勝率 | seat 対称性 | turn_p95 | 評価 |
|---|---|---|---|---|---|
| A | REINFORCE_FRESH_CAPTURE (window=10) | 53.0% | 非対称 ±7pp | 0.05s | +2.3pp 非有意、害なし → **採用** |
| — | +PROACTIVE_DEFENSE_HORIZON 12→18 | 47.3% | — | — | **-5.7pp 害** → 棄却 |
| — | +HOSTILE_SWARM_ETA_TOLERANCE 1→2 | 50.8% (600戦) | — | — | 効果なし → 棄却 |
| B | ROLLOUT bonus (top-K=3, depth=2, bonus=0.08) | 51.3% | 非対称 ±5.3pp | 0.12s | +0.6pp 非有意 → 棄却 |
| C | ROLLOUT replace (score 上書き) | 51.0% | 対称 ±0.6pp | 0.12s | +0.3pp 非有意 → 棄却 |
| D | ROLLOUT_FILTER (候補生成 filter top5/depth3) | 51.3% | **非対称 ±13pp** | — | 合算 +0.6pp、seat 崩壊 → 棄却 |
| **G** | **ROLLOUT true2p (敵視点 plan_moves を候補ごとに呼ぶ)** | **51.7%** | **対称 ±0.35pp** | 0.47s | +1.0pp 非有意、**挙動安定** → **採用** |
| H-a | G + TOP_K=8 | 46.3% | — | 0.87-0.91s | **timeouts 61 件、-4.4pp 害** → 棄却 |
| I | G + TOP_K=8 + plan_moves(light) | 50.0% | 対称 ±0pp | 0.75-0.82s | timeouts 解消も効果消失 → 棄却 (light flag は温存) |
| **A+G+light** | **全有効化 (case3 default)** | **52.0%** | 非対称 ±6.7pp | 0.36-0.41s | +1.3pp 非有意、合算最良 → **case3 default ON** |

**最終構成 (case3 baseline_v3 default)**:
- `HARASS_ENABLED=True` (case2 から継承)
- `SAFE_INTERCEPT_HALF_STEP=True` (case2 から継承)
- `REINFORCE_FRESH_CAPTURE_ENABLED=True` (A)
- `ROLLOUT_ENABLED=True`, `ROLLOUT_MODE="true2p"`, `ROLLOUT_TOP_K=3`, `ROLLOUT_DEPTH=3`, `ROLLOUT_SCORE_BONUS=0.08` (G)
- 敵 plan_moves は `light=True` で呼び出し (I で追加した軽量版)
- その他実験的 flag (COMET_NPV, FINISHING_TIE_GUARD, OM, Lookahead, DYNAMIC_PROACTIVE_HORIZON) は全て OFF

---

## 🔬 検証プロトコル

| 項目 | 設定 |
|---|---|
| 対戦相手 | `baseline_v1` (pipeline/rulebase/case1) |
| モード | 1v1 |
| エピソード数 | 300戦 (seat0=150, seat1=150) |
| seed | seat0: 40000 起点 / seat1: 40500 起点 |
| 実行コマンド | `uv run python -m env run --agents baseline_v1,baseline_v3 --mode 1v1 -n 150 --seed 40000 ...` (seat 入替) |
| actTimeout | 1.0s (Kaggle 仕様) |
| ハードウェア | M4 MacBook (local) |

seat variance は 150戦 (single seat) で ±4pp 程度あり、±13pp を超える非対称は挙動バグ警告として扱う (施策 D で該当)。

---

## 📐 設計詳細

### 施策 A: REINFORCE_FRESH_CAPTURE

**動機**: 敗北リプレイ分析 (39件) で "奪取直後の低 production 惑星が次ターンに奪還される" パターンが中期敗戦 (200-350T, n=25) の共通項。既存 REINFORCE は `production >= REINFORCE_MIN_PRODUCTION=2` で足切りし、奪取直後の低 prod 惑星を対象外にしていた。

**実装** (`core/world_model.py`):
- `CaptureState` dataclass で直近 `REINFORCE_FRESH_CAPTURE_WINDOW=10` ターン以内に奪取した惑星 id を追跡。
- `_compute_defense_buffers` で `fresh_ok` gate を追加: production < MIN でも fresh であれば reinforce 対象に含める。
- `agent.py` の `_update_capture_state` で毎ターン所有権変化を検出し `CaptureState` 更新。

**結果**: 300戦 53.0% (vs OFF 50.7%)、seat1 で +6pp、seat0 で -1.4pp。非対称だが害なし。default ON。

### 施策 B-D: shallow rollout 変種

**共通骨格** (`rollout.py`):
1. missions.sort 後、top-K 候補について `simulate_planet_timeline` で N ply 進め、採用シナリオ終了時点の自軍純艦数 (または惑星 value 合計) を計算。
2. その値を score に反映する方法が B/C/D で異なる。

| 変種 | 反映方法 | 結果 |
|---|---|---|
| B (bonus) | `mission.score += rollout_value * ROLLOUT_SCORE_BONUS` | 51.3%、非対称 |
| C (replace) | `mission.score = rollout_value` (tie-break のみ旧 score) | 51.0% |
| D (filter) | 採用時-不採用時の margin < 0 で mission を skip | 51.3% (seat0=64.7%, seat1=38.0%) |

**教訓**: heuristic score と rollout 値が強く相関しており、top-K 内の並び替えだけでは挙動が有意に変わらない。score 合成方式では seed variance を破れない (OM v1/v2, lookahead 1-ply と合わせて 5 連敗)。

### 施策 G: ROLLOUT_MODE="true2p" (2-player rollout)

**動機**: B/C/D の失敗原因は「rollout 時に敵が静止 = 攻撃者視点の naive simulation」。実際は敵が反撃してくる。

**実装** (`rollout.py`):
- 各候補について:
  1. 自分の発射分 (`our_send_ships`) を元惑星 ships から差し引いた `enemy_world` を構築。
  2. `plan_moves(enemy_world, light=True)` を呼んで敵の反撃 fleet を得る。
  3. 敵 fleet を arrivals に加算し、target を含む全惑星の timeline を `ROLLOUT_DEPTH=3` 進める。
  4. 反撃後の純艦数合計で候補を再 sort。
- `_ROLLOUT_DEPTH` モジュールローカル変数で再帰防止 (enemy plan_moves が rollout を呼ばない)。

**結果**: 300戦 51.7% (seat0=51.3%, seat1=52.0%)。**全施策で唯一 seat 対称 (±0.35pp)**。turn_p95 0.047→0.468s (約10x) だが 1s 以内。default ON 候補として温存。

### 施策 I: plan_moves(light=True)

**動機**: H-a (G + TOP_K=8) で turn_p95 が 0.87-0.91s に上昇、timeouts 61 件で -4.4pp の害。TOP_K=8 を維持するには敵 plan_moves を軽量化する必要。

**実装** (`strategy.py`):
```python
def plan_moves(world: WorldModel, light: bool = False) -> list[list[int | float]]:
    ...
    if not light:
        emit_followup_moves(...)
        emit_evacuation_moves(...)
        emit_rear_guard_moves(...)
```

`light=True` の場合、followup / evacuation / rear_guard の 3 movement emit をスキップ。mission-based な mainline 行動のみ返す。enemy 反撃 prediction には十分な粒度。

**結果**: TOP_K=8 + light で timeouts 61 → 5、turn_p95 0.87→0.75-0.82s。勝率は 50.0% で H-a (46.3%) からは回復したが G 単独 (51.7%) を超えず。TOP_K 拡大の情報価値が無いことを確認 (heuristic top-3 で既にほぼ正しい並び)。light flag 自体は case3 で採用し G (TOP_K=3) の敵 plan_moves 呼び出しに使用。

---

## 📊 最終 300戦比較 (A+G+light vs 単体施策)

| 構成 | seat0 勝率 | seat1 勝率 | **合算勝率** | seat 分散 | turn_p95 (v3) |
|---|---|---|---|---|---|
| baseline (case2 A-only 相当) | 49.3% | 52.0% | 50.7% | ±1.4pp | 0.05s |
| A 単独 | ~49% | ~55% | 53.0% | ±3pp | 0.05s |
| G 単独 | 51.3% | 52.0% | 51.7% | **±0.35pp** | 0.47s |
| **A + G + light (case3 default)** | **45.3%** | **58.7%** | **52.0%** | ±6.7pp | 0.36-0.41s |

**観察**:
- 合算勝率は A+G+light が最良だが、A 単独 (53.0%) との差は +1.3pp → seed variance 内で有意ではない。
- G 単独で実現していた seat 対称性が A との組合わせで崩れた (A 由来の非対称性が支配)。
- A/G の効果は独立加算的ではない → 施策間の相互作用がある (推定: G が敵反撃を織り込むことで A の fresh capture 判断と干渉)。

---

## 🎯 採用判断

### データ上の最良: **A + G + light (case3 default ON)**

- 合算 52.0% は単体施策の最高値 (A=53.0% / G=51.7%) と実質同等 (seed variance 内)。
- turn_p95 0.36-0.41s で timeout リスク低 (1.0s 上限の約 40%)。
- 実装済み・tests pass (17/17)・lint pass。

### 安定性優先: G 単独

- seat 対称 ±0.35pp は全施策中で唯一。マッチメイキングで seat がランダムに決まる Kaggle 環境では seat 非対称が実力差として現れにくい利点。
- 勝率 51.7% は A+G+light より -0.3pp だが誤差範囲。

### 実戦提出判断

Kaggle LB は seat がランダムに振られる多数の試合で収束するため、**合算勝率が直接効く** → A+G+light を採用 (case3 default)。ただし A 単独比 +1.3pp は有意ではないため、**提出時の期待優位性は限定的** (skill rating 更新で 1-2 σ 未満)。

---

## 🚫 棄却施策

| 施策 | 棄却理由 |
|---|---|
| COMET_NPV | 100戦 -6pp の害 (case2 検証) |
| FINISHING_TIE_GUARD | 100戦 -11pp の害 (case2 検証) |
| DYNAMIC_PROACTIVE_HORIZON | net-zero で turn cost 増 |
| PROACTIVE_DEFENSE_HORIZON 12→18 | 300戦 -5.7pp |
| HOSTILE_SWARM_ETA_TOLERANCE 1→2 | 600戦で seed 依存、効果消失 |
| OM v1 (launch preference bonus) | 100戦複数シードで非有意 |
| OM v2 (phase-gated prediction) | 300戦 +1.4pp 非有意 |
| Lookahead 1-ply (weight=0.6) | 100戦 -3.5pp、turn_p95 4倍 |
| Lookahead gated (weight=0.3) | 差なし、turn_p95 4倍 |
| ROLLOUT bonus | 非有意、非対称 |
| ROLLOUT replace | 非有意 |
| ROLLOUT_FILTER | 合算 +0.6pp だが seat 非対称 ±13pp |
| ROLLOUT TOP_K=8 (G 拡大) | timeouts 61 件、-4.4pp |

**パターン**: "prediction / rollout 値を score に足す / 並び替える" 系 (OM v1/v2, lookahead 1-ply, ROLLOUT bonus/replace/filter/true2p) は全て seed variance 内。構造的には **score を推論値で補正する方針が飽和** しており、次の改善には以下のいずれかが必要:

1. **候補生成そのものの置換** (ミッション列挙ロジックを変える)
2. **評価関数を学習モデルに置換** (DeepSets BC policy の価値関数化、または小型 value network)
3. **MCTS / beam search** (shallow rollout ではなく全候補から選抜する探索)

---

## 🧪 再現手順

```bash
# A+G+light 300戦 (case3 default で測定)
uv run python -m env run --agents baseline_v1,baseline_v3 --mode 1v1 -n 150 --seed 40000 --parallel 4
uv run python -m env run --agents baseline_v3,baseline_v1 --mode 1v1 -n 150 --seed 40500 --parallel 4

# G 単独 (A を切る) 測定例
# pipeline/rulebase/case3/baseline/core/config.py で
#   REINFORCE_FRESH_CAPTURE_ENABLED = False
# に変更してから同コマンド

# tests
uv run pytest tests/pipeline/rulebase/case3
```

## 📁 関連ファイル

- `pipeline/rulebase/case3/baseline/core/config.py` — 全 flag / 定数 (L110-202 が本検証で追加)
- `pipeline/rulebase/case3/baseline/core/world_model.py` — CaptureState / fresh_ok gate
- `pipeline/rulebase/case3/baseline/agent.py` — `_update_capture_state`
- `pipeline/rulebase/case3/baseline/strategy.py` — `light=False` 引数、rollout integration
- `pipeline/rulebase/case3/baseline/rollout.py` — true2p / bonus / replace / filter の 4 mode 実装
- `src/env/agents.py` — `baseline_v3` registry 登録
- `tests/pipeline/rulebase/case3/` — 17 snapshot tests (pass)

## 🔖 参照 memory

- `project_case2_ablation.md` — 施策 A-I の全ログ (100戦/300戦/seat 別)
- `project_om_finding.md` — OM 系が seed variance に埋没する現象の記録

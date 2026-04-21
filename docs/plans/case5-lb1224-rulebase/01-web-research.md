# case5 (LB 1224 rulebase) — Web 技術調査

`00-codebase-research.md` で得た知見を補完するため、Kaggle の規約・類似 OSS 戦略・Python 実装パターンを外部ソースから調査した結果。

## 1. ライセンス / 帰属の確認

### Kaggle Public Notebook のライセンス

- Kaggle で **Public** に設定された notebook は **Apache 2.0 License** が自動適用される（出典: [How to add a license like MIT or Apache to a Kaggle notebook?](https://www.kaggle.com/discussions/getting-started/215978)、[Notebook Licensing Questions](https://www.kaggle.com/general/159834)）
- 対象 notebook ([`romantamrazov/orbit-star-wars-lb-max-1224`](https://www.kaggle.com/code/romantamrazov/orbit-star-wars-lb-max-1224)) は Public 公開のため Apache 2.0 適用と推定される
- Apache 2.0 は **再配布時に著作権表示 (NOTICE)、ライセンス本文、変更点の明示が必須**

### 本プロジェクトでの対応方針

- `pipeline/rulebase/case5/baseline/LICENSE` に Apache 2.0 本文をコピー (case4 と同パターン: `pipeline/rulebase/case4/baseline/LICENSE` 参照)
- `pipeline/rulebase/case5/baseline/__init__.py` の冒頭コメントに以下を記載:
  - 出典 URL (Kaggle notebook 永続リンク)
  - 著者名 (Roman Tamrazov)
  - 適用ライセンス (Apache 2.0)
  - "Adapted from ... with refactoring for readability and testability." と変更概要
- README に同様の出典明記 (case5 直下の docstring か `pipeline/rulebase/case5/README.md` のどちらか — case4 と統一)

## 2. 類似 OSS プロジェクトの分析

### 2.1 Lux AI Season 1 — `vitoque-git/Kaggle-luxai-Season1-Python-Framework` ([repo](https://github.com/vitoque-git/Kaggle-luxai-Season1-Python-Framework))

- **関連性**: 1ターン秒単位の actTimeout を持つ Kaggle simulation 競技で、ルールベース・ミッション駆動エージェントを構築している
- **アプローチ**: ゲーム状態と「ミッション」を pickle で永続化し、毎ターン「mission を継続するか / 新ミッションを発行するか」を判断
- **本プロジェクトへの適用**: case5 では Kaggle 環境のステートレス制約 (turn 間に永続ストレージ無し) のため pickle 戦略は不採用。ただし **「Mission を独立データクラスにして優先度ソートする」設計は採用** (notebook も同パターン)
- **学べる落とし穴**: mission 継続判定が無いと「同じ艦隊を毎ターン違う方向に振る」アンチパターンに陥る → notebook の `apply_score_modifiers` がこの役割を兼ねている

### 2.2 Halite IV — `0Zeta/HaliteIV-Bot` (4位ソリューション、[repo](https://github.com/0Zeta/HaliteIV-Bot))

- **関連性**: Two Sigma Halite IV のルールベース上位解。船を **役割 (MINING / RETURNING / HUNTING / GUARDING)** に分類し、各役割ごとにスコアリング → 線形割当で目標を決定
- **本プロジェクトへの適用**: notebook の `Mission` 構造は Halite の役割分類と同思想。ただし notebook はミッション側にスコアを持たせ、**全ミッションを優先度ソートして順次採用** する方式。case4 もこの方式に近いため、case5 もそのまま踏襲できる
- **学べる落とし穴**: スコア関数が役割ごとに独立しすぎると「自軍内で目標重複」が起きる → notebook は `settle_plan` で送出後の艦隊・惑星状態をシミュレートして次ミッションのスコアに反映 (重要パターン)

### 2.3 Lux AI Season 2 — `AdamSlay/LuxAI-competition-agent` (top 25, [repo](https://github.com/AdamSlay/LuxAI-competition-agent))

- **関連性**: 完全ルールベース上位解。複数の戦略フェーズ (factory placement / resource pathfinding / combat) を mode flag で切り替え
- **本プロジェクトへの適用**: notebook の `build_modes` (ahead/behind/finishing/total_war) は同思想。**mode の切替条件を define-then-apply 方式で 1 関数にまとめると保守性が上がる** という pattern を踏襲

### 2.4 OSS パターン比較表

| 観点 | case5 (notebook 移植) | Lux S1 (vitoque) | Halite IV (0Zeta) | Lux S2 (AdamSlay) | 推奨 |
|------|---------------------|------------------|--------------------|--------------------|------|
| Mission 抽象化 | dataclass + score | dict + state | role enum + score | strategy class | dataclass + score (notebook 準拠) |
| Mode 切替 | 5+ flag を `build_modes` 集約 | turn数で switch | 中盤/終盤 2 段階 | mode flag | flag 集約方式 (notebook 準拠) |
| Settle/再評価 | `settle_plan` で逐次反映 | mission 永続化 | 線形割当 | フェーズ分け | 逐次 settle (notebook 準拠) |
| 時間予算管理 | `SOFT_ACT_DEADLINE` + per-phase min time | turn 内で固定 | 計測なし | 計測なし | deadline 必須 (notebook 準拠) |

**結論**: notebook のアーキテクチャは Halite/Lux 上位解と独立に同じ設計に到達しており、実績ある形。**そのまま踏襲が最善**で、case4 のサブパッケージ分割に合わせて再編する。

## 3. Python 実装パターン

### 3.1 deadline ベース soft budget (`time.perf_counter()`)

`time.perf_counter()` は **モノトニック・最高解像度の経過時間計測関数**で、ベンチマークや短時間タイムアウトに最適 (出典: [time.perf_counter() function in Python - GeeksforGeeks](https://www.geeksforgeeks.org/python/time-perf_counter-function-in-python/))。

#### notebook 採用パターン (1986 行目以降より)

```python
def agent(obs, config=None):
    start = time.perf_counter()
    act_timeout = (config or {}).get("actTimeout", 1.0)
    deadline = start + min(SOFT_ACT_DEADLINE, act_timeout * SOFT_ACT_FRACTION)
    world = build_world(obs)
    return plan_moves(world, deadline=deadline)
```

各重いフェーズ (例: `build_recapture_missions`, `build_crash_exploit_missions`) の冒頭で:

```python
remaining = deadline - time.perf_counter()
if remaining < HEAVY_PHASE_MIN_TIME:
    return []   # フェーズ全体スキップ
```

#### case5 への適用方針

- `core/config.py` に以下を移植:
  - `SOFT_ACT_DEADLINE = 0.82` (絶対上限)
  - `SOFT_ACT_FRACTION = 0.82` (actTimeout の 82%)
  - `HEAVY_PHASE_MIN_TIME = 0.16` (重いフェーズの最低必要時間)
  - `OPTIONAL_PHASE_MIN_TIME = 0.08` (オプションフェーズ)
- `agent.py` で deadline 計算 (case4 には無い処理)
- `strategy.py` の `plan_moves(world, *, deadline: float)` シグネチャに変更
- ヘルパー: `_remaining(deadline)` または `_should_skip(deadline, min_time)` を `core/timing.py` に集約 (テスト容易性のため)

### 3.2 dataclass の不変化 (`frozen=True`)

backend.md ルール: "NEVER mutate objects — always create new instances"

notebook の `ShotOption` は `frozen dataclass`。`Mission` は notebook 内では **mutable** (`score` を後付け加算) だが、case5 移植時は backend.md 規則に従い **`frozen=True` + 新インスタンス生成** に統一する:

```python
@dataclass(frozen=True, slots=True)
class Mission:
    kind: str
    source_id: int
    target_id: int
    angle: float
    ships: int
    score: float
    eta: int

    def with_score(self, score: float) -> "Mission":
        return replace(self, score=score)
```

### 3.3 大規模関数の分解 (`plan_moves` 480 行)

backend.md ルール: "Functions are small (<50 lines), files are focused (<800 lines)"。
notebook の `plan_moves` 480 行は完全に違反。case4 同様、以下の責務に分解:

1. **収集フェーズ** (`collect_missions(world, deadline)`): 各 mission builder を呼び priority list を返す
2. **解決フェーズ** (`resolve_missions(world, missions, deadline)`): score sort + settle_plan で実行可能 action に変換
3. **後処理フェーズ** (`apply_movements(world, planned, deadline)`): evacuation / rear-guard / followup の追加

`strategy.py` をオーケストレーション層 (~150 行) に保ち、ロジック本体は `missions/` `movements/` 配下へ。

## 4. ライブラリ選定

新規依存は **不要**。notebook は標準ライブラリと NumPy のみ。case4 と同じ依存セット (`numpy`, `kaggle-environments`) で完結する。

| 検討項目 | 採用 | 理由 |
|---------|------|------|
| `numpy` | 採用 | 既存依存、ベクトル化済 |
| `dataclasses` | 採用 | 標準ライブラリ |
| `attrs` | 不採用 | 標準 dataclass で十分 |
| `pydantic` | 不採用 | 速度 critical 、不要 |
| `numba` / `cython` | 不採用 | 既存パイプラインの依存範囲外、ROI 低 |
| `polars` | 不採用 | エージェント本体で表形式処理は不要 |

## 5. API / 環境仕様の確認

`docs/competition/abstract.md` および backend.md より:

- `actTimeout = 1.0s / step` (Lux S1 は 3s。Orbit Wars は厳しめ)
- `obs` の構造: `player`, `step`, `planets`, `fleets`, `comets`, `comet_planet_ids`, `initial_planets`, `angular_velocity`
- `action` 形式: `[[from_planet_id, angle, num_ships], ...]`
- 4-player FFA も同 actTimeout (mode 駆動の `is_four_player` で挙動分岐)

case5 は notebook 同様 1v1 と FFA 両対応。

## 6. Research Summary (設計に直結する結論)

1. **ライセンス遵守**: notebook は Apache 2.0。case5 にライセンス本文と出典コメントを必須配置
2. **アーキテクチャ**: notebook のミッション駆動設計は Halite/Lux 上位解と同形 → そのまま採用、case4 のサブパッケージ分割パターンに合わせて再編
3. **deadline 制御は移植必須**: notebook の最大の強みであり、削ると notebook の品質が出ない → `core/timing.py` を新設して集中管理
4. **不変データクラス化**: notebook の mutable Mission は backend.md ルール違反 → `frozen=True` + `replace()` パターンに統一
5. **`plan_moves` の分解**: 480 行 → 3 段階 (collect / resolve / apply) に分解、責務ごとに別ファイル
6. **新規依存ゼロ**: 既存パイプラインの依存セットで完結
7. **OM/lookahead は持ち込まない**: notebook 自体に無く、memory (`project_om_finding.md`) でも有意改善なしと記録済 → case5 では完全に削除

## 出典一覧

- [Kaggle: How to add a license like MIT or Apache to a Kaggle notebook?](https://www.kaggle.com/discussions/getting-started/215978)
- [Kaggle: Notebook Licensing Questions](https://www.kaggle.com/general/159834)
- [vitoque-git/Kaggle-luxai-Season1-Python-Framework](https://github.com/vitoque-git/Kaggle-luxai-Season1-Python-Framework)
- [0Zeta/HaliteIV-Bot (4位)](https://github.com/0Zeta/HaliteIV-Bot)
- [AdamSlay/LuxAI-competition-agent (top 25)](https://github.com/AdamSlay/LuxAI-competition-agent)
- [GeeksforGeeks: time.perf_counter() function in Python](https://www.geeksforgeeks.org/python/time-perf_counter-function-in-python/)
- [Kaggle: orbit-star-wars-lb-max-1224 (移植元 notebook)](https://www.kaggle.com/code/romantamrazov/orbit-star-wars-lb-max-1224)

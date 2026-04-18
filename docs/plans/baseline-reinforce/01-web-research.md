# baseline-reinforce — Web技術調査

## Kaggle Environments 公式仕様

### `orbit_wars` 環境ファイル配置
`kaggle-environments` レポジトリの `kaggle_environments/envs/orbit_wars/` 配下：

| ファイル | 内容 |
|----------|------|
| `orbit_wars.py` | ゲームロジック本体。`Planet`, `Fleet` namedtuple、定数、`interpreter`, `renderer`, 戦闘解決、軌道更新 |
| `orbit_wars.json` | spec.json。configuration schema、observation/action 形式、default agents |
| `orbit_wars.js` | 可視化 |
| `test_orbit_wars.py` | 環境のユニットテスト |
| `README.md` | 環境ドキュメント |

- リポジトリ: https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/orbit_wars
- ベースライン依存バージョン: `kaggle-environments>=1.17.0` で orbit_wars 同梱。

### Named Tuple と主要定数
```python
Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet  = namedtuple("Fleet",  ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]
```

### Agent Entrypoint
- 関数名 `agent`、シグネチャ `agent(observation, configuration=None)`。
- observation は **dict**。`planets`, `fleets`, `player`, `angular_velocity`, `initial_planets`, `next_fleet_id`, `comets`, `comet_planet_ids`, `remainingOverageTime`, `step`。
- 返り値 `list[[from_planet_id:int, angle:float, num_ships:int]]`。

### 実行API
```python
from kaggle_environments import make
env = make("orbit_wars", configuration={"episodeSteps": 500}, debug=True)
env.run([agent1, agent2])             # 1v1
env.run([agent1, agent2, agent3, agent4])  # 4 FFA
env.render(mode="json")                # リプレイ取得
```

### Kaggle ノートブック取得
- 認証: `~/.kaggle/kaggle.json` (`{"username": "...", "key": "..."}`)、chmod 600 推奨。
- CLI: `pip install kaggle` もしくは `uv add kaggle`。
- 取得コマンド: `kaggle kernels pull sigmaborov/orbit-wars-2026-reinforce -p pipeline/case1/notebook/ -m`（`-m` でメタデータも取得）。
- 出力: `orbit-wars-2026-reinforce.ipynb` + `kernel-metadata.json`。
- `.ipynb` → `.py` 変換: `jupyter nbconvert --to script pipeline/case1/notebook/orbit-wars-2026-reinforce.ipynb`（jupyter は optional、ノートブックをそのまま扱う場合は不要）。

## 類似OSSプロジェクト

### 1. aichallenge/planet-wars (2010)
- 前身コンペ。状態管理は関数ベース、ターゲット優先度はシンプルなスコア式。
- **再利用可能なパターン**: 「defensive / expansion / aggressive」のフェーズ分けと、ターゲット価値を `production / (distance + const)` で表す基本式。
- **注意**: Planet Wars は直線距離のみで、Orbit Wars の「軌道公転」「コメット」「太陽衝突」は未考慮。再現対象のノートブックは **これらを精密にモデル化している点** が差別化要素。

### 2. SimonLucas/planet-wars-rts
- Kotlin 実装。エージェント評価フレームワーク。
- **再利用可能なパターン**: 「ミッション」という概念で行動を抽象化し、ミッションごとにスコアを計算→ソート→リソース制約下で順次実行。**再現対象のノートブックと同じ発想**。
- **落とし穴**: 過剰に汎用化すると本コンペの `actTimeout=1s` を破る。Python では軽量 namedtuple + 早期 return に徹する。

### 3. eonarheim/planet-wars-competition
- JS 実装だが heuristic の参考に。
- **再利用可能なパターン**: 「Doomed planet（陥落確定星）からの事前撤退」アイデア。本ノートブックの `doomed_planets` と同発想。

### パターン比較表

| 観点 | 再現対象 (Reinforce 928.5) | Planet Wars 標準 | Kaggle Tactical Heuristic (587.5) | 推奨 |
|------|---------------------------|------------------|-----------------------------------|------|
| 状態管理 | `WorldState` dataclass | 関数ベース | 関数ベース | **`WorldState` に統一** |
| ミッション型 | 6種 (expand/attack/snipe/swarm/reinforce/crash_exploit) | 2種 (expand/attack) | 4種 | **6種全部を再現** |
| Swarm | 2-source + 3-source | なし | 2-source | **両方サポート** |
| 軌道予測 | `predict_target_position` 統一API | なし | 基本的 | **統一API必須** |
| フェーズ分け | early/opening/normal/late/very_late | なし | 3段階 | **5段階維持** |
| パラメータ数 | ~80 | ~10 | ~30 | **80維持（後のチューニング余地）** |

## ライブラリ/サービス選定

### 依存追加候補

| 目的 | 候補 | 推奨 | 理由 |
|------|------|------|------|
| Kaggleノートブック取得 | `kaggle` (公式) | **⭐採用** | コンペ内では唯一の API クライアント、メンテナンス継続中 |
| ipynb→py 変換 | `jupyter`, `nbconvert`, `jupytext` | **`nbconvert`** | 既に `jupyter` 経由で広く使われ、.py 出力が素直 |
| プログレス表示 | `rich` | **既存採用** | 既に `rich>=13.9.4` が依存に入っている |
| CLI | `typer` | **既存採用** | 既に依存に入っている |
| テスト乱数制御 | `numpy`, `random` stdlib | **stdlib** | 追加不要 |

### 既存依存で十分
- `numpy>=2.2.6`: ベクトル計算・行列演算。
- `pandas>=2.3.3`, `polars>=1.39.0`: 自己対戦結果の集計。
- `pyarrow>=23.0.1`: parquet 出力（リプレイ集計用）。

### 追加する依存（最小限）
1. `kaggle` (ノートブック取得のみ) — **dev依存で追加** し、`uv run kaggle ...` で呼ぶ。
2. （任意）`nbconvert` — `.ipynb → .py` が必要な場合のみ。最初は `.ipynb` のまま保持する方針であれば不要。

## API / Protocol 研究

### Kaggle Kernels API
- `kaggle kernels pull <owner>/<slug>` — 最新版を取得。
- `kaggle kernels pull <owner>/<slug> --version <n>` — バージョン指定。対象は **v2** で Public Score 928.5。
- `kaggle kernels push -p <dir>` — 提出（本ベースラインではそのまま提出可能）。

### Submission 制約
- 1日最大5提出、最新2件が最終候補。
- エージェントは単一 `.py` を推奨（`main.py`）。依存モジュールを import する場合は、同ディレクトリから相対 import にする。
- Kaggle 側環境では `kaggle_environments` と `numpy` は利用可。`torch` は Submission の Python 環境で可だがコールドスタートが重い。**heuristic なら不要**。

## 研究結果サマリ

### 採用する設計方針
1. **ノートブック取得は `kaggle` CLI** を `dev` 依存として追加し、`pipeline/case1/notebook/` に保存。`.ipynb` のまま保持して差分追跡しやすくし、Pythonソースは同ディレクトリに `baseline_agent.py` として抽出。
2. **二段階実装**: Step Aで「ノートブック丸写しの単一 `.py` で動作確認」、Step Bで「`src/` のモジュール階層に分解」。本 featureでは **Step A までを対象** とし、Step B は次のイテレーションで実施（scope を守る）。
3. **ミッションシステム 6種を完全移植**: `expand`, `attack`, `snipe`, `swarm`, `reinforce`, `crash_exploit`。
4. **自己対戦の検証**: `pipeline/case1/evaluation/selfplay.py` を `typer` CLI で用意し、Reinforce vs Reinforce / Reinforce vs Random を N エピソード実行し、勝率・平均ターン・タイムアウト率を記録する。
5. **Ruff/Mypy**: `pipeline/case1/baseline/**` に `per-file-ignores` を設定し `C901`（複雑度）と `E501`（長大行）を緩和。ノートブック互換性を優先。
6. **依存追加**: `kaggle` のみ (dev)。重い依存 (`torch` 等) は追加しない。

### 外部調査から採用する具体的知見
- **Swarm攻撃の2/3-source 両対応**: Planet Wars 研究で「multi-source 攻撃は並みのソロ攻撃より遥かに強い」とされ、本ノートブックの 3-source Swarm（PLAN_PENALTY=0.93）はこれを具現化。
- **Doomed planet evacuation**: 陥落確定星から艦を逃がすアイデアは複数 OSS で共通。
- **Forward simulation + cache**: `projected_state` の base_need_cache パターンは Planet Wars 系で定番。1ターン1秒制約下では必須。
- **フェーズ依存 multiplier**: early/opening/normal/late/very_late の各段階で評価関数の乗数を切り替えるのは、強豪Planet Warsボットの共通テクニック。

## 出典

- [Kaggle/kaggle-environments (GitHub)](https://github.com/Kaggle/kaggle-environments)
- [orbit_wars env directory](https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/orbit_wars)
- [kaggle-environments PyPI](https://pypi.org/project/kaggle-environments/)
- [Orbit Wars 2026 - Reinforce (Kaggle notebook)](https://www.kaggle.com/code/sigmaborov/orbit-wars-2026-reinforce)
- [Planet Wars Strategy idea guide](https://github.com/aichallenge/aichallenge/wiki/Planet-Wars-Strategy-idea-guide)
- [SimonLucas/planet-wars-rts](https://github.com/SimonLucas/planet-wars-rts)
- [eonarheim/planet-wars-competition](https://github.com/eonarheim/planet-wars-competition)
- [Kaggle Orbit Wars Competition](https://www.kaggle.com/competitions/orbit-wars)

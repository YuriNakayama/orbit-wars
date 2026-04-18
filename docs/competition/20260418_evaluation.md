# Orbit Wars ローカルシミュレーション まとめ

## 1. ローカル実行の基本コード

Overview および Starter ノートブックで紹介されている、ローカルで実行する基本パターンです。

```python
from kaggle_environments import make

env = make("orbit_wars", debug=True)
env.run(["main.py", "random"])

# 結果確認
final = env.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")

# ノートブックでのレンダリング
env.render(mode="ipython", width=800, height=600)
```

---

## 2. セットアップの注意点（重要）

Discussion「**can't run locally, orbit_wars Unknown Environment Specification [Solution Found]**」で報告されている既知の落とし穴があります。

### 症状

単に `pip install kaggle-environments` すると、以下のエラーが発生します。

```
InvalidArgument: Unknown Environment Specification
```

原因は、公開 PyPI 版に `orbit_wars` 環境がまだ含まれていない可能性があるためです。

### 解決策A：バージョン指定

c-number さんによれば「1.28.0 が PyPI からダウンロード可能」とのこと。

```bash
pip install kaggle-environments==1.28.0
```

### 解決策B：GitHub master から直接インストール（Timme さん推奨）

`requirements.txt` に以下を記載:

```
kaggle>=1.6.0
kaggle-environments @ git+https://github.com/Kaggle/kaggle-environments.git@master
```

そして:

```bash
pip install -r requirements.txt
```

---

## 3. ソースコード（ローカルデバッグ／自前シミュレータ用）

Discussion「**Source code**」で Kaggle スタッフの Bovard 氏が確認したとおり、全シミュレーション系コンペのソースは公開されています。

**リポジトリ:**
<https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/orbit_wars>

### ここで確認できる実装内容

- Planet / Fleet のデータ構造
- ターン処理順序
  1. コメット消失
  2. コメットスポーン
  3. 艦隊発射
  4. 生産
  5. 艦隊移動
  6. 惑星回転・コメット移動
  7. 戦闘解決
- 戦闘解決ロジック
- 艦隊速度式: `speed = 1.0 + (maxSpeed - 1.0) * (log(ships)/log(1000))^1.5`

自前で高速シミュレータを書く場合や、observation の挙動をデバッグしたい場合はこのコードを読むのが近道です。

---

## 4. エージェント側の便利ツール

`kaggle_environments.envs.orbit_wars.orbit_wars` から namedtuple や定数をインポートでき、observation のアンパックが楽になります。

```python
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT
)

def agent(obs):
    planets = [Planet(*p) for p in obs.get("planets", [])]
    fleets = [Fleet(*f) for f in obs.get("fleets", [])]
    return []
```

---

## 5. 主要な Configuration パラメータ

ローカルで `make("orbit_wars", configuration={...})` に渡せる設定です。

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `episodeSteps` | 500 | 最大ターン数 |
| `actTimeout` | 1 | 1手あたり秒数 |
| `shipSpeed` | 6.0 | 艦隊の最大速度 |
| `sunRadius` | 10.0 | 太陽の半径 |
| `boardSize` | 100.0 | ボードサイズ |
| `cometSpeed` | 4.0 | コメットの速度（units/turn） |

短いエピソードで多数回のテストを回したい場合などに調整できます。

---

## 6. 関連する Discussion / Notebook（参考）

### Discussion

- **How to download replays and logs via kaggle cli?** — リプレイとログのダウンロード方法
- **baseline method: Planet Wars AI Competition** — 前身コンペのベースライン流用
- **Possible orbit_wars observation inconsistency: initial_planets differs by player after comet updates** — コメット更新後に `initial_planets` がプレイヤーごとに異なる可能性がある問題（13票）

### Notebook（Code タブ）

- **Orbit Wars 2026 - Starter**（48票・Silver）
- **🗡️ Orbit Wars: Structured Baseline**（19票・Bronze）
- **☀️ Orbit Wars: Sun-Dodging Baseline**（21票・Bronze）

これらはローカルで動かして挙動を観察するのに便利です。

---

## 7. 既知の留意事項

observation に関する既知の問題として「**initial_planets differs by player after comet updates**」（コメット更新後に `initial_planets` がプレイヤーごとに異なる可能性がある）という報告があります。ローカルで再現・検証する際には知っておくと役立ちます。
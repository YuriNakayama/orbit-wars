# rulebase-to-jax — ローカル自己対戦 & GPU 高速化選択肢

## A. ローカルで JAX agent vs 本物 Python を自己対戦する方法

### 結論: JAX env は CPU でそのまま動く (GPU 不要)

Web 調査 ([JaxMARL](https://blog.foersterlab.com/jaxmarl/)) と repo 確認の通り、JAX env は GPU 比で遅いだけで **CPU でも普通に動く**。本 repo の `orbit_wars_jax` は `JAX_PLATFORM_NAME` 等の device 設定なしで CPU 既定で走る。**parity / debug にはこれで十分**。

### 最短経路: CPU で 1 ゲーム (JAX seat0 vs 本物 Python seat1)

`reset() → step()` ループに両 agent を差すだけ。GPU 不要、vmap 不要、`bot/` 配下で:

```python
import jax, jax.numpy as jnp
from orbit_wars_jax.reset import reset
from orbit_wars_jax.step import step, MAX_LAUNCHES_PER_AGENT
from orbit_wars_jax.observation import state_to_obs
from pipeline.rulebase.case1.baseline.agent import agent as v1_py
from pipeline.rulebase.case1.baseline_jax import compute_actions_jax

state = reset(seed=42, num_agents=2)
for turn in range(500):
    a0 = compute_actions_jax(state, seat=0)            # JAX port (seat0)
    moves = v1_py(state_to_obs(state, player=1))       # 本物 Python (seat1)
    a1 = jnp.full((MAX_LAUNCHES_PER_AGENT, 3), -1.0, jnp.float32)
    for i, m in enumerate(moves[:MAX_LAUNCHES_PER_AGENT]):
        a1 = a1.at[i].set(jnp.asarray([m[0], m[1], m[2]], jnp.float32))
    actions = jnp.full((4, MAX_LAUNCHES_PER_AGENT, 3), -1.0, jnp.float32).at[0].set(a0).at[1].set(a1)
    state, rewards, done = step(state, actions)
    if done:
        print("winner:", 0 if rewards[0] > rewards[1] else 1 if rewards[1] > rewards[0] else "draw")
        break
```

参照: `bot/tests/unit/jax_env/test_step_smoke.py:22-32` (500-step CPU ループの実例)、`compute_actions_jax` は `case1/baseline_jax/agent_jax.py:44`。

### デバッグの実務 (Web 調査の知見)

- **`jax.disable_jit()` / `JAX_DISABLE_JIT=1`**: jit を切ると素の Python として 1 行ずつ実行・print・pdb 可能。port のロジックバグ追跡に必須 ([disable_jit](https://docs.jax.dev/en/latest/_autosummary/jax.disable_jit.html))。
- **`jax.debug.print(...)`**: jit 内でも中間値を出力 ([debugging](https://docs.jax.dev/en/latest/debugging/index.html))。
- **`jax.config.update("jax_enable_x64", True)`**: parity test は float64 で (07-implementation-strategy.md 原則2)。
- **本物 callback で正解を生成**: 既存 `_host_python_v1_action` (`rollout_jax.py:201`) と同じパターンで「同一 obs に対する本物 Python の action」を golden として保存し、JAX port の出力と diff する差分テストが作れる。

### 既存ハーネスの活用

- **単一 match runner**: `bot/src/dataset/selfplay/executor.py:93` `run_one_match(MatchSpec)`。agent は名前文字列で resolve。勝者・ターン数・replay を返す。ただし現状は kaggle env (rust/python backend) ベースで、JAX agent は名前未登録。
- **1-episode vmapped rollout**: `collect_rollout_jax(..., episodes_per_iter=1, opponent="python_v1")` で CPU 単一 match をデバッグ可能 (`rollout_jax.py:552`)。
- → **方針**: parity 検証用に「CPU で reset→step ループを回し JAX port vs 本物 Python の勝敗 + step 毎 action 一致を出すローカルスクリプト」を `_bench` か test として 1 本用意する (Step 4 の e2e parity に組み込み)。GPU 起動不要で素早く回せる。

## B. JAX 以外の GPU 高速化選択肢

Web 調査で比較。**branchy なゲームロジック**という本件の性質が効いてくる。

| 技術 | 速度 | branchy ロジック適性 | 本件での評価 | 出典 |
|------|------|---------------------|--------------|------|
| **JAX (XLA)** | Mandelbrot で CuPy/Numba 比 **6×** | ⚠️ static control flow 向き。分岐多いと XLA fusion が効きにくい | ◎ 既存資産 (env/featurizer/PFSP) と統合、vmap self-play が主目的に最適 | [jax#11078](https://github.com/jax-ml/jax/discussions/11078) |
| **Rust (PyO3)** | 単一 match **27× (serial) / 180× (parallel12)** | ◎ 任意の制御フロー自然に書ける | ◎ **既に backend 実装済**。CPU 並列で self-play 大量実行に最強。GPU は使わない | `simulator/rust/README.md:100`, [PyO3 RL](https://huggingface.co/papers/2603.12145) |
| **Numba CUDA** | irregular workload で JAX より自然な場合あり | ◎ 明示 CUDA kernel で分岐自由 | △ 新規依存・CUDA 知識要・既存資産と非統合。本件の利得薄 | [Numba/CuPy](https://medium.com/@ThinkingLoop/7-numba-cupy-boosts-that-give-gpus-to-plain-python-bb6b931e0cc9) |
| **CuPy** | numpy 互換 GPU array | ✗ vectorize 前提、分岐は苦手 | △ JAX と役割重複、優位なし | 同上 |
| **Triton / Warp / Taichi** | custom kernel / 粒子・物理 simulation 向き | ○ | ✗ ゲームロジックには過剰、新規学習コスト大 | [Warp](https://news.ycombinator.com/item?id=40680737), [Taichi](https://docs.taichi-lang.org/blog/taichi-compared-to-cub-cupy-numba) |

### 重要な気づき: GPU が常に最速とは限らない

- Web 調査 ([Rust PyO3 RL](https://huggingface.co/papers/2603.12145)): 「JAX GPU は並列 simulation で大きな speedup、**Rust+PyO3 は GPU が無い時の CPU 並列に有効**」。
- 本 repo の **Rust backend は単一 match 27×・並列 180×** を CPU だけで達成済み。**大量 self-play / 評価 (NFR の throughput) は実は Rust backend が既に高速**。
- JAX vmap の真価は **RL 訓練ループと env が同一 device に乗り CPU↔GPU 転送を消す**点 (PureJaxRL 1000×, [HN](https://news.ycombinator.com/item?id=35474968))。**opponent も JAX 化されて初めて rollout 全体が GPU 上で閉じる** — これが今回 rulebase を JAX 化する本質的理由 (Rust opponent だと毎 step CPU 呼び出しが残り GPU rollout が途切れる)。

### 方針への反映

1. **GPU 高速化技術は JAX 一択で正しい** — 理由は速度単体でなく「PFSP rollout 全体を GPU 上で閉じる」ため。Numba/CuPy/Triton 等への乗り換えは利得が新規コストに見合わず不採用。
2. **ただし「大量 self-play / 評価」だけが目的なら既存 Rust backend (CPU 並列 180×) で足りる** — JAX 化の主目的を「RL opponent pool」に絞った 02-requirements の判断は妥当 (評価高速化は副次)。
3. **ローカル parity は CPU JAX で完結** — GPU 起動 (RunPod) は Step 10 の throughput bench / live 300戦 のみ。Step 1–9 は全て手元 CPU で回せる ⇒ 反復が速く失敗回避 (07 の差分テスト) を高速に実践できる。

## まとめ

| 問い | 答え |
|------|------|
| ローカルで JAX vs 本物 Python 自己対戦は? | CPU で `reset→step` ループに両 agent を差すだけ。GPU 不要 ([smoke test 実例あり](bot/tests/unit/jax_env/test_step_smoke.py)) |
| デバッグは? | `jax.disable_jit()` + `jax.debug.print` + x64、本物 callback を golden に差分 |
| JAX 以外の GPU 技術は? | Numba CUDA/CuPy/Triton/Warp あるが本件は不採用 (新規コスト>利得、branchy で JAX も万能でない) |
| なぜ JAX なのか (速度以外) | opponent も JAX 化して **PFSP rollout 全体を GPU 上で閉じCPU↔GPU 転送を消す**ため。これが Rust opponent では達成不可 |
| 大量評価だけなら? | 既存 Rust backend が CPU 並列で 180×、JAX 不要 |

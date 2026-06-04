# rulebase-to-jax — Web Technical Research

## Official Documentation

- **`jax.lax.scan`**: 固定 horizon の先読みループは `lax.scan(length=N)` に。carry は固定 shape/dtype を全 iteration 維持、葉は全て JAX array。蓄積量は carry でなく per-step output (`ys`) に出す。XLA `While` HLO にコンパイルされ GPU で効率実行。([jax.lax.scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html), [control-flow](https://docs.jax.dev/en/latest/control-flow.html))
- **`lax.cond` / `lax.switch` の罠**: vmap 下で全 branch が実行され `select` 化される。mission 分岐や opponent 分岐に使うと毎バッチ全 branch 評価 + コンパイル肥大。([#20916](https://github.com/jax-ml/jax/discussions/20916), [lax.switch](https://docs.jax.dev/en/latest/_autosummary/jax.lax.switch.html))
- **マスク付き argmax**: `masked = jnp.where(valid_mask, scores, -jnp.inf); jnp.argmax(masked)`。全 masked 時は argmax が index0 を返すので `mask.any()` で NO_OP 判定を別持ち。
- **⚠️ `lax.scan` は GPU で遅い場合がある (追加調査 2026-06)**: scan は**コンパイル時間/メモリを削減**するが、GPU では iteration 毎の kernel launch overhead で **unrolled Python loop より execution が遅い**ことがある (CPU/TPU では非該当、GPU 固有)。短い固定ループ (HORIZON=8 等) は unroll 有利。「scan = 速度最適化」ではない。([jax#16611](https://github.com/jax-ml/jax/issues/16611), [discussion#16106](https://github.com/jax-ml/jax/discussions/16106))
- **`jnp.where` / `lax.cond` は両 branch を評価**: `jnp.where`(=`lax.select`)は常に両 branch を concretely 評価。`lax.cond` も vmap 下で `select` 化され全 branch 実行 → OOM 要因。「全 mission 並列 score→argmax」方式は元々全計算前提なのでこの制約と整合し正しい。([jax#8409](https://github.com/jax-ml/jax/issues/8409), [discussion#12281](https://github.com/jax-ml/jax/discussions/12281))
- **throughput 実績**: PGX (盤面ゲーム JAX) は A100 で既存 Python RL env 比 **10–100×**。vmapped self-play の速度向上の裏付け。([Pgx arXiv:2303.17503](https://arxiv.org/abs/2303.17503))

## Similar OSS Projects

#### PGX — board game envs in JAX — https://arxiv.org/abs/2303.17503
- **Relevance**: Chess/Go/Shogi を pure JAX 関数として実装。本 repo の orbit_wars_jax と同じ思想。
- **Reusable patterns**: agent 数固定・iterator 不使用、多分岐ロジックを固定 shape の mask/select に平坦化。
- **Pitfalls**: 動的 agent 数は非対応 → MAX_* padding 必須。

#### PureJaxRL — https://github.com/luchris429/purejaxrl
- **Approach**: rollout を `runner_state=(train_state, env_state, last_obs, rng)` の tuple carry で `lax.scan`。8-turn 防衛シミュレーションの雛形に流用可。

#### JaxMARL — https://arxiv.org/abs/2311.10090, https://blog.foersterlab.com/jaxmarl/
- **Approach**: heterogeneous agent は「同 type だけ毎ステップ並列化」。異種 opponent の vmap は ① type ごとグループ別 vmap か ② param 化して 1 関数 vmap。
- **Validation**: 形式検証でなく OpenSpiel/PettingZoo に対する経験的 parity (合法手・遷移・throughput 突合)。

### Pattern Comparison

| 課題 | 第一選択 | 避ける |
|------|----------|--------|
| 8-turn 先読み | `lax.scan(length=8)` + 固定 shape carry + active mask | 動的長リスト |
| mission 選択 | 全 mission を並列 score → mask → `argmax` | `lax.switch` で mission 分岐 |
| ray-circle 当たり | 判別式 + `jnp.where` 閉形式 | `if` 分岐 |
| parity | `assert_allclose(rtol=1e-5)` + action 一致率 + tie-break 統一 | 形式検証 |
| 異種 opponent | param 化 1 関数 vmap、無理なら id でグループ別 vmap | opponent-id `lax.switch` |

## Library/Service Selection

新規依存は不要。`jax>=0.4.30` / `equinox` / `optax` は既に `bot/pyproject.toml` にあり、GPU は `--group cuda` (RunPod) で対応済み。既存 `orbit_wars_jax` と `case1/baseline_jax` を直接踏襲する。

## Research Summary

- **最重要原則(全課題共通)**: branch を select に潰さず、**固定 shape の mask 算術へ平坦化**する。`lax.cond`/`lax.switch` は vmap 下で全 branch 実行になるため、5 mission・複数 opponent のような分岐の多い箇所こそ mask + `jnp.where` + `argmax` に書き換えるのが性能・parity 両面で有利。
- **parity 安定化**: float reduction の順序差で score 同値タイの argmax がズレ得る → **tie-break ルール (index 最小等) を Python/JAX で明示的に揃える**と action 一致率が安定。
- **異種 opponent pool**: ルール agent を「strategy_id を data で渡し内部は分岐レス mask 演算の共通関数」に寄せれば param 化 1 関数 vmap に乗る。これが reinforce/case6 PFSP pool と最も相性が良い。
- 既存テンプレ `case1/baseline_jax(_full)` + `case2/aim_jax.py` + `tests/.../test_*_jax_parity.py` がそのまま流用可能。

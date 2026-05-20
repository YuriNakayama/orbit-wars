# reinforce/case1 — JAX acceleration (iter1) RESULT

> 関連: hypotheses.md
> 作業ブランチ: feature/reinforce-learning-case0
> 期間: 2026-05-18 ～ 2026-05-21
> commit range: `f8d9949` (JAX env Phase B) → `1ce9627` (W6-a)

## Summary

PPO 学習の wall-clock 短縮を目的に、reinforce/case1 の rollout + policy +
PPO update を JAX に移植 (W1-W5)、PPO update を単一 jit に圧縮 (W6-a)。
ローカル M-series CPU で **iter1 wall-clock 270s → 16.7s = 16× 高速化**
を達成。GPU bench (A100 80GB) では **27.6s = 9.8×** だが host call
overhead (固定 ~15s) が支配しており、W6-a 適用後の GPU 実測は未計測。

仮説 「PPO 学習を JAX 化すれば GPU で 10-25× 高速化できる」は **CPU で
は支持** (16×)、**GPU では条件付き支持** (現状 9.8×、W6-a 適用後の
RTX 4090 実測待ち)。

## Numbers

### Featurizer (JAX vmap)

| stage | single env | vmap(16) | vs PyTorch |
|---|---|---|---|
| PyTorch baseline | 2.97 ms | — | 1.0× |
| W1 (基本列のみ) | 0.044 ms | 0.006 ms/env | 67× / 500× |
| W2a-f (全列) | 0.766 ms | 0.52 ms/env | 3.9× / 5.7× |

W1 単独では 67-500× だが、W2 で全列実装すると 3.9-5.7× に。
parity test: **74 件全 pass、tol=1e-4** (PyTorch 出力に bit 一致)。

### Rollout + PPO update (end-to-end)

| 環境 | iter1 (16 ep × 500 step) | vs baseline |
|---|---|---|
| PyTorch baseline (pod CPU) | **270s** | 1.0× |
| JAX on M-series CPU (laptop) | **16.7s** | **16×** |
| JAX on pod CPU (RTX 4090 host) | 21.0s | 12.8× |
| JAX on A100 80GB GPU (W4 のみ) | 27.6s | 9.8× |
| W6-a 効果 (M-series, smoke 4ep × 50step) | ppo_update 18s → 3.1s | **5.8×** |

W6-a の GPU 実測は本 iter で未取得 (interactive pod setup 中)。

### Test coverage

| suite | count |
|---|---|
| jax_env (Phase B + parity) | 622 |
| featurizer parity (W1-W2f) | 74 |
| model parity (Equinox port) | 7 |
| **合計** | **703** |

全 pass、lint/mypy clean。

## Diagnosis

### 想定通りだった点
- W1 (簡易 featurizer) は vmap で 500× — JAX の本領通り
- W2a-b (planet/fleet ETA) は parity 1e-4 を維持しつつ vectorize
- W3 (ActorCritic Equinox 移植) は forward parity 1e-6 まで詰められた
- W6-a の PPO update jit 圧縮で 5.8× speedup

### 想定外だった点 / 学習

1. **W2c (orbit) で PyTorch featurizer のバグ発覚**
   - vendor `predict_comet_position` が `paths[idx]` を `(x, y)` raw として読むが、
     `paths` は `[[y, x] for x, y in visible]` で実は (y, x)。
   - BC 重みが**バグった出力で学習されている**ため、JAX 側も同じバグを再現する必要があった。

2. **W6 (host call elimination) が GPU では本質的**
   - A100 で 9.8× 止まりだった理由は **host↔device 同期コスト**。
   - W6-a (PPO update を 1 jit に圧縮) で大幅改善見込み。W6-b/c は ROI 低い。

3. **`uv sync --group X` は declarative**
   - GPU bench を 11 回投入する過程で、`uv sync --group cuda` が
     dvc[s3] の `s3fs` を剥がして dvc pull が壊れる挙動を発見。
   - 順序を `base → env → dvc → cuda → train` に固定する必要があった。

4. **JAX env と Rust simulator の comet path 表現が swap**
   - 軌道 bug + per-quadrant swap を発見、613 件の parity test を整備して
     trajectory parity 1e-5 を達成 (`28ccfef`)。

### GPU で 9.8× 止まりの root cause (W4 単独)

- A100 PCIe で 16 ep × 500 step = 27.6s
- per-step は 3.45 ms (CPU per-step 8 ms 比 2.3× 速い)
- **固定 overhead が ~15s** — jit compile warmup + host↔device 同期
- W6-a 適用後はこの 15s overhead の大部分が消える見込み

### GPU bench infra で発見した bug 6 件

1. cuda jax `uv pip install --reinstall` が jax 既ロード時に SIGABRT
2. RunPod EU-RO-1 の永続化 venv に bin/python 無し (broken venv)
3. `uv sync --group A` が前回の `--group B` の packages を剥がす
4. dvc[s3] の transitive `s3fs` が group sync で消える
5. CuDNN ABI mismatch (runtime 9.1 vs jax compile-time 9.8)
6. RTX 4090 stock 切れの retry の handling

これらを修正して 11 回目で完走。

## Decision

- **採否: adopted** (CPU で 12-16× の明確な speedup、GPU は W6-a 適用後の実測待ち)
- **次の一手**:
  1. **interactive pod で W6-a 適用 train_jax を直接実行** (現在 setup 中)
  2. iter1 wall-clock 確定 → 期待値 5-10s on RTX 4090
  3. もし 10s 以下なら W6-b/c は不要と判断、A2 完了宣言
  4. もし > 10s なら W6-b (reset JAX 化) で再評価

## Artifacts

- ブランチ: `feature/reinforce-learning-case0`
- 主要 commit:
  - `28ccfef` JAX env trajectory parity fix
  - `fb8c619` W1 JAX featurizer (基本列)
  - `21cd7fe` W2-final candidate block
  - `e645cdf` W2e timeline (full featurizer parity)
  - `87e6c60` W3 ActorCritic Equinox port + BC loader
  - `df3f7a1` W4 rollout driver
  - `bfc6c67` W4-c PPO update Optax + Equinox autograd
  - `d50db52` W5 train_jax end-to-end
  - `d16d960` W6-a PPO update jit 圧縮
- 設計 doc: `docs/plans/jax-env/01-design.md` (Phase C 追加)
- bench results: `data/output/models/reinforce/rollout_gpu/runs/20260520-133515_*` (A100 完走分)
- module:
  - `bot/src/jax_env/` (Phase B、613 test)
  - `bot/pipeline/reinforce/case1/policy/{featurizer,model,sampling,sampling_eval}_jax.py`
  - `bot/pipeline/reinforce/case1/training/{rollout,ppo,train}_jax.py`
  - `bot/pipeline/_bench/rollout_gpu/` + `_bench/featurizer_gpu/`

## 未着手 (W6 残り)

| Task | 内容 | 工数 | 必要性 |
|---|---|---|---|
| W6-b | env.reset を JAX 化 | 2-3 日 | 低 (reset 数十ms × N、ボトルネックではない) |
| W6-c | 学習ループ全体を lax.scan に閉じ込め | 1 週間 | 低-中 (iter ループは 5 回程度) |

W6-a 適用後の GPU 実測値次第。10s 未満なら **W6-b/c 不要、A2 完了宣言**。

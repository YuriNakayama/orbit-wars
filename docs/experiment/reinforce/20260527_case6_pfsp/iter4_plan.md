# Reinforce/case6 — Real Python baseline_v1 opponent via host callback (iter4)

> 作成日: 2026-06-02
> 仮説 ID: H5 (P1, depends on iter3 結果)
> hypotheses.md: docs/experiment/reinforce/20260527_case6_pfsp/hypotheses.md
> 関連: iter3_plan.md / iter3_result.md / iter3_analysis.md
> スコープ: opponent_mode に `python_v1` (本物 baseline_v1) を追加し、`pure_callback` で seat 1 に注入。
> noop → python_v1 curriculum で 20 iter 学習

## 仮説 (Hypothesis)

H4 (f_hard) は 100 iter 完走で vs full=0.359 → live baseline_v1 30戦は依然 0/30。
原因は **opponent (baseline_jax_full) が本物 baseline_v1 と挙動乖離**:
- baseline_jax_full は近似 rule (target 一致 44% vs 本物 v1)
- 学習時 opponent と eval 時 opponent が違うため、転移しない

→ opponent に **本物の Python baseline_v1 を直接注入**することで train/eval gap を消す。
case_jax で確立した `pure_callback(vmap_method='sequential')` パターンを再利用。

期待結果: 学習後の rl_v6 が live baseline_v1 30戦で勝率 > 30% (現状 0%) を達成。

## 既存コードの現状

- `rollout_jax.py`: opponent_mode は {noop, baseline_jax_lite/full, self_snapshot} の 4 種
- vmap した rollout 内で **本物 Python agent を呼ぶ手段がない** (Python agent は jit/vmap 不能)
- case_jax で同じ問題を `_host_python_v8_action` + `jax.pure_callback(vmap_method='sequential')`
  で解決済 (sha 7e1e5d4)。host 側で seat ごと逐次 Python 実行、device に戻す

## 実装方針

1. **rollout_jax.py** に opponent_mode `python_v1` (=4) / `python_v4` (=5) / `python_v8` (=6) 追加
   - `_host_python_vN_action(state, seat) -> np.ndarray(L, 3)`: state→obs→agent(obs)→pad
   - `_python_vN_opponent_actions(state, player)`: `jax.pure_callback(host_fn, proto, ..., vmap_method='sequential')`
   - `jax.lax.switch` 分岐を 4 → 7 へ拡張
2. **config**: `kaggle_jax_train_pool_v1.yaml`
   - opponent: curriculum (early=noop, late=python_v1, switch_iter=5)
   - iterations 20 / episodes_per_iter 8 / horizon 500
   - 想定 cost ~$1.0 (RunPod A100) or 0 (Kaggle T4x2)
3. **eval**: 学習完了 → `jax_to_torch.py` で weights 変換 → `evaluation/eval_vs_baseline.py --baseline baseline_v1 --episodes 30`

## リスクと対策

- **R1**: host callback で rollout が重くなる (smoke 6.7s → RunPod A100 で 383s/iter = 57x)
  → episodes 32→8 / iter 50→20 で短期化、cost cap 内
- **R2**: 9h Kaggle T4x2 上限 → 学習中断
  → 20 iter で十分短い (1-2h 想定)、上限超過なし
- **R3**: RunPod stockout (実際 20連続)
  → Kaggle free T4x2 で代替実行

## 実行

- run_id: `20260601-164103__feature-agent-pool-learning__d350181__seed0`
- commit: `d3501814`
- launched: 2026-06-01 16:41:25 UTC
- accelerator: Kaggle gpu-t4x2 (free, $0)
- status: 結果は iter4_result.md に追記

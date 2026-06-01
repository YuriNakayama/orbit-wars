# Reinforce/case6 — Real Python baseline_v1 opponent via host callback (iter4) RESULT

> 関連: iter4_plan.md / iter3_result.md / hypotheses.md
> run_id: 20260601-191229__feature-agent-pool-learning__06b6faf__seed0 / commit: 06b6faf3
> case: reinforce_case6_kaggle_jax_train_pool_v1
> 開始: 2026-06-01 19:12:52 UTC / 終了: 進行中 (~5h17 経過時点で RUNNING)
> 環境: Kaggle Kernel gpu-t4x2 (free, $0)

## 経緯 (Kaggle slug 衝突 → 修正)

最初の試行 (run 171440 / pool_v1) と 2 回目 (run 184105 / pool_v8) は
**Kaggle kernel slug 50char 切捨** で同一 slug 化し、後発 push (v8) が
先行 RUNNING (v1) を kill していた:

- 旧 slug 形式: `orbit-wars-{case}-{ts}-{sha}` → 50char 制限で
  両 case が `orbit-wars-reinforce-case6-kaggle-jax-train-pool-v` に化けて衝突
- 修正 (06b6faf3): `ow-{ts}-{sha}-{case}` 形式 + trailing '-' rstrip (76a7e640)
- 修正後の slug 例:
  - pool_v1: `ow-20260601-191229-06b6faf-reinforce-case6-kaggle`
  - pool_v8: `ow-20260601-200421-76a7e64-reinforce-case6-kaggle`

## Kaggle dataset の simulator/jax 同梱バグも修正

最初の 8ea27b44 試行は別の致命バグで死亡 (224s で ERROR):

```
ModuleNotFoundError: No module named 'orbit_wars_jax'
```

`bot/src/gpu/kaggle/dataset/builder.py` の INCLUDE_RELATIVE_PATHS から
`simulator/jax` と `simulator/adapter` が抜けていた (case1/case2 reinforce JAX
学習も同じバグで動かなかったはず)。

修正 (8ea27b44):
- dataset builder に `simulator/jax`, `simulator/adapter` 追加
- kernel template の sys.path/PYTHONPATH 両方に 3 simulator パス追加

## Summary (進行中)

- ✅ slug 衝突 + simulator path 両バグ修正で **pool_v1 が初めて長時間 RUNNING**
- 70 min 経過時点で `kernels_status` API 返答は RUNNING
- pool_v8 (200421) は Kaggle GPU session 上限 2 で push 拒否、pool_v1 完了待ち
- 結果数値 (iter ごとの win/loss, 学習後 live v1 30戦勝率) は完了時追記

## Next steps (完了後)

1. `dev/kaggle pull <run_id> --case reinforce_case6_kaggle_jax_train_pool_v1`
2. `bash bot/pipeline/reinforce/case6/training/post_train_eval.sh <run_id>`
   (jax_to_torch → eval_vs_baseline --baseline baseline_v1 --episodes 30)
3. 勝率改善 (現状 0/30 → ?/30) を確認
4. pool_v8 を起動して PFSP pool 拡張へ

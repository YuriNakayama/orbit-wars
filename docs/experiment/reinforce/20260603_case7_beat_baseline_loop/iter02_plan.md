# case7 「ルールベースに勝つ」ループ — iter02 PLAN

時刻: 2026-06-03 02:09 (cron tick 2 続き)

## 前 iter の結論
- 16-iter モデル vs baseline_v1 = **0/10**。
- resume +追加 PPO (vs baseline_jax_lite, BC なし) は iter 2-8 で win 0-0.25 横ばい、
  **改善せず** → BC なし PPO は強い rule 相手に伸びないと確認 (memory と一致)。

## 本 iter の主施策: BC warm-start (本筋の改善)
memory `project_reinforce_case6_pool_v1_rejected` の「BC warm-start + curriculum」を実行。

### DVC blocker 解決 (記録)
- BC 元 weights (`imitation/case9_per_planet/.../best.pt`, 12.5MB) はローカル未取得だった。
- `dev/dvc pull` は失敗 (wrapper が `bot/` 相対でパス解決 → data symlink と不整合、
  remote 名は `s` でなく `s3`)。
- **解決**: worktree venv の dvc bin を **main repo (`/Users/user/project/orbit-wars`)
  を真の cwd** にして `dvc checkout <.dvc>` → cache から symlink 材化成功。
  使用時は main repo 絶対パスを直接指定 (worktree の `data` symlink は循環気味で不安定)。
- `load_bc_weights_jax` で case7 JAX model に **loaded=133 missing=0** で完全ロード確認。

## 学習設定 (loop_iter02_bc_warmstart.yaml)
- `bc_warmstart.enabled: true`, weights=case9_per_planet best.pt (絶対パス)
- `kl_beta: 0.1` (BC reference を KL anchor として凍結) ← bc_kl が効いていることをログ確認
- shaping: ratio / coef 1.0 (case5 H4 維持)
- curriculum: `noop(2) → baseline_jax_lite` (v1相当)
- 効率: iters=14, episodes=6, horizon=160 (lite は host callback で重いので縮小)

## 期待 / 判定
- BC 起点なら iter 0 から noop に勝ち越し (実際 win=0.5, bc_kl=0.07 で anchor 機能)。
- iter 2+ で baseline_jax_lite に **0.25 超**を出せれば BC warm-start 採用。
- 完走後 → jax_to_torch → 10戦 vs baseline_v1 で **0/10 から動くか**が本判定。

## NEXT (iter03)
- 結果を iter02_result.md に。動けば iters を増やすか 3段 curriculum (noop→lite→full)。
- 動かなければ featurizer parity (train JAX sampling vs eval torch greedy) を疑う。

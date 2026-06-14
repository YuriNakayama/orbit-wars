# Self-Imitation Learning (SIL) 実装設計メモ — ladder12

> ユーザー選択 (2026-06-14): ladder11 完走後の次の1点工夫 = self-imitation
> 根拠: web分析 (Oh+ 2018 SIL) — 稀な strict 勝利 (1/64) を replay buffer +
> clipped advantage 優先で再学習し増幅。逆カリキュラム問題で序盤が学習できない中、
> 偶発的勝利の信号を捨てず終端への勾配を作る。

## 現状アーキテクチャ (確認済み)
- `_run_iter`: collect_rollout_jax → `flat = _flatten_rollout(rollout, gamma, lambda)`
  → `_vmpo_update_jit(model, vp, opt, opt_state, flat, vmpo_cfg, key)`
- `FlatRollout` (ppo_jax.py:61): (N, ...) NamedTuple、~15 特徴配列。
  planet_feats(N,P,F) / candidate_feats(N,P,K,C) 等の大配列を含む。
- V-MPO は既にバッチ内 top-half advantage 重み付け (median 閾値) を実施。
  → SIL の差分価値は **cross-iteration の勝利保持** (現状は当該 iter のバッチのみ)。

## SIL 設計 (最小・config-gated)
1. **勝利リプレイバッファ**: 固定容量 C=2048 timestep の循環バッファ (FlatRollout
   と同型の固定形状 pytree)。`rollout.episode_outcomes > 0` の勝利エピソードの
   flat timestep のみを scatter で追加 (count を carry、満杯で FIFO 上書き)。
2. **挿入点**: `flat = _flatten_rollout(...)` 後、`_vmpo_update_jit` 前。
   SIL有効時は buffer から優先サンプリング (priority = clipped advantage
   (R - V)+) した B_sil 件を flat に concat してから update。
3. **config**: `training.sil.enabled` (bool) / `sil.capacity` (2048) /
   `sil.sample_size` (256) / `sil.min_buffer` (256, これ未満は concat しない)。
   enabled=false で完全 no-op (既存と bit-identical) → A/B 規律。
4. **メモリ**: candidate_feats(P=48,K=64,C~30) が支配的。2048 timestep で
   ~2048*48*64*30*4B ≈ 1.5GB。A100 80GB/RTX4090 24GB で許容範囲だが要確認。
   不足時は capacity を 1024 に。

## 実装ステップ
1. ppo_jax.py に `WinReplayBuffer` (FlatRollout同型 + count) と
   `sil_buffer_init` / `sil_buffer_add` / `sil_buffer_sample` (全て jit可).
2. train_jax.py `_run_iter` に SIL 分岐 (config gated)。buffer は train ループの
   carry として _run_iter の外で保持。
3. vmpo_jax の update は flat をそのまま使うので、concat 済み flat を渡すだけ。
4. smoke test: ローカルで sil.enabled=true、2-3 iter、buffer add/sample が
   shape 整合・no-op (enabled=false) bit-identical を確認。
5. ladder12.yaml: ladder9 pool + ep192 + sil.enabled=true、resume=ladder11 best.pt。

## リスク
- メモリ (上記)。- jit recompile (buffer の固定形状を厳守)。
- 勝利が稀すぎて buffer が min_buffer に達しない → SIL が発火しない可能性
  (strict 勝利 1/64 だが、pool には勝てる弱段もあるので self/弱T0 の勝利で充填される)。

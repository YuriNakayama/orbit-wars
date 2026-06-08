# 設計: f_var matchmaking + held-out eval + Elo

## 1. matchmaking ロジック (PFSP f_var)

### 研究的根拠
- AlphaStar の main agent は **PFSP** で過去self pool から勝率比例サンプル。
- 重み関数: `f_hard(x)=(1−x)^p` (勝てない相手を厚く) / `f_var(x)=x(1−x)` (同レベルを厚く)。
  `x = P(agent beats opponent)`。
- survey 2024: **FSP は弱すぎ相手に interaction を浪費**する → フラット mix が非効率な理由。
- 合意: 強すぎ (勾配消失) も弱すぎ (浪費) もダメ。**~0.5 近傍を厚く**するのが最良。

### 実装 (`_PrioritizedOpponentSelector`, train_jax.py)
```python
def _pfsp_weight(x, p, mode):
    if mode == "f_var": return (x*(1-x))**p   # ~0.5 近傍を厚く
    return (1-x)**p                            # f_hard: 強敵を厚く
```
- エントリ = `[baseline_jax_full] + snapshot pool`。各エントリは agent のそのエントリへの
  **win_ema** (= x, init 0.5) を保持。
- sample: `weight_i = f_var(x_i)` → 正規化 → 確率サンプル。rollout後 `win_ema` を EMA 更新。
- config: `opponent_pool.priority: f_var` (既存の `f_hard` と共存)。

### 選択フローの数値例
```
pool: full(x=0.2) snap_A(x=0.45) snap_B(x=0.55) lite(x=0.7)
f_var=x(1-x): 0.16   0.2475       0.2475        0.21
→ snap_A/snap_B (実力近傍) が最も選ばれる。x→0 や x→1 の相手は薄くなる。
```

## 2. 進捗測定 (matchmaking 非依存)

### なぜ必要か
f_var で match を ~0.5 に保つと、**per-iter 勝率は設計上ほぼ一定**になり
「勝率↑ = 強くなった」が成立しない。研究でも *「adversarial game の cumulative
reward / 対戦勝率は相手の強さにしか依存せず進捗指標にならない」*。
→ 固定基準に対する絶対量で測る。

### held-out eval (`_heldout_eval`)
- **固定相手** (`heldout_eval.opponent`, 既定 baseline_jax_full) と
  **固定 seed** (`heldout_eval.seed`) で N iter 毎 (`heldout_eval.every`) に
  episodes 戦の rollout-only (PPO 更新なし)。
- 固定なので iter 間で **比較可能な絶対進捗曲線**。`row["heldout_win"]` に記録。

### Elo (`_elo_update`)
- 固定 held-out 相手を **固定レーティング** (`ref_elo`, 既定1500) のアンカーとし、
  held-out 勝率 `s` で agent_elo を更新:
  `expected = 1/(1+10^((ref−agent)/400))` ; `agent += k·(s − expected)`。
- agent_elo は **絶対 skill 曲線**。`row["agent_elo"]` に記録。

### entropy (健全性)
- 既存 metrics の `entropy`。**<10 に落ちたら policy collapse** の兆候 (H1-H4 で観測)。

## 3. 観測の読み方 (正常判定)
```
進捗(主):   heldout_win ↑     ← これが上がれば本当に強くなっている
進捗(副):   agent_elo  ↑
健全性:     entropy が崩壊(<10)していない / value_loss 発散なし / approx_kl ~ target
学習信号:   match win_rate が ~0.5 帯  ← f_var が効いている証 (0/1 張り付き = match 機能不全)
```
可視化 `plot_metrics.py` 3×3: row0=進捗(held-out/Elo/match), row1=健全性, row2=context。

## 4. config (`h5_fvar_heldout.yaml`)
- `priority: f_var`, `heldout_eval.every: 5`, opponent=baseline_jax_full, 80 iter。
- pool: self_snapshot + full mix (`late_full_prob: 0.4`), cap 6, snapshot_every 8。

## 5. 評価プロトコル
1. GPU 80 iter (~35min)。中間 ckpt+metrics は S3 (crash-safe)。
2. **held-out 勝率と Elo が単調 ↑** すれば matchmaking が機能 (= 実力が上がっている)。
3. entropy が崩壊しなければ健全。
4. 最終 ckpt を外部 paired 30戦 (vs baseline_v8) で確認。

# rulebase-to-jax — 実装方針 (失敗回避戦略)

## なぜ過去の JAX 化は「1勝もできなかった」のか (根本原因)

memory `project_reinforce_case6_live_eval` と既存コード (`rollout_jax.py:71-76`) が同じ結論を記録:

> case6 PFSP was 0/30 vs live v1 because the in-JAX baseline_jax_lite/full **approximate but don't match** the real rule.

**失敗の機序は速度でなく parity (忠実度) の崩壊**:

```
JAX port の score がわずかに違う (float32 reduction 順序差)
   → argmax の tie-break がズレる
   → 別の target/mission を選ぶ
   → 1 手違えば盤面が分岐
   → 以降まったく別ゲーム
   → 本物に対して 0 勝
```

ゲーム agent は**逐次決定の連鎖**なので、1 手の差が指数的に発散する。NN 推論 (1 回 forward, tol 1e-5 で十分) とは根本的に難易度が違う。「だいたい合っている JAX port」は opponent として**無価値かつ有害**。

### Web 調査で裏付けた失敗要因 (2026-06)

| 要因 | 内容 | 出典 |
|------|------|------|
| float32 既定 | JAX は float64 を float32 に truncate。Python(numpy)は float64。**そもそも精度が違う** | [Default dtypes and X64](https://docs.jax.dev/en/latest/default_dtypes.html) |
| reduction 順序差 | `sum`/`mean` の累積順が numpy と違い float32 で誤差。score の微差を生む | [jax#6624](https://github.com/jax-ml/jax/discussions/6624), [jax#641](https://github.com/jax-ml/jax/issues/641) |
| argmax tie-break | score 同値/微差で argmax が別 index → 別 action | [Sharp Bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) |
| GPU 演算差 | matmul/sqrt 等で CPU と GPU でも結果が違う | [jax#18123](https://github.com/jax-ml/jax/issues/18123) |

## 失敗回避の 5 原則

### 原則 1: 「本物 gold-standard」を常に手元に置く (退避路を断たない)

既存の `OPPONENT_PYTHON_V1/V4/V8` (`jax.pure_callback` で本物 Python を呼ぶ) を**捨てない**。
- JAX port は**速度のための最適化**であり、本物 callback opponent の**置換ではなく追加**。
- 各 port は「対応する本物 callback opponent と **300戦して互角**」を合格条件にする (NFR の最終関所)。
- ⚠️ Web 調査: `pure_callback` は XLA↔Python の round-trip でオーバーヘッド大、`vmap_method='sequential'` で逐次化されスループット低下 ([External callbacks](https://docs.jax.dev/en/latest/external-callbacks.html))。だが**正しさは保証される**。JAX port が parity 達成するまでは本物 callback を使い続けられる体制を維持。

### 原則 2: float64 (x64) で parity を取り、port を確定してから float32 へ

- **parity test は `jax.config.update("jax_enable_x64", True)` で float64 を有効化**して実行 ([X64 flag](https://docs.jax.dev/en/latest/default_dtypes.html))。Python(numpy float64) と土俵を揃え、**アルゴリズム移植のバグ**(精度差でなくロジック差)を切り分ける。
- float64 で action 完全一致 → アルゴリズムは正しい。
- その後 float32 に落として再度一致率を測定。落ちる分は「精度差由来」と特定でき、tie-break 統一・int 化・閾値処理で吸収する。
- **本番 self-play は float32** (GPU 速度優先) だが、parity は x64 で先に保証してから float32 化の劣化を管理する 2 段検証。

### 原則 3: tie-break と境界条件を Python 側に合わせて明示実装

- argmax の同値 tie は **index 最小**等のルールを Python/JAX 両方で明示。JAX 側は `score - eps * index` 等で決定論化。
- Python 版の `>` / `>=` / `int()` 切り捨て / `round` を**逐一同じ演算**で再現。「だいたい同じ式」は禁止、**演算子レベルで一致**させる。
- 浮動小数比較は Python 版の閾値・順序をそのまま写す。

### 原則 4: ボトムアップ差分テスト (関数単位で本物と突き合わせ)

agent 全体をいきなり比較せず、**下位関数から順に本物と一致を確認**:
```
geometry → physics → safety → worldmodel(ledger) → 各 mission score → 統合 argmax → agent
```
各層で本物 Python の同名関数と x64 で完全一致を確認してから上位へ。どこで発散したか即座に切り分く ([差分テストの考え方])。これは `python-to-jax` skill の TDD (parity test 先書き) と完全に整合。

### 原則 5: action 一致率 100% を合格条件にする (許容を緩めない)

- ⚠️ 既存 case2 parity test は「12 step 中 1 mismatch 許容 / angle tol 1e-2」= **緩すぎる**。この緩さが乖離の温床。
- 新 port は **同一 obs 大量サンプルで action 完全一致 (100%)** を assert。許容するのは float 位置の rtol 1e-5 のみで、**選択 (from_pid / mission / ships int) は完全一致必須**。
- 一致率が 100% 未満なら**その obs を replay して原因特定**し、100% になるまで port を直す。緩めない。

## opponent dispatcher の修正 (Web 調査反映)

既存 `rollout_jax.py:399-416` は `jax.lax.switch(opponent_mode, [...])` を使用。Web 調査 ([jax#20916](https://github.com/jax-ml/jax/discussions/20916)) で vmap 下の switch は**全 branch 実行**と判明 → 全 opponent (本物 callback 含む) が毎 step 評価され**本物 callback の逐次 round-trip が全 mode で発火**する深刻な無駄。

→ 方針: JAX port 群は **strategy_id を data で受ける分岐レス共通関数** (architecture 通り) に統一。本物 callback opponent は parity 検証/最終確認用に分離し、訓練時は switch でなく「選んだ 1 つだけ計算」する構造へ。dispatcher の switch 依存を解消する (Step 8 で対応)。

## 実装フローへの反映 (改訂)

```
各 port (Step 1–7) で:
  1. parity test を x64 で先書き (本物 Python の同名関数 vs JAX)
  2. ボトムアップに下位関数から完全一致を確認 (原則4)
  3. tie-break/境界を Python に合わせ明示 (原則3)
  4. x64 で agent action 100% 一致 → float32 で再測定し劣化を吸収 (原則2)
  5. 本物 callback opponent と 300戦 互角を確認 (原則1, 原則5)
  6. 不一致 obs は replay で原因特定、緩めず直す
本物 pure_callback opponent は全工程で温存 (退避路, 原則1)
```

## まとめ: 1 勝もできない事態の回避策

| 過去の失敗 | 今回の対策 |
|-----------|-----------|
| JAX 近似 port が本物に 0/30 | full parity (action 100%一致) を合格条件、緩い許容を禁止 (原則5) |
| float32 精度差で score がズレ tie-break 発散 | x64 で先に parity 保証 → float32 劣化を管理 (原則2) |
| どこで発散したか不明 | ボトムアップ差分テスト (原則4) |
| 近似 port を本物の置換にした | 本物 callback を温存し JAX port は追加扱い、互角確認まで退避路維持 (原則1) |
| lax.switch で全 opponent 評価 | strategy_id data 化 + 選んだ1つだけ計算 (dispatcher 修正) |

# env reset の JAX 化計画 (web-research ベース)

> 作成日: 2026-06-09 (改訂)
> 背景: rollout jit 化 (W7) 後も GPU util ~0%/80%。残存ボトルネックは
> `collect_rollout_jax` が毎 iter host で呼ぶ env reset (`simulator/jax/orbit_wars_jax/reset.py`)。
> 本計画は reset を `jax.random` + jit/vmap でネイティブ JAX 化し、reset を rollout の
> jit/vmap グラフ内に取り込む方針 (web で確立された JAX-RL env パターンに準拠)。

## 現状 reset の JAX 化を阻む3要素

1. **可変長 rejection sampling** — `generate_planets` は `for _ in range(5000)` + `while len(planets)<...`
   で条件を満たすまで惑星を draw。`comet_gen` も `for _ in range(300)`。
2. **動的 list 構築** — 惑星/コメット個数が seed 依存で `append` (固定 shape でない)。
3. **host RNG** — `random.Random(seed)` (Python 標準) + `np` 配列。

## web research の知見 (採用する JAX パターン)

| 課題 | 確立された JAX イディオム | 出典 |
|---|---|---|
| 可変長 rejection sampling | **`lax.while_loop`** で carry=`(rng, fixed_buffer, accept_count)`。受理サンプルを**事前確保した固定長バッファ**へ `buf.at[i].set(x)`、accept 時のみ `i+=1`。終了条件 `i < num_target` | jax #11219 |
| vmap 下の per-instance 乱数 | `keys = jax.random.split(rng, B)` で**事前分割**し `vmap(reset, in_axes=(0,))` | jax #11219 |
| 可変個エンティティの固定 shape 化 | **MAX_PLANETS 固定確保 + valid マスク**で表現 (既に EnvState は valid マスク方式)。vmap 内で branch 不可 → **両分岐計算 + `jnp.where` select** | gymnax / jumanji / Craftax |
| 手続き生成 reset | reset を**純関数 (key→state)** にし jit/vmap。Craftax は手続き生成ワールドをこれで実現 | Craftax, PureJaxRL |

→ rejection sampling は **JAX で実装可能** (`lax.while_loop` + 固定バッファ + マスク)。
「JAX 化不可」は誤り。正しくは「**固定長バッファ + while_loop + マスク + jax.random** に書き換える」。

## 方針 (reset を JAX-native 化)

### Step 1: RNG を jax.random に置換
- `random.Random(seed)` → `jax.random.PRNGKey(seed)` + 明示的 split。
- `rng.uniform(a,b)` → `jax.random.uniform(k,minval=a,maxval=b)`、`rng.randint` → `jax.random.randint`。
- **注意**: vendor `random.Random` の byte-stream parity は**放棄する**(jax.random は別系列)。
  これは seed→惑星配置の対応が変わることを意味するが、本実験の held-out は「同じ env 分布で
  生成された固定相手 (baseline_jax_full) と対戦」なので、**reset 分布が自己無撞着なら parity 喪失は問題ない**
  (vendor の正確な配置を再現する要件はない — 学習・評価とも新 reset を使えばよい)。
  ※ 既存の本物 vendor リプレイとの突合が必要な箇所のみ旧 reset を残す。

### Step 2: generate_planets を while_loop + 固定バッファ化
- `MAX_PLANETS` 個の固定 shape 配列を確保。
- `lax.while_loop`: carry=`(key, planet_buf, valid_mask, count, attempts)`。各反復で1惑星候補を draw、
  受理条件 (重なり判定等) を満たせば `buf.at[count].set(...)` + `valid_mask.at[count].set(True)` + `count+=1`。
  終了条件 = `count >= target or attempts >= max_attempts`。
- 重なり判定など「既存惑星との距離」チェックは固定バッファ + valid マスクで vectorize。
- num_q1 (グループ数) など seed 依存の数も jax.random.randint で引き、target を動的carryで持つ。

### Step 3: comet 生成 (precompute_all_comets) も同様に固定 shape + while_loop 化
- `for _ in range(300)` の rejection を while_loop に。MAX_COMETS 固定 + path_len マスク。

### Step 4: home 割当 / EnvState 構築を jnp + lax.cond/where で
- `num_agents==2/4` の分岐は vmap 内では両方計算 + select、または num_agents を static 引数に。
- home_group 選択は jax.random.randint。index 代入は `.at[].set`。

### Step 5: reset を rollout の jit/vmap グラフに取り込む
- `collect_rollout_jax`: `init_states = [reset(seed+i) for i]` (host loop) を廃止し、
  **`reset_jax(key)` を vmap して device 上で B 本生成** → そのまま `_rollout_one_env` に渡す。
- これで host reset が消え、reset+rollout が単一 XLA 実行に。GPU が連続稼働。

## 検証 (parity と速度)

1. **数値 parity test**: `reset_jax(key)` を vmap で B 本生成し、(a) 各 state が `validate_state` を通る、
   (b) 惑星個数分布 / 初期 ships=10 home / comet 個数が旧 reset と統計的に整合 (分布一致でよい、
   byte-equal は要求しない) を pytest で確認。
2. **速度検証**: iter30 run を再実行し rollout_secs と GPU util を W7 (jit のみ) と比較。
   reset が device 化されれば GPU util が上がり rollout_secs が更に短縮するはず。
3. **学習不変性**: win/reward/held-out が reset-JAX 化前と統計的に大きく変わらない (初期 state 分布同等)。

## 段階リリース
- まず `simulator/jax/orbit_wars_jax/reset_jax.py` を**新規追加** (旧 `reset.py` は残す)。
- case8 の rollout だけ新 reset に切替えて検証 → 良好なら他 case へ展開。
- 旧 reset (vendor parity 版) は vendor リプレイ突合用に保持。

## リスク
- rejection sampling の while_loop 化は重なり判定ロジックの忠実な vectorize が要 (バグると配置が偏る)。
- max_attempts 上限で稀に target 個数に届かない seed → valid マスクで吸収 (惑星数が少し変動)。
- vendor parity 放棄の影響: 本実験 (self-play + in-JAX 固定相手) には無いが、vendor リプレイ解析や
  Kaggle 本番 env との突合には旧 reset が要る点を明記。

## 参考 (web)
- JAX rejection sampling under vmap/jit (lax.while_loop + fixed buffer): https://github.com/jax-ml/jax/discussions/11219 , https://github.com/jax-ml/jax/discussions/5028
- lax.while_loop 解説: https://apxml.com/courses/advanced-jax/chapter-1-advanced-jax-transformations-control-flow/looping-lax-while-loop
- JAX-RL env reset パターン (pure fn key→state, vmap, branch→select): gymnax https://github.com/RobertTLange/gymnax , pgx https://github.com/sotetsuk/pgx , Jumanji https://arxiv.org/pdf/2306.09884 , Craftax (手続き生成) https://arxiv.org/pdf/2402.16801 , PureJaxRL https://chrislu.page/blog/meta-disco/

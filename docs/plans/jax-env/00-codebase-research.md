# JAX Env — Phase 0: Codebase Research

> 作成日: 2026-05-18
> 目的: 公式 simulator/python/orbit_wars_vendor/orbit_wars.py の JAX 化可能性とスコープを精査
> 後続: `01-design.md` で JAX pytree 構造と parity 戦略を確定

## 公式 sim の構造 (793 行、単一ファイル)

### モジュール責務マップ

| セクション | 行 | 責務 | JAX 化難易度 |
|---|---|---|---|
| 定数 (BOARD_SIZE 等) | 17-27 | 物理定数 | 自明 |
| `distance` / `point_to_segment_distance` | 30-43 | 幾何 | **容易** (pure function) |
| `swept_pair_hit` | 46-66 | 連続衝突判定 (2 次方程式) | **容易** (pure function) |
| `generate_planets` | 69-190 | rejection sampling で 5-10 グループ生成 | **困難** (while + 棄却) |
| `generate_comet_paths` | 193-333 | 楕円軌道 5000 点 + rejection | **困難** (棄却 + 動的長) |
| `interpreter` (= env step) | 336-698 | reset + step 統合関数 | **複雑** (動的 list + dict + namedtuple) |
| renderer / agents | 700-793 | 表示・参考実装 | スコープ外 |

### Step ループの内部構造 (interpreter 関数)

1. **初期化分岐** (343-395): obs.planets が空なら reset、generate_planets 呼出、home 割当
2. **expired comet 削除** (401-420): comet の `path_index` が path 長を超えたら planet/initial_planets/comets から削除
3. **comet spawn** (422-462): step+1 ∈ {50,150,250,350,450} で generate_comet_paths
4. **fleet launch** (464-497): 各 agent の action を planet[5] (ships) から差し引いて fleets に追加
5. **production** (499-502): 所有 planet の ships += production
6. **planet 移動先計算** (504-548): 静止/回転/comet で planet_paths[pid] = (old, new, check)
7. **fleet 移動 + 衝突** (550-590): 各 fleet で `swept_pair_hit` を planets 全部にチェック、最初に当たった planet で break → combat_lists[pid].append + remove fleet。範囲外/sun 通過も remove
8. **planet 移動適用** (593-596): planet_paths[pid][1] を planet[2:4] にコミット
9. **expired comet 削除 (再)** (599-612)
10. **combat resolution** (617-655): combat_lists[pid] 毎に owner 別 ships を集計、勝者決定 (top - second = survivor)
11. **observation broadcast** (657-663): obs0 を全 agent に複製
12. **termination** (665-697): step >= episodeSteps-2 / alive_players <= 1 で終了、スコア計算

### 動的 size の対象

| エンティティ | 範囲 | 変動要因 |
|---|---|---|
| **planets** | 12-40 個 (初期) → +4 個/comet (50,150,250,350,450 step) → expired で減少 | comet spawn / expire |
| **fleets** | 0 〜 ~数百 | launch / 衝突 / 範囲外 / sun |
| **comets (group)** | 0-5 | 各 step での spawn 成否 |

## JAX 化の主要ハードル

### 1. 動的 list (planets/fleets) → 固定 shape 化必須

jit + vmap は **動的 size を許可しない**。解決策:
- `MAX_PLANETS = 48` (12 base + 5 comet spawn × 4 quadrant + 余裕 16)
- `MAX_FLEETS = 256` (経験的上限、超過時は新規発射を drop)
- `valid_mask` 配列で「実在」フラグ管理
- 削除は in-place で `valid_mask[i] = False`、append は free slot search

### 2. Rejection sampling → CPU 実行に隔離

`generate_planets` / `generate_comet_paths` は while + 5000 試行で棄却するため jit 不可。
**reset 関数を CPU (eager numpy + python random) で実行** し、結果を jax.Array に変換して GPU に転送。
これは 1 episode 開始時の 1 回だけなので overhead 許容。

### 3. RNG 系列の再現性

公式 sim は `random.Random(seed)` で planet 生成、`random.Random(f"orbit_wars-comet-{seed}-{step+1}")` で comet 生成。
**reset を CPU で行えば公式 sim と同じ `random` モジュールが使えるため RNG 完全一致可能**。
step 内では非決定論なし (action 結果は決定論的)、PRNG 不要。

### 4. swept_pair_hit による O(F × P) 衝突

各 fleet が「最初にヒットした planet」で break → JAX では break 不可。
解決: 全 (fleet, planet) ペアで hit 判定 → `argmax` で最初の hit を取る。
shape: `(F, P)` の bool 配列 → `jnp.argmax(hits, axis=1)`。O(F·P) は許容 (P≤48, F≤256 で 12k 要素)。

### 5. combat resolution の per-planet aggregation

combat_lists[pid] は planet 毎の fleet リスト。JAX では:
- `fleet_target_planet[F]`: 各 fleet がヒットした planet id (-1 = noop)
- `segment_sum` で planet × owner 別 ships を集計 → `(P, num_agents)` テンソル
- 各 planet で `sort_descending(top, second)` → survivor 計算

### 6. termination + score

`alive_players <= 1` 判定: `(P, num_agents)` テンソルから owner 集合 → `bool any` で生存判定。
step >= 498 と OR、reward は max-score 比較で `+1/-1`。すべて pure jax。

## 上限定数の確定 (固定 shape)

| 定数 | 値 | 根拠 |
|---|---|---|
| `NUM_AGENTS` | 2 | 1v1 のみサポート (FFA は scope 外、後続で対応) |
| `BOARD_SIZE` | 100.0 | 公式定数 |
| `MAX_PLANET_GROUPS` | 10 | 公式上限 |
| `MAX_PLANETS` | **48** | 10 group × 4 quadrant + 5 comet spawn × 4 |
| `MAX_FLEETS` | **256** | 経験的、500 turn で fleet 上限ヒット率を後で確認 |
| `EPISODE_STEPS` | 500 | 公式 default |
| `MAX_COMET_PATH_LEN` | 40 | 公式 sim の `5 <= len(visible) <= 40` |

## Parity 戦略

### 1:1 一致を目指す範囲

| 要素 | 一致レベル | 検証方法 |
|---|---|---|
| **planet 初期配置** | **完全一致** (RNG 同一、CPU reset) | seed×100、planets[:,2:4] (xy) と planets[:,4] (radius) の bit 比較 |
| **fleet 移動** | float32 精度 | abs(jax_xy - py_xy) < 1e-5 |
| **comet path** | float32 精度 | 同上 |
| **combat 結果** | **完全一致** (整数) | 公式 sim と勝者 owner / 残存 ships が完全一致 |
| **action 結果 (step 後 state)** | float32 + 整数 | seed×action 系列 100 step で各フィールド比較 |
| **termination + reward** | **完全一致** | scores 比較 |

### 一致が原理的に困難な点

| 困難 | 対処 |
|---|---|
| `MAX_FLEETS` 超過時の drop | 公式 sim は無制限。超過した seed は parity テストから除外 (実際の rollout で 256 超過する頻度を別途測定) |
| float32 累積誤差 | 各 step で abs diff を記録、tolerance 段階的決定 |
| RNG forking (planet generation 内部の 5 種 randint) | reset を Python random で実装すれば一致 |

## 既存資産との関係

- `simulator/rust/` (3310 行、parity 取れている前提) は今回使わない。将来 JAX env が parity 取れなかった場合の保険として保持
- `simulator/python/orbit_wars_vendor/` は parity テストの reference として常に保持
- BC 学習データ (case9_per_planet) は公式 sim ベースで生成済、JAX env で 1:1 parity が取れれば BC 重み再学習不要

## 次フェーズ (01-design.md) で確定する内容

1. JAX pytree 構造 (`EnvState`, `Action`, `Observation`)
2. `jax_env/` モジュール分割 (geometry / planet_gen / comet_gen / step / combat / parity)
3. ベンチマーク戦略 (vmap 16 vs 32 vs 64 env、CPU vs GPU)
4. rollout 統合方針 (batch inference、reset trajectory length 揃え)

## リスクと未確定事項

- **MAX_FLEETS=256 で実 rollout の頻度** 未測定。iter1 rollout の `planet_feats.shape[0]` から逆算可能、Phase B 前に確認
- **NUM_AGENTS=2 固定の妥当性**: case1 は 1v1 baseline_v1 学習なので OK。4-player FFA は scope 外
- **GPU 利用判断**: env が小さい場合 (P=48, F=256) GPU launch overhead が勝つ可能性。CPU JAX で十分なら GPU 不要、ベンチで決定

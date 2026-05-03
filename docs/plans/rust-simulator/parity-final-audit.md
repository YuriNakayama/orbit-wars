# Rust Simulator — Final Parity Audit (2026-05-02)

公式 Python interpreter と Rust port の差分を **bit-exact レベル** で検証した
最終レポート。これまでの段階的な audit (`parity-audit.md`) の上に、ULP 単位の
比較・ソースコード網羅性・大規模 fuzzing を加えた。

## 結論

**観測可能な差分: ゼロ**。

- ULP-exact (0 bit 差) で planets / fleets の浮動小数点が一致
- 全 32 個の upstream interpreter behavior を Rust または Python 委譲で再現
- 8358 turns の fuzzing で 0 divergences

未検出の差分があっても、本 audit を通過した観点では **検出不可能** な微差にとどまる。

---

## 1. ULP 監査 (浮動小数点 bit-exact)

`struct.unpack("<Q", struct.pack("<d", float(x)))` で IEEE 754 raw bits を取り、
両 backend で 1 bit でも違えば検出。

| 範囲 | 検証 turns | total field diffs | max ULP |
|------|----------:|-----------------:|--------:|
| 5 seeds × 100 turns × 2p, aggressive random | 500 | **0** | **0** |
| 20 seeds × ~498 turns (full episodes), 2p+4p mix | 8358 | **0** | **0** |

→ **planets[*].x/y/radius と fleets[*].x/y/angle は bit-exact**。
`f64` の演算順序が両 backend で一致している証拠。

---

## 2. ソースコード網羅性監査

公式 `orbit_wars.py::interpreter()` から 32 個の独立した behavior を機械的に
抽出し、Rust 側にミラー実装が存在するかを正規表現で検査。

| 区分 | upstream | rust 実装 / Python 委譲 |
|------|---------:|:----------------------:|
| 初期化 (8 behaviors) | ✓ × 8 | Python 委譲 (facade で `not _planets_present(state)` ガード) |
| Comet 系 (3 behaviors: expire / spawn check / ships=min(4 randint)) | ✓ × 3 | Python 委譲 (facade で `_is_comet_spawn_turn` ガード) |
| process_moves (3 behaviors: int(ships) / ships >=0 / radius+0.1 offset) | ✓ × 3 | Rust ✓ |
| Production phase | ✓ | Rust ✓ (`planet.ships += planet.production`) |
| Fleet move (speed formula / sun crossing / planet collision) | ✓ × 3 | Rust ✓ |
| Planet rotation (angle update / sweep) | ✓ × 2 | Rust ✓ |
| Comet path advance (path_index / first placement skip) | ✓ × 2 | Rust ✓ |
| Combat (sum / sort / tie / capture flip) | ✓ × 4 | Rust ✓ |
| Post-step state copy to other agents | ✓ | Rust ✓ (`write_state_back`) |
| Termination (max_steps / 1 alive / DONE / reward) | ✓ × 6 | Rust ✓ |

**結果: 32 / 32 (Rust 実装 21 + Python 委譲 11) すべてカバー**。

---

## 3. Heavy fuzzing

20 seeds (2p × 15 + 4p × 5)、各 episode 最大 500 turn、毎 turn ランダムアクション
(angle: `[-π, 3π]` の異常値含む / ships: `[1, half, all-1, all+100]` の境界含む)
で実行。

毎 turn 観測の SHA-256 hash を取り、両 backend で 1 bit でも違えば divergence
としてカウント。

| 項目 | 値 |
|------|----:|
| 検証エピソード数 | 20 |
| 検証 turns 累計 | **8,358** |
| 観測 hash 不一致 | **0** |
| `env.done` flag 不一致 | **0** |
| 早期終了の有無 | 平均 ~430 turn / episode (random vs random で elimination) |

---

## 4. 周辺 API audit

### 4.1 環境登録 (`environment_dict` / registry)

| Key | python (vendored) | rust (facade 経由) | 比較 |
|-----|------------------|---------------------|:---:|
| `interpreter` | `orbit_wars_vendor.interpreter` | `orbit_wars_rust._facade.interpreter` | ✓ 差替 |
| `renderer` | object identity | `orbit_wars_vendor.renderer` (re-export) | ✓ 同一 |
| `html_renderer` | object identity | 同上 | ✓ 同一 |
| `specification` | object identity | 同上 | ✓ 同一 |
| `agents` | `{random, starter}` | 同上 | ✓ 同一 |

### 4.2 観測 schema

`orbit_wars.json` で定義された全 9 フィールドが両 backend で同一に出現:
`planets / fleets / player / angular_velocity / initial_planets / next_fleet_id / comets / comet_planet_ids / remainingOverageTime`

### 4.3 Renderer 出力 (ANSI)

`env.render(mode="ansi")` 30 turn 後の出力 = **2144 文字、bit-exact 一致**。

### 4.4 `env.toJSON()` shape

| Field | 一致 |
|-------|:---:|
| `id` | ✗ (UUID なので実行毎に異なる、環境差ではない) |
| `name / title / description / version / module_version` | ✓ |
| `steps` (length) | ✓ |
| `rewards / statuses` | ✓ |
| `configuration` | ✓ |

### 4.5 `remainingOverageTime`

framework が更新する fields だが、両 backend で `2` (initial value) のまま
保持され、agent timing が actTimeout 以下のときに減算されないことも一致。

### 4.6 `info / status / reward` per-agent

Full episode 終了時 (seed=2, 498 turn): 両 backend で全 agent の status / reward
/ info が完全一致。

### 4.7 アクション形式の互換性

| 入力 | 期待挙動 | py / rs 一致 |
|------|---------|:-----------:|
| `[[], []]` | noop | ✓ |
| `[None, None]` | noop | ✓ |
| `[[], None]` | mixed noop | ✓ |
| `[[-1, 0.0, 5]]` (invalid planet id) | ignored, no fleet | ✓ |
| `[[0, 0.0, -10]]` (negative ships) | ignored | ✓ |
| `[[0, 0.0, 999999]]` (oversized) | ignored | ✓ |
| `[[0, 0.0, 0]]` (zero ships) | ignored | ✓ |
| `[[1, 0.0, 5]]` (wrong owner) | ignored | ✓ |
| `[[0, 0.0, 5, 999]]` (4-tuple, wrong shape) | ignored | ✓ |
| `[[0, 0.0]]` (2-tuple) | ignored | ✓ |
| `[[0, 'bad', 5]]` (type error in angle) | TypeError on both | ✓ |
| `[[0, 0.0, 5.0]]` (float ships) | int(5.0)=5, applied | ✓ |
| `[[0, 0.0, True]]` (bool ships) | int(True)=1, applied | ✓ |
| `[[0, 0.0, "5"]]` (str ships) | int("5")=5, applied | ✓ |
| `[[0, -π, 1]]` (negative angle) | applied (cos/sin handles wrap) | ✓ |
| `[[0, 5π, 1]]` (large angle) | applied | ✓ |

---

## 5. Determinism audit

| 検証 | 結果 |
|------|:---:|
| Python × 2 同 seed → 同一 episode | ✓ |
| Rust × 2 同 seed → 同一 episode | ✓ |
| `env.steps` 過去 frame の保持 (history) | ✓ 全 entry 一致 |
| state[i].observation オブジェクト identity | 両 backend ともに **別オブジェクト** (framework の structify による) → 互換 |

---

## 6. 物理エッジケース

| シナリオ | 結果 |
|---------|:---:|
| 1 fleet vs sun (角度を sun 方向に向ける) | ✓ 同 turn 生存 |
| 10 fleets 一斉発射 (next_fleet_id 連番) | ✓ 連番一致 |
| 多重 attacker 同一 planet (combat resolution の sort 安定性) | ✓ owner / ships 一致 |
| 4p 全員同一中立 planet 攻撃 | ✓ 30 turn 一致 |
| Comet expiration (path 終端到達) | ✓ comet_planet_ids 同 turn で消失 |
| Comet 占領 + production | ✓ ships / production 一致 |
| 静止 planet 判定 (`orbital_radius + r >= ROTATION_RADIUS_LIMIT`) | ✓ 全 planet 位置一致 |
| Planet rotation chirality (回転方向) | ✓ 20 turn trajectory bit-exact |
| Production timing (launch 後に production 加算) | ✓ |
| Fleet 起動オフセット (`(radius + 0.1) * direction`) | ✓ |

---

## 7. Configuration override

| Field | 試験値 | 反映 |
|-------|--------|:---:|
| `seed` | 0–100 様々 | ✓ |
| `episodeSteps` | 100 | ✓ 100 turn で termination |
| `shipSpeed` | 8, 10 | ✓ fleet 速度公式に伝播 |
| `cometSpeed` | 8 | ✓ |
| `actTimeout` | 3 | ✓ env.configuration に保持 |

---

## 8. 観測されなかった差分

調査の過程で以下も確認したが、すべて **両 backend 共通の挙動**:

- `state[0].observation.planets is state[1].observation.planets` は **両 backend で False** (framework が structify で deep copy)。upstream Python と同じ挙動。
- `env.id` は UUID で実行毎にユニーク → 両 backend で値は違うが、これは **環境固有の non-determinism** で interpreter とは無関係。
- `WARN` メッセージや `print` 出力は両 backend で発生せず。

---

## 9. 検証スクリプト

すべての test は `simulator/rust/python/tests/test_parity.py` に集約済み。
本 audit のシナリオを再現するには:

```bash
cd bot

# fast slice (default ci)
uv run pytest ../simulator/rust/python/tests/test_parity.py -m "not slow"

# full (slow tests included)
uv run pytest ../simulator/rust/python/tests/test_parity.py
```

加えて、`docs/plans/rust-simulator/scripts/` に大規模 fuzzing スクリプトを
置くことを推奨 (将来追加)。

---

## 10. 残存リスクと推奨

| 観点 | 状態 |
|------|------|
| **挙動の差分** | 検出されず |
| **生成 RNG パス** | Python 委譲 (5–6 step / episode、~1.2%) で完全互換 |
| **upstream version drift** | `kaggle-environments==1.28.0` 固定。bump 時は parity test を再実行する運用 |
| **cross-platform** | macOS arm64 で audit 完了。Linux x86_64 / Vast.ai で再実行を **推奨** (現状未検証) |
| **stress (long-run)** | 8358 turn で 0 diff。10⁵ turn 級の連続実行は未実施 |

**推奨**: CI に `test_parity_full_episode` (slow marker) を nightly cron で
組み込み、upstream 更新や Rust リファクタ時の retrogression を自動検出する。

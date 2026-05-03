# Comet Spawn の Rust 実装 (2026-05-03)

`generate_comet_paths` (公式 Python: 1 回 ~91 ms) を Rust 化し、per-episode
speedup を **1.09× → 1.52×** に改善した。

## 実装方針

`random.uniform/randint` の **global state 消費順序を upstream と完全一致** させる
ため、RNG は依然 Python 側で消費する hybrid 構成:

1. Python facade が `random.uniform` を upstream と同じ順で呼んで
   `(e, a, phi)` 300 試行ぶんを事前 sampling
2. Rust に list として渡す
3. Rust が一試行ずつ rejection sampling (perihelion / 衝突判定) を実行し、
   最初に成功した試行の paths を返す
4. Python facade が新規 comet 4 体と group を obs に push

この設計の利点:
- **Python `random` の消費順が完全保存** → planet 生成や comet ships
  (`min(4 random.randint(1,99))`) など他の random 消費と整合
- **Rust が物理計算 (5000 点 sampling × N planets × visible 30 pts collision)
  を並列に高速処理**
- bit-exact parity を狙える

## 実装ファイル

| ファイル | 役割 |
|---|---|
| `simulator/rust/src/generation.rs` | `generate_comet_paths` の Rust 純粋数学実装 |
| `simulator/rust/src/fast_helpers.rs` | PyO3 wrapper `fast_generate_comet_paths` |
| `simulator/rust/python/orbit_wars_rust/_facade.py::_spawn_comet_via_rust` | Python facade で random sampling + Rust 呼出 + obs 更新 |
| `simulator/rust/python/orbit_wars_rust/_facade.py::_expire_comets_at_path_end` | upstream の pre-launch expire を Python facade で再現 |

## ベンチマーク

| 項目 | Python | Rust | speedup |
|---|---:|---:|---:|
| **Per-spawn (1 回 generate_comet_paths)** | 91.3 ms | 1.8 ms | **51×** |
| **Per-episode (env.run, 2p, 20 episodes)** | 1130 ms | 742 ms | **1.52×** |
| Per-episode (Rust comet OFF, 比較用) | 1502 ms | 1093 ms | 1.37× |

Rust comet ON にした効果だけで rust-side: 1093 → 742 ms (= **1.47×**)。

## Parity

`random` 消費順は完全一致するが、**collision check の数値計算で微小差分** が出る
seed が存在する。

| Seed | spawn count match | 備考 |
|------|:---:|---|
| 0    | ✓ 5/5 | bit-exact |
| 1    | ✓ 5/5 | bit-exact |
| 2    | ✓ 5/5 | bit-exact |
| 3    | ✓ 4/4 | upstream は 4 spawn 成功、Rust も同じ 4 spawn |
| 4    | ✓ 5/5 | bit-exact |
| 5    | ✓ 5/5 | bit-exact |
| **7** | **✗ 4 vs 5** | step=350 で Rust が spawn 成功、Python は失敗 |
| 42   | ✓ 5/5 | bit-exact |
| 100  | ✓ 5/5 | bit-exact |

→ **9 seeds 中 8 seeds (89%) で完全一致**。

`test_parity.py` (seed=0/1/2/7/42 で 12 ケース) は **すべて pass**:
- spawn 個数差は seed=7 のみ
- 全 episode 終了後の reward は seed=7 でも一致
- per-step planet/fleet 物理は全 seed で一致

### seed=7 step=350 の divergence について

詳細追跡で以下を確認:
- env を step=349 まで進めた状態で `comet_planet_ids` は両 backend 完全一致
- 両 backend の `generate_comet_paths` を **同じ initial_planets / RNG state**
  で直接呼ぶと **両方 None** を返す
- ところが env.step を経由して測ると **Rust 経由で spawn 成功** する

これは facade の `_spawn_comet_via_rust` が呼ばれるタイミングと
`_expire_comets_at_path_end` の差で `comets[*].path_index` の状態が微小ずれする
ことに起因と思われる。本格修正には Rust interpreter 側の expire 処理タイミングを
Python と完全一致させる必要があるが、影響が **9 seeds 中 1 seed の 1 spawn**
(全 45 spawn events 中 1 件 = 2.2%) で済んでいるため、現状は受容。

## オプトアウト

```bash
# Rust comet spawn を無効化、すべての spawn を Python interpreter に委譲
ORBIT_WARS_RUST_COMET=0
```

学習データの厳密 parity が必要な場合 (例: `kaggle_episodes/` を強再現
したい場合) はこの env var を 0 に設定する。代償は per-episode speedup
1.52× → 1.37×。

## 残タスク

- [ ] seed=7 step=350 の divergence の根本原因特定
- [ ] 100 seed × 500 step の大規模 parity test を nightly cron で回す
- [ ] Rust interpreter の expire 処理を upstream の pre-launch ordering に
      合わせる調査

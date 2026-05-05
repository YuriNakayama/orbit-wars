# rulebase/case9 — anti_ping_pong (iter2 result)

> 作成日: 2026-05-05
> 関連: `iter2_plan.md`, `iter1_*.md`
> Status: **棄却** (49.5% / 200戦、+5pp しきい値未達。ただし iter1 比 +3.5pp 改善)

## サマリ (Summary)

iter1 の analysis で特定した「劣勢時 cooldown 連鎖発火」を抑止するため、**`LOW_PLANET_BYPASS_THRESHOLD=8` で my_planets ≤ 8 のとき cooldown を全 bypass** + cooldown 値短縮 (3→1, 5→2) を実装。
200戦評価で **49.5%** (vs baseline_v4)。iter1 の 46.0% から **+3.5pp 改善**だが、採択しきい値 55% には到達せず **棄却**。
中盤 (120戦時点) では **55.8%** で推移しており、後半 80戦で v4 が押し返した。Seat bias は iter1 の 16pp → iter2 の 7pp に縮小。

## 数値 (Numbers)

### Phase B: vs baseline_v4 200戦 (seed 3000)

| 配置 | エピソード | v9 勝 | v4 勝 | draw | v9 勝率 |
|---|---|---|---|---|---|
| seat=0 (v9 先手) | 100 | 53 | 47 | 0 | **53.0%** |
| seat=1 (v9 後手) | 100 | 46 | 54 | 0 | **46.0%** |
| **合計** | **200** | **99** | **101** | **0** | **49.5%** |

- 平均試合長: 369.0 turn (iter1 の 370.5 とほぼ同)
- 信頼区間 (Wilson 95%): 約 [42.7%, 56.3%] → +5pp 達成 (≥55%) は信頼区間上限ぎりぎり
- iter1 (46.0%) との差 +3.5pp は信頼区間が重なるため統計的有意ではないが、改善方向にはある

### 中間値推移

| chunk | v9 / total | 累積勝率 |
|---|---|---|
| 0–20 | 10/20 | 50.0% |
| 0–40 | 26/40 | 65.0% (急上昇) |
| 0–60 | 34/60 | 56.7% |
| 0–80 | 41/80 | 51.3% |
| 0–100 | 53/100 | 53.0% |
| 0–120 | 67/120 | **55.8%** (しきい値到達) |
| 0–140 | 74/140 | 52.9% |
| 0–160 | 82/160 | 51.3% |
| 0–180 | 92/180 | 51.1% |
| 0–200 | **99/200** | **49.5%** (確定) |

→ seed が後半 (3160–3199) でばらつく。これは seed 依存ではなく **後半に v4 で seat=1 が連勝** した可能性 (試合番号順は seat0/seat1 別塊)。

### 関連 SHA / Run

- Branch: `feature/rulebase-planet-ping-pong`
- Base SHA: `5da249d` (iter2 実装は worktree のみ、未 commit)
- Seed range: 3000–3199 (各 seat 100戦)

## 診断 (Diagnosis)

**iter1 比で改善した点**

- iter1: 46.0% → iter2: 49.5% (+3.5pp)
- Seat bias: iter1 16pp → iter2 7pp (cooldown bypass が劣勢パターンの非対称性を一部解消)
- 想定通り、`LOW_PLANET_BYPASS_THRESHOLD=8` は崩壊シナリオでの「沈黙」を抑制している

**まだ +5pp 未達の理由 (推察)**

- 中盤の高勝率 (120戦時 55.8%) → 後半失速 = **seat1 で v9 が伸びない**。`bypass_threshold=8` がまだ厳しい (= 8 に達する前の「劣勢シグナル」を見逃している) 可能性
- cooldown 値 1 に短縮した結果、iter1 の主目的だった「ping-pong 抑制」効果が部分的に消失している可能性
- 余剰 ship 流用ロジック未実装 (case7 ACCUMULATE 等) なので、抑止された ship が遊休のまま

**測定上の懸念**

- 信頼区間 [42.7%, 56.3%] は wide。300戦で再評価すべきだが、コスト最小化方針なので iter3 に委ねる

## 判定 (Decision)

- **棄却** (採択しきい値 ≥55% 未達)
- ただし iter1 比で改善方向 (+3.5pp) は明確 → **iter2 の設計変更は iter3 のベースとして残す**
- ANTI_PING_PONG_ENABLED フラグ + LOW_PLANET_BYPASS_THRESHOLD 機構は維持

## NEXT ACTION (iter3 引き継ぎ)

1. **`LOW_PLANET_BYPASS_THRESHOLD=8 → 10` に緩和**: 劣勢シグナルを早めにキャッチ
2. **余剰 ship 流用 (case7 ACCUMULATE port)**: 抑止した launch を「複数ターン蓄積 → 遠距離 1 発」に転用
3. **rust simulator** (rustup インストール済みなら): `orbit_wars_rust` で実行時間を ~50 分 → ~20 分に短縮し、300戦評価を可能に
4. **agent 速度最適化** (別 commit): iter3 で性能維持を ablation 確認しつつ build_world cache 化

## 成果物 (Artifacts)

- compare 出力: `/tmp/compare_v4_iter2.log` (一時ファイル、永続化したい場合は別途)
- 実装: case9 source (worktree、未 commit)

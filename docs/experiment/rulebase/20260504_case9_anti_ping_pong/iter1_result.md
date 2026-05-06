# rulebase/case9 — anti_ping_pong (iter1 result)

> 作成日: 2026-05-04
> 関連: `plan.md`, branch `feature/rulebase-planet-ping-pong`, base SHA `5da249d`
> Status: **保留** (採択しきい値未達 / 設計の再調整を iter2 で実施)

## サマリ (Summary)

仮説 (ping-pong 抑制で余剰 ship 有効化 → 勝率改善 +5pp) のうち、**前半 (ping-pong 現象の存在 + 抑制機構の効果)** は確認できた。**後半 (それが勝率改善に繋がる)** は 100戦評価で **不成立** (vs v4 = 46.0%、しきい値 55% 未達)。
ANTI_PING_PONG_ENABLED は副作用として正当な reinforce / harass まで停止させ、勝率にはマイナスに作用していると推察。iter2 では **抑制パラメータを緩める方向 + 余剰 ship の用途を明示的に組み込む** ことが必要。

## 数値 (Numbers)

### Phase A: ping-pong 件数 (100戦, baseline_v9 vs baseline_v4 head-to-head, seed 1000)

| 指標 | baseline_v4 | baseline_v9 | 差分 |
|---|---|---|---|
| Episodes | 100 | 100 | — |
| Total turns | 38,140 | 38,140 | 同一 |
| Total launches | **75,360** | **62,019** | **-17.7%** |
| Ping-pong incidents | **2,692** | **2,382** | **-11.5%** |
| Incidents / episode | 26.92 | 23.82 | -11.5% |
| Incidents / 100 launches | 3.57 | 3.84 | +7.6% |

#### Top ping-pong pairs (v9 側)

`(6, 10)` 96 / `(4, 8)` 74 / `(5, 9)` 48 / `(11, 19)` 46 / `(7, 19)` 46
→ 中心付近の隣接惑星間で集中。production が高い planet 同士の reinforce 振動が主因と推察。

### Phase B: head-to-head 勝率 (100戦, baseline_v9 vs baseline_v4, seed 2000)

| 配置 | エピソード数 | v9 勝 | v4 勝 | draw | v9 勝率 |
|---|---|---|---|---|---|
| seat=0 (v9 が先手) | 50 | 27 | 23 | 0 | **54.0%** |
| seat=1 (v9 が後手) | 50 | 19 | 31 | 0 | **38.0%** |
| **合計** | **100** | **46** | **54** | **0** | **46.0%** |

- 平均試合長: 370.5 turn
- 信頼区間 (Wilson 95%): 約 [36.6%, 55.7%] → +5pp 達成は信頼区間下限が 51% 以上必要、未達
- **Seat bias が大きい** (54% vs 38%)。seed の偏り or anti 機構が seat0 で有利に効いている

### 関連 Run / SHA

- Branch: `feature/rulebase-planet-ping-pong`
- Base SHA: `5da249d0a8cff76a3dad08926332572ba340b04a`
- Diagnose run_id: `20260504_210528`
- Compare run log: `/tmp/compare_v4_100.log` (一時ファイル、要保管なら別途 dvc add)

## 診断 (Diagnosis)

**仮説どおりだった点 (H1, H2 部分肯定)**

- `26.92 / episode` の高頻度で ping-pong は実在。1試合の launch の **3.6% が小規模相互輸送に消費** されていた。
- Anti-ping-pong cooldown は launches を 17.7% 削減 (= 不要 launch を抑制)、絶対 incident は 11.5% 減。

**仮説に反した点 (勝率改善が起きなかった理由)**

- **per-100-launch 比率はむしろ +7.6%**: launches を分母として削減した分、relative ping-pong 率は微増。これは「cooldown が効いた pair 以外でも依然として振動が残る」ことを示し、抑制機構の局所性が原因。
- **正当な reinforce が止まる**: `REINFORCE_MIN_DEFICIT=3` で deficit=1-2 の小脅威に reinforce しないため、本来必要だった補強が遅れて惑星を取られるケースがあると推測。`PING_PONG_PAIR_COOLDOWN_TURNS=3` も同 pair の正当な再補強を 3 ターン止める。
- **Harass cooldown の副作用**: `HARASS_TARGET_COOLDOWN_TURNS=5` は 1 回 harass 成功後 5 ターン同 target に行けない。敵がすぐ奪い返して production を取り戻す時間を与える。
- **seat bias 16pp**: 配置順で勝率が大きく変わる = anti 機構と環境の相互作用に偏りがあり、評価ロバストさも問題。

**未検証だった「余剰 ship の有効化」**

iter1 では「ship を抑止する」だけで「貯めた ship を別の mission に流用する」設計を入れていない。plan.md の iter2 引き継ぎリスト (ACCUMULATE / multi-source swarm / rear-guard) を実装しないと、抑制した ship は遊休のまま勝率に貢献しない。

## 判定 (Decision)

- **棄却 (default OFF)**: `ANTI_PING_PONG_ENABLED = True` のまま baseline_v9 を production 採用しない。
- **iter2 へ持ち越し**: 設計を **(a) cooldown 緩和 + (b) 余剰 ship の流用ロジック** の 2 軸で再調整。
  - cooldown 緩和案: `PING_PONG_PAIR_COOLDOWN_TURNS=3 → 1`、`HARASS_TARGET_COOLDOWN_TURNS=5 → 2`、`REINFORCE_MIN_DEFICIT=3 → 2`
  - 余剰 ship 流用案 (優先順): ACCUMULATE 連携 (case7 から port) → multi-source swarm 増強 → rear-guard reserve

## 採否 / Promote

- `dev/runpod promote` 不要 (rule-based のため weights なし)
- AGENT_REGISTRY の `baseline_v9` 登録は維持 (iter2 で同 case を上書き編集)
- Kaggle 提出は **しない** (publicScore は採否根拠にしないルールに従い、勝率改善が確認されてから検討)

## 成果物 (Artifacts)

- 診断 summary: `data/output/diagnostics/ping_pong/20260504_210528/summary.json`
- (将来的に) compare_v4 出力を JSON 化して `data/output/experiment/rulebase/case9/iter1/compare_v4.json` に置く設計に変更すべき (iter2 で対応)

## 次の iter で確認すべきこと (NEXT ACTION)

1. cooldown 値を緩めて 100戦 → 勝率が 50% 帯まで戻るか確認
2. ACCUMULATE 連携 (case7 の `ACCUMULATE_*` 定数 + mission を case9 に port) で余剰 ship を遠距離 1 発に変換
3. seat bias を抑えるため seed range を広く (今回は seed=2000 起点) / より多い episode 数で再評価
4. Top ping-pong pair の構造分析 (惑星 production / 距離別) を replay-viewer で見る → 次の対策設計に活かす

詳細な原因分析は `/experiment-analysis` で別途実施推奨。

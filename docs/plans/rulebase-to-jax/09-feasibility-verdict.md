# rulebase-to-jax — 実現可能性の最終判定と方針確定

## 背景

「選択肢3 = Python 決定ロジックを丸ごと JAX で忠実再実装」が本当に実現可能か、最大の難所を実コードで精査した結果。

## パート別 JAX 化可能性 (実コード検証済み)

| パート | 参照 | 難易度 | JAX 構造 |
|--------|------|--------|----------|
| `aim_with_prediction` (角度) | physics.py:195, **case2 aim_jax.py:261 に既存** | 🟢 EASY | 5-iter refinement + 220 候補 grid、共に固定。流用可 |
| `resolve_arrival_event` (戦闘) | world_model.py:98 | 🟢 EASY | owner 最大4、argsort + where、分岐なし |
| 110ターン timeline | world_model.py:150 | 🟡 MEDIUM | `lax.scan` + 固定 MAX_FLEETS arrivals |
| keep_needed 二分探索 | world_model.py:194-220 | 🟡 MEDIUM | 二分探索を捨て全 keep 候補を並列評価し min 成立を取る |
| mission 列挙 (48²=2304) | option_collector.py:26 | 🟡 MEDIUM | pair grid 並列 + mask。guard は全て比較演算 |
| **逐次 greedy 予算配分** | mission_resolver.py:58 (`spent_total[src]+=send`) | 🔴 要注意 | mission N の可否が N-1 の消費に依存 |

## 🔴 逐次 greedy の正確な評価 (誤解を解く)

サブエージェントは当初これを「BLOCKER」と評価したが、**精査の結論は「BLOCKER ではない」**:

- 逐次依存そのものは `lax.scan` の carry (`spent_total`, `planned_commitments`) で**表現できる**。JAX が苦手なのは「**可変長**の逐次」であって「**固定長**の逐次」ではない。
- mission を score 降順に並べ、`MAX_MISSIONS` 固定長で `lax.scan` し、各 step で `spent_total` を carry すれば、本物の `for mission in sorted(missions)` ループと**同じ順序・同じ消費**を再現できる。
- サブエージェントが「lossy」と言ったのは「mission 順を勝手に固定すると starve する」ケースだが、**本物と同じ score sort + 同じ tie-break で並べれば順序が一致** → starve のタイミングも一致 → 結果一致しうる。

つまり **BLOCKER は「最適性の喪失」ではなく「固定 MAX_MISSIONS で打ち切る分の取りこぼし」だけ**。MAX_MISSIONS を十分大きく取れば(pair 上限 2304 で十分)取りこぼしゼロにできる。

## 最終判定: FEASIBLE-WITH-EFFORT (BLOCKER なし、ただし実測必須)

**選択肢3 (full faithful port) は実現可能**。ただし「一致するか」は机上では確定できず、**逐次 greedy 部分の parity を実測で確かめる**のが必須。

### 一致の鍵 (全て本物と演算子レベルで揃える)
1. mission score の計算式 (value / (send + turns·w + 1))
2. **score sort の tie-break** (同値 mission の順序。index 最小等で本物と統一) ← ここがズレると逐次消費順がズレて全崩壊
3. `spent_total` の carry と `source_inventory_left` 判定の閾値 (`< 1` で abort 等)
4. `preferred_send` の margin 群、`keep_needed`/`available` (turn0 で=10 になる核心)

## 確定方針: 逐次 greedy を固定shape unroll で JAX 化 → CPU で parity 実測

`lax.scan(mission_step, carry=(spent_total, planned), xs=sorted_missions[:MAX_MISSIONS])` で本物の逐次ループを忠実再現し、**CPU で action 一致率を実測**して判断する (机上で決めない)。

### 実装順序 (改訂、ボトムアップ差分テスト)

```
PoC0 (まず最小実証): turn0 の available 計算だけ本物一致
   → lite の reserve=prod×3 を捨て keep_needed 並列評価に置換
   → turn0 home が available=10 になり「10隻 launch」を再現するか CPU 実測
   ↓ 一致したら本格着手
Step1 core_jax: geometry/physics (aim_jax 流用) — x64 parity
Step2 worldmodel: timeline(scan) + keep_needed(並列) + resolve_arrival(argsort)
Step3 missions: 2304 pair 並列 score + 本物一致の sort/tie-break
Step4 mission_resolver: 固定 MAX_MISSIONS の lax.scan で逐次消費を再現 ★最重要・実測関所
   → ここで action 一致率を CPU 実測。100%なら成功、崩れたら原因 obs を replay
Step5 case1 full 統合 → 大量 obs で 100% 一致 assert
```

### 撤退条件 (失敗回避)

PoC0 または Step4 で **CPU 実測の action 一致率が 100% に届かず、原因 obs の replay でも詰められない**場合:
- 逐次 greedy の取りこぼし/順序が固定shape で再現不能と確定 → **physics-only JAX + mission resolver は host callback** にフォールバック (100%一致は保証、GPU rollout は一部 host round-trip)。
- これを 09 の Open Item とし、Step4 の実測結果で確定する。机上で先に諦めない。

## まとめ

| 問い | 答え |
|------|------|
| 選択肢3は実現可能か | **可能 (FEASIBLE-WITH-EFFORT)**。当初 BLOCKER とされた逐次 greedy は固定長 `lax.scan` で表現でき、MAX_MISSIONS を pair 上限で取れば取りこぼしゼロ |
| 唯一の不確実性は | 逐次消費順 (score sort の tie-break) が本物と完全一致するか。**机上不可、CPU 実測で確定** |
| 最初の一手は | PoC0: turn0 の available=10 を本物一致させる (lite の reserve 式を捨てる) |
| 失敗時の退避 | physics-only JAX + mission resolver host callback (100%一致は保証) |

# iter5 — Loop Resume State (Phase 4 RE-LAUNCHED with FIX)

> 作成日: 2026-05-05
> Status: **Phase 4 評価実行中** (PID 58827, seed 7000-7199, 200戦) — バグ修正後の再実行

## バグ発見 + 修正

最初の Phase 4 評価 (seed 6000) で 40戦中 v9=7 (17.5%、壊滅的) を観測 → 即停止。

原因: ACCUMULATE は **自惑星数 1 個 (case9 序盤、turn=11) でも発火** し、
唯一の src (id=32) の available 20 ships を全て hold (+ iter2 bypass=8 の趣旨と矛盾)。
結果として `source_attack_left = max(0, 20 - 20) = 0` で全 mission 不発。

修正: `strategy.py` で `len(my_planets) > LOW_PLANET_BYPASS_THRESHOLD (=8)` の時のみ
ACCUMULATE 起動。case9 序盤・劣勢時は通常 mission に専念。

snapshot test:
- バグあり: `[]` (0 moves)
- 修正後: `[[16, -0.853, 22]]` (1 move、iter2 と同等の動作)

## Phase 4 評価実行中 (修正版)

- コマンド: `compare_v4.py -n 100 -p 4 --seed 7000`
- ETA: ~100 分
- ログ: `/tmp/compare_v4_iter5_v2.log`
- PID: 58827

## Phase 5 で次にやること

PID 58827 の重複ガード確認 → 進行中なら skip / 完了済みなら:
- iter5_result.md 作成
- replay 比較で iter5 が改善したかを analyze
- 採否判定:
  - +5pp (≥54.5%): 採択 + 完了 → loop 停止
  - +2pp (≥51.5%): 採択
  - <+2pp: 棄却

## 過去 iter サマリ

- iter1 (cooldown 抑止): 46.0%
- **iter2 (bypass=8 + 値短縮)**: **49.5% (best)**
- iter3 (bypass=10): 47.8%/180中断
- iter4 (multi-source): 47.0%
- iter5 (ACCUMULATE port): バグ修正後、200戦評価中

## 既知の todo

- snapshot_update.py の出力先バグ (worktree root に tests/ を作成) は別 commit で修正

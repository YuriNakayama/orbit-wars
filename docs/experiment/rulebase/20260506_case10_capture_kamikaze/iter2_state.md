# case10 iter2 — Loop Resume State (RUNNING)

> 作成日: 2026-05-06
> Status: **iter2 評価実行中** (PID 26882, seed 13000-13199, 200戦)

## 進行中

iter1 棄却 (45.5%) を受けて **iter2 = 逆方向 (defense 寄り)** に振り切り:
- `STATIC_NEUTRAL_VALUE_MULT: 1.6 → 1.2` (default 1.4 より低く、capture を意図的に弱める)
- `HARASS_MIN_SRC_RESERVE: 6 → 14` (default 10 より高く、kamikaze 抑制)
- 残り 3 定数は default 復元:
  - `EARLY_NEUTRAL_VALUE_MULT: 1.4 → 1.2`
  - `SNIPE_VALUE_MULT: 1.30 → 1.12`
  - `HARASS_PRODUCTION_STEAL_TURNS: 8 → 5`

## 評価

- コマンド: `compare_v4.py -n 100 -p 4 --seed 13000`
- ETA: ~100 分
- ログ: `/tmp/compare_v4_case10_iter2.log`
- PID: 26882

## 採否しきい値

- ≥55%: 採択候補、iter3 で 300戦 confirm
- 51-55%: 弱採択、iter3 で 300戦 confirm
- <51%: 棄却 → 連敗カウント 2、iter3 で別軸
- iter1 (45.5%) 比 +5pp 改善 (≥50.5%) なら方向性として意味あり

## 連敗状況

iter1 棄却で **1 連続棄却**。iter2 も棄却なら 2 連続、3 連続で memory 11 連敗パターンと類似 → 早期 loop 終了判断。

## 過去 iter サマリ (case10)

- iter1: 45.5% (5 定数同時改造、+5pp 未達、棄却)
- iter2: 逆方向 (defense 寄り)、評価中

# rulebase/case5 JAX port — plan (genuine from-scratch port required)

> 記録: 2026-06-03 ~15:00 / 状態: scoped, not-started / 親: ../20260603_case8_jax_port/

## case5 は lineage 外 — core_jax 再利用不可

case5 = **baseline_v5 (LB1224 Kaggle notebook の verbatim port)**。case1-4,6-9 の
sigmaborov LB897 lineage とは **別作者・別戦略**。`agent_full.py` 2495行 monolith
(独自 WorldModel / plan_shot / build_modes / opening_filter / target_value、formula
も constant も LB897 と異なる)。

reuse 可能なのは core/safety.py (case1 と identical) のみ。physics は 232行差、
geometry は case5 に無し。→ **case8 core_jax は流用不可、from-scratch port が必要**
(case1 原 port と同等 ~多 tick の労力)。

## 重要: 安直な流用は honest でない

検証として case8 core_jax (LB897) を case5 に置いて gate を回すと **vs case5 Python
60% (6/10)** で非劣化に見える。**但しこれは case5 の faithful port ではない** — LB897
agent に case5 ラベルを貼っただけ。case5 (LB1224) は memory
[[project_case5_validation]] で publicScore 600 (v4=745 下回る) と弱く、LB897 core が
たまたま勝つだけ。**「劣化しない JAX 化」の目的は元実装の忠実再現**なので、この placeholder
は採用せず削除した。

## 進め方 (本来の Step1→実装)

1. Step1: case5 用 baseline_jax/core_jax を新規作成し、case5 agent_full.py の
   WorldModel/plan_shot/missions を JAX 化。結合テストは **JAX vs case5 Python**。
   最初は RED (lite/部分実装) から、bottom-up parity で詰める。
2. case5 独自の formula (fleet_speed, plan_shot guards, build_modes, target_value) を
   case1 同様 parity test で 1 つずつ移植。
3. 各段で foreground gate (vs case5 Python) で非劣化確認。

## 判断材料

case5 は **採用されなかった弱い実験 case** かつ **唯一 from-scratch が必要** な case。
費用対効果は低い。残り 8/9 case (case1,2,3,4,6,7,8,9) は JAX 化済。case5 を full
port するか、lineage 8 case 完了をもって区切るかは進捗とコストを見て判断。
まずは Step1 scaffold から着手する方針。

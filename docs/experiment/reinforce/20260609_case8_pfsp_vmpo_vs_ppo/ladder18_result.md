# case8 ladder18 — rollout aim を先読み intercept 化 RESULT

> 関連: strict_loss_replay_diag.md / hypotheses.md
> run_id: 20260615-114258__feature-poc-v-mpo__0ee7e14__seed0 / commit: 0ee7e14f
> case: reinforce_case8_vmpo_ladder18 / GPU: RTX 4090 SECURE ($0.69/h)
> 開始: 2026-06-15T11:42Z / 早期停止: iter34 (full が 0.81-0.83 帯で plateau 確認、best.pt 確保、ladder19 を優先) / pod destroy済 (0確認)

## Summary

replay診断で特定した train/eval aim 不一致 (訓練 rollout は naive atan2 直射で動く
planet を外す、eval/strict は intercept 先読み) を修正。**aim修正は正しく機能** —
full held-out が 0.797→0.812→0.828 と上昇し **全run最高 (ladder11 0.812) を更新**、
高T0段の訓練 win も改善 (ladder17 T0=225 0.667 → ladder18 0.786)。**しかし
held-out strict_v1 は 0.0→0.0156 (noise floor) のまま** — aim修正だけでは strict の
壁を破れない。素strict (T0=0) も win 0.0/reward -3.42 で不変。

## Numbers

### held-out

| iter | strict_v1 | full | elo |
|---|---|---|---|
| 0  | 0.0    | 0.797 | 1484 |
| 10 | 0.0156 | 0.812 | 1469 |
| 20 | 0.0156 | **0.828** | — |

→ full は単調上昇で **新記録 0.828**。strict_v1 は noise 床で平坦。

### strict 段 (訓練 win)

| iter | T0 | win | 備考 |
|---|---|---|---|
| 1 | 225 | 0.786 | ladder17同段 0.667 より↑ (aim効果) |
| 9/18 | 170 | 0.641/0.594 | ~0.6帯で頭打ち (ladder11と同水準) |
| 4 | 0 | 0.0 (rew -3.42) | 素strict 不変 |

## Diagnosis

aim修正は **中終盤の捕獲効率を上げ full 天井を 0.828 へ押し上げた** (恒久的に有用な
修正) が、strict_v1 攻略には不十分。これで campaign の「機械的」要因 — aim・
curriculum (T0/ε/逆カリ)・対戦量 — は**すべて出し尽くし、いずれも素strict 0% を
動かせない**ことが確定。残る真因は2つに絞られた:
1. **戦略的弱さ**: aim が直っても strict の序盤/中盤の意思決定に有利を作れない。
   strict は少数精鋭 launch で territory を複利的に拡大、agent は追随できない。
2. **reward 支配**: 素strict段 reward -3.42 は shaping (-2.4) が ±1勝利を埋没させ、
   「勝つ」勾配が立たない。f_var も勝てない T0=0段をほぼ選ばない。

## Decision

- 採否: **partial adopted** — strict 攻略は inconclusive/不可だが、**aim修正は full
  新記録 0.828 を出した恒久改善**として保持。ladder18 best.pt を次の resume 基盤に。
- 次の一手: strict 攻略は「機械」でなく「戦略+reward」。複合策:
  (a) **reward 再設計**: shaping_coef を素strict段で下げ terminal_scale を gentle に
      (2-3) 上げて勝利信号優位に + 捕獲イベント報酬。
  (b) **難段強制サンプリング**: force_rung_low_every で T0=0段に毎K iter 勾配を届ける。
  (c) (将来) opponent-modeling (DPIQN) / BC でstrictの序盤手を直接教える。
  resume は ladder18 best.pt (full 0.828)。

## Artifacts

- model: `data/output/models/reinforce/case8_vmpo_ladder18/runs/20260615-114258__feature-poc-v-mpo__0ee7e14__seed0/` (best.pt, metrics.json)
- aim修正コード: `policy/aim_jax.py`, parity test `tests/.../test_intercept_aim_jax_parity.py` (9 passed)

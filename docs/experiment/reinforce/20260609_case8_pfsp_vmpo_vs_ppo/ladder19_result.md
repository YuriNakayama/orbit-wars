# case8 ladder19 — reward勝利信号優位 + 難段T0=0強制サンプル RESULT

> 関連: ladder18_result.md / strict_loss_replay_diag.md / hypotheses.md
> run_id: 20260615-135210__feature-poc-v-mpo__194e336__seed0 / commit: 194e336e
> case: reinforce_case8_vmpo_ladder19 / GPU: A100 80GB SECURE ($1.39/h)
> 開始: 2026-06-15T13:58Z / 早期停止: iter9 (forced T0=0 win 床固定が明確) / pod destroy済 (0確認)

## Summary

aim修正後に残った真因 (素strict reward支配 + f_var難段回避) を複合対処: terminal_scale
1.0→2.5 + shaping_coef 1.0→0.5 (勝利優位) + force_rung_low_every=2 (T0=0段を2 iter毎
強制照射)。resume ladder18 best.pt (aim修正込, full 0.828)。結果、**forced T0=0段の
win は 9 iter で 0.021→0.0→0.0→0.005→0.01 と床に固定**、勝利優位 reward + 確実な
サンプリングでも素strict を全く動かせず。仮説は棄却。

## Numbers

### forced T0=0 (素strict) 段 win 軌跡

| iter | win | reward |
|---|---|---|
| 0 | 0.021 | -4.24 |
| 2 | 0.0   | -4.47 |
| 4 | 0.0   | -4.44 |
| 6 | 0.005 | -4.44 |
| 8 | 0.01  | -4.44 |

- force_rung_low_every=2 で T0=0段が確実に毎2 iter 出る (狙い通り作動) も win は床
- terminal_scale=2.5 は持続的敗北を増幅し reward を -4.4 へ (勝てないので逆効果)

## Diagnosis (campaign の最深の結論)

ladder19 は最後の「純粋RL探索」レバー — reward整形と難段強制サンプリング — を両方
投入したが素strict を動かせなかった。**根本原因が確定**:

> agent は素strict にほぼ1回も勝てない (win ~0.005-0.02 = 192戦中0-4勝)。
> **勝利エピソードが存在しないので、増幅すべき勝利勾配が無い** (zero-variance)。
> reward を勝利優位にしても、強制的に難段を見せても、「勝った経験」がゼロでは
> 方策勾配は何も学べない。これは戦略的劣位による信号欠如の壁。

campaign で否定されたレバー (すべて素strict 0%):
- curriculum 形状 (T0/ε 窓 ladder1-13, 逆カリ ladder14-16)
- 対戦量 (ladder17)
- aim修正 (ladder18 — full 0.828 達成も strict 不可)
- reward整形 + 難段強制 (ladder19)

## Decision

- 採否: **rejected**
- 次の一手 (要ユーザー確認): 「探索で勝ちを発見」が無理なので **「勝ちを人工生成して
  勾配を作る」** クラスへ転換:
  - **(本命) handicap**: 素strict相手に learner の初期 ship を boost (mode=ships,
    h>1) して*時々勝てる*局面を人工生成 → 勝利勾配を bootstrap → win 上昇に応じ
    h→1.0 へ anneal。過去の handicap 棄却 (phaseB) は「壊れた aim + from-scratch」が
    交絡 — aim修正済 + ladder18 強base (full 0.828) で再試行価値あり。
  - (代替) BC: strict_v1 の手を直接模倣学習 (探索でなく教示で序盤を獲得)。
  resume は ladder18 best.pt。

## Artifacts

- model: `data/output/models/reinforce/case8_vmpo_ladder19/runs/20260615-135210__feature-poc-v-mpo__194e336__seed0/` (best.pt, metrics.json iter 0-9)

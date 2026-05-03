# [rulebase/case6] iter7: cap 値さらに緩める方向 + MAX_TARGET_TURNS 絞る方向の周辺探索

## 仮説

iter6 で `MAX_HOLD_TURNS` を 2/3/4 で 100戦 sweep した結果:

| cap | 勝率 | fleet peak ratio | launches/ep ratio |
|---|---|---|---|
| 2 | 50.0% | 1.23 | 0.91 |
| 3 | 56.0% | 1.25 | 0.96 |
| 4 | 58.0% | 1.24 | 0.99 |

**fleet peak ratio はほぼ同じ (1.23/1.25/1.24)** なのに **launches/ep が単調増加 (0.91/0.96/0.99)** で **勝率も単調上昇**。「cap が緩いほど launches が増えて勝率がじわっと上がる」傾向。

ただし cap=∞ (= iter3 cap なし) は 54.7% (300戦) と cap=3 より明確に低いので、**最適は 4 と ∞ の間** (cap=5 or cap=6 で局所頂点があり得る)。

### 主仮説 (A/B)
cap=5, cap=6 で launches/ep が 1.0+ に達し、勝率が cap=4 (58%) を上回る可能性がある。ただし cap=∞ で崩れる以上、必ずどこかで頭打ち→劣化に転じる。**cap=5/6 のうち高い方を見て局所頂点を当てる**。

### 副次仮説 (C)
iter5 Stage 2 で seed=2000 のみ 51% (seat=1 44%) と崩れた。これは `MAX_TARGET_TURNS=30` (broad) で **長距離 hold が解放後に間に合わない** 害が seed=2000 の盤面で顕在化した可能性。`MAX_TARGET_TURNS=25` に絞ると遠距離 burst が抑えられ、seed variance が減る → 100戦 でも勝率がじわっと上がる可能性。

## 変更点

### 1. compare_v5.py に `--max-target-turns` flag 追加

既存の `--max-hold-turns` と完全に同パターン。`int | None` で default `None` → 何も override せず config 値 (=30) を使用。起動 echo に `MAX_TGT={値}` を追加して取り違え防止。

```python
max_target_turns: int | None = typer.Option(
    None, "--max-target-turns",
    help="Override STAY_BURST_MAX_TARGET_TURNS for ablation (default: config value).",
),

if max_target_turns is not None:
    cfg.STAY_BURST_MAX_TARGET_TURNS = max_target_turns

typer.echo(
    f"STAY config: ENABLED=... DEFENSE=... BURST=... "
    f"MAX_HOLD={cfg.STAY_BURST_MAX_HOLD_TURNS} "
    f"MAX_TGT={cfg.STAY_BURST_MAX_TARGET_TURNS}"
)
```

config.py 自体は **変更しない** (試行錯誤痕跡を残さない方針、iter6 と同じ)。

### 2. case6 baseline 本体の変更なし

`STAY_DEFENSE_ENABLED=False`、burst パラメータ (gain=1, ships=8)、状態管理 (`_STAY_STATE`)、cap=3 (config 既定) は iter6 確定構成のまま。cap および target は CLI 経由で run-time override する。

## 評価方針

### Stage 1: 100戦 sweep (3 並列, 推定 ~55分)

```bash
mkdir -p /tmp/case6_iter7
nohup uv run --directory backend \
  python -m pipeline.rulebase.case6.evaluation.compare_v5 \
  -n 50 --seed 1000 --max-hold-turns 5 > /tmp/case6_iter7/A_cap5.log 2>&1 &
nohup uv run --directory backend \
  python -m pipeline.rulebase.case6.evaluation.compare_v5 \
  -n 50 --seed 1000 --max-hold-turns 6 > /tmp/case6_iter7/B_cap6.log 2>&1 &
nohup uv run --directory backend \
  python -m pipeline.rulebase.case6.evaluation.compare_v5 \
  -n 50 --seed 1000 --max-target-turns 25 > /tmp/case6_iter7/C_tgt25.log 2>&1 &
```

各 100戦 (seat=0/1 各 50)、seed=1000。基準は iter6 cap=3 の 100戦 = **56.0%**。

### 判定基準

iter6 で得た cap=3 100戦 baseline = 56% を基準に比較:

| 結果 | 判定 |
|---|---|
| A (cap=5) or B (cap=6) が **65%+** で明確優勢 | Stage 2 (300戦, seed 1000/2000/3000) でその cap を確証 |
| C (tgt=25) が **65%+** で明確優勢 | Stage 2 で C+cap=3 を 300戦 |
| 全部 56% ±5pp (51〜61%) | **cap=3, tgt=30 が局所最適確定、case6 完全終了 → case7 推奨** |
| いずれかが iter6 cap=4 (58%) と同等で launches/ep がさらに伸びている | 弱い証拠、Stage 2 やるか別アプローチに振るかは慎重判断 |

### Stage 2: 300戦確証 (条件付き)

100戦判定で「明確優勢」が見えた場合のみ起動。3 並列 (seed 1000/2000/3000) で各 100戦 = 300戦、推定 ~50分。ログは `/tmp/case6_iter7/stage2_seedNNNN.log`。

## 期待効果

iter6 までの観測 (cap が緩いほど launches+勝率が単調上昇) を信頼すれば、cap=5 or cap=6 で 60%+ が出る可能性は中程度。ただし cap=∞ (54.7%) で崩れる事実から、**cap=5/6 のどちらかが頂点で逆側は劣化** する非単調性が想定される。

副次仮説 C は seed variance を経由した間接的効果なので、100戦単発では検出力低い (せいぜい数 pp 改善)。本命は A/B の cap 緩め方向。

最頻シナリオ予想:
- A=B≒cap=4 (58%) ±5pp、C≒cap=3 (56%) ±5pp → 「cap=3 周辺が局所最適」が再確認され case6 確定
- A or B が 60%+ → Stage 2 起動して確証

## 非ゴール

- defense の再導入はしない (iter2 で害確認済)
- burst パラメータ (gain/ships) は触らない
- cap=7+ や tgt=20 以下は試さない (cap=∞ 54.7% に近すぎ / tgt=20 では遠距離 burst を切り過ぎ)
- Vast.ai / Kaggle 提出はしない

## 実装範囲

| ファイル | 変更 |
|---|---|
| `case6/evaluation/compare_v5.py` | `--max-target-turns` CLI 追加、起動 echo に MAX_TGT 表示 |
| (テスト) | 既存テスト (13 tests) 合格確認のみ、新規追加なし |

## 制約 (絶対遵守)

- `STAY_DEFENSE_ENABLED` は False のまま
- case0〜case5 のコードは変更しない
- compare_v5.py 以外で config を書き換えない
- 300戦 Stage 2 は user 認可済 (この iter の流れ) ただし 100戦判定で必要時のみ起動
- 実行は背景化 (nohup &) し sleep でブロックしない
- pytest case6 を Step 1 後に必ず走らせる (compare_v5 API 後方互換確認)

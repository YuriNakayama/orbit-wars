# [rulebase/case6] iter6: STAY_BURST_MAX_HOLD_TURNS の cap 値 sweep

## 仮説

iter5 で `STAY_BURST_MAX_HOLD_TURNS=3` が 300戦 vs baseline_v5 で 59.7%
(95% CI 54.1〜65.2%, p≈0.0004) と統計的に有意な改善を達成した。
ただし「3」を採用した根拠は **「broad burst の累積効果は維持しつつ stuck
hold を防ぐ」** という質的設計であり、数値そのものは一発打ちで決めたもの。

iter6 では cap 値の周辺 (2 / 3 / 4) を sweep し、**3 が局所最適か (= 周辺
値が同等以下) を確認する**。これは新たな勝率の押し上げを狙う実験ではなく、
iter5 採用構成の妥当性確認 ablation である。

## 変更点

### 1. compare_v5.py に `--max-hold-turns` オプション追加

既存の `--no-defense` / `--no-burst` と同じパターン。`int | None` で
default は `None` → 何も override せず config の値 (= 3) をそのまま使用。

```python
max_hold_turns: int | None = typer.Option(
    None, "--max-hold-turns",
    help="Override STAY_BURST_MAX_HOLD_TURNS for ablation (default: config value).",
),

if max_hold_turns is not None:
    cfg.STAY_BURST_MAX_HOLD_TURNS = max_hold_turns

typer.echo(
    f"STAY config: ENABLED=... DEFENSE=... BURST=... "
    f"MAX_HOLD={cfg.STAY_BURST_MAX_HOLD_TURNS}"
)
```

config.py 自体は **変更しない**。試行錯誤の痕跡を残さない方針。

### 2. case6 baseline 本体の変更なし

`STAY_DEFENSE_ENABLED=False`、burst パラメータ (gain=1, ships=8, dist=30)、
状態管理 (`_STAY_STATE`) は iter5 のまま。

## 評価方針

### Stage 1: 100戦 sweep (3 並列, ~55 分)

```bash
mkdir -p /tmp/case6_iter6
nohup uv run --directory backend \
  python -m pipeline.rulebase.case6.evaluation.compare_v5 \
  -n 50 --seed 1000 --max-hold-turns 2 > /tmp/case6_iter6/cap2.log 2>&1 &
nohup uv run --directory backend \
  python -m pipeline.rulebase.case6.evaluation.compare_v5 \
  -n 50 --seed 1000 --max-hold-turns 3 > /tmp/case6_iter6/cap3.log 2>&1 &
nohup uv run --directory backend \
  python -m pipeline.rulebase.case6.evaluation.compare_v5 \
  -n 50 --seed 1000 --max-hold-turns 4 > /tmp/case6_iter6/cap4.log 2>&1 &
```

各 100戦 (seat=0/1 各 50)、seed 1000。cap=3 が iter5 Stage 1 の再現
(62/100 想定 ±数 pp の seed 上振れ込み) になるはず。

### 判定基準

| 結果 | アクション |
|---|---|
| いずれかの cap が **65%+** で明確優勢 | その cap を Stage 2 (300戦, seed 1000/2000/3000) で確証 |
| 3 つとも 100戦 ±5pp 以内 | cap=3 が局所最適、case6 確定 (Stage 2 不要) |
| cap=3 再現が iter5 (62/100) と ±5pp 以上ずれる | 実装ドリフト疑い、調査優先 |

### Stage 2: 300戦確証 (条件付き)

Stage 1 で cap=2 or cap=4 が cap=3 より明らかに優勢な場合のみ起動。
3 並列 (seed 1000/2000/3000) で各 100戦 = 300戦、~50 分。

## 期待効果

- 3 候補のいずれかが iter5 を有意に上回る確率は低い (cap 値変化は感度小と
  予想)、最頻シナリオは「cap=3 が確認され、case6 が確定」
- 副次的に cap 値感度の幅が定量的に分かる (案件説明に有用)

## 非ゴール

- defense の再導入はしない
- burst パラメータ (gain/ships/dist) も触らない
- cap=1, cap=5+ は試さない (cap=1 は iter4 厳しめの縮小版で害見込み、cap=5+
  は累積 hold の変動が小さく差が出にくい)
- Vast.ai / Kaggle 提出はしない

## 実装範囲

| ファイル | 変更 |
|---|---|
| `case6/evaluation/compare_v5.py` | `--max-hold-turns` CLI 追加 |
| (テスト) | 既存テスト合格確認のみ、新規テストは追加しない |

## 制約 (絶対遵守)

- `STAY_DEFENSE_ENABLED` は False のまま
- case0〜case5 のコードは変更しない
- compare_v5.py 以外で config を書き換えない
- 300戦 Stage 2 は user 認可済み (この iter の流れ)
- 実行は背景化 (nohup &) し sleep でブロックしない

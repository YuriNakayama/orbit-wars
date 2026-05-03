# [rulebase/case6] STAY mission を追加して防衛保留 + 速度ボーナス発射の二目的を解く

## 仮説

ユーザー原文:

> rulebaseモデルのmissionに
> 1. 敵shipのガード
> 2. 先ターンに敵や中立惑星のcaptureする
> 上記二つの目標のためにshipを射撃せずにstayするものを追加します。
> この実験ではmissionの追加だけではなく、scoreの計算追加(1に紐づくものとして、敵の出発済みshipを元に先のターンに自身の惑星を失う可能性、2に紐づくものとして、先ターンにまとめてshipを出発させることによる移動速度向上)が必要です。

つまり、現行 case4 の greedy な mission ループは「いま発射可能な ship は今ターン使う」前提で動いており、

- (a) 既に飛んできている敵 fleet に対する **保留 (HOLD)**
- (b) 1 ターン待って artillery を集めることで `fleet_speed = 1 + (max-1) × (log(ships)/log(1000))^1.5` を引き上げ、結果的に「先ターンに到着する」**蓄積発射 (BURST)**

の二つの STAY 動機が現状ロジックには欠けている。STAY judge を **planet 単位の per-source veto + ship 留置量** として導入し、case4 (= production champion baseline_v4) の総合勝率を超えるかを 100 戦で評価する。

## 比較 baseline

- **baseline_v4 (case4)** を比較対象とする。case6 は case4 のフルコピー上に STAY judge を追加した形なので、純粋な ablation になる。
- 将来的に baseline_v5 / baseline_v1 / baseline_v3 にも回す価値があるが、初回 100 戦は **case6 vs case4** に絞る (memory: `<300 戦は seed variance 大` のため、まずは方向性確認)。

## 既存アーキテクチャとの接続点

case4 (= 採用ベース) は以下の構造:

```
agent(obs)
  └ build_world(obs) → WorldModel
      ├ arrivals_by_planet      … 敵fleetの到着台帳
      ├ predicted_arrivals      … OM予測 (現状 default OFF)
      ├ base_timeline           … per-planet ships/owner timeline
      ├ reserve / available     … _compute_defense_buffers の出力
      └ threatened_candidates   … fall_turn が読める惑星
  └ plan_moves(world)
      └ collect_missions()      … reinforce / capture / swarm / crash / harass を score 順にコミット
      └ _process_*_mission()    … 各 mission に対して append_move を呼ぶ
      └ emit_followup / emit_evacuation / emit_rear_guard
      └ _enforce_inventory_cap
```

すでに `reserve` (留め置く下限) は計算されているので、STAY 判定は **「reserve を超えて available を temporarily 0 に縛る」per-source 上書き** として最小侵襲に挿入できる。

## STAY 判定の設計

### 仕組み

`baseline/missions/stay.py` を新設し、以下を返す:

```python
@dataclass
class StayDecision:
    src_id: int
    kind: str            # "defense" or "burst"
    held_ships: int      # この src で送出を抑制する ship 数 (上限)
    score: float         # ログ用、判定自体は kind ごとの threshold 比較
    reason: str          # 説明文 (スナップショットテストでも検証可)
```

`plan_moves` 冒頭で `stay_holds: dict[int, int] = build_stay_holds(world, planned_commitments, modes)` を呼び、`source_attack_left(src_id)` を以下に変更:

```python
def source_attack_left(src_id: int) -> int:
    raw = world.source_attack_left(src_id, spent_total)
    return max(0, raw - stay_holds.get(src_id, 0))
```

これにより capture / swarm / harass / followup / rear_guard 全てが STAY 分の ship を発射できなくなる。`reserve` ベースの defense は今まで通り効き、**STAY はそれに上乗せする一時的 hold**。

### Defense score (mission #1: 敵 ship のガード)

短期 horizon (`STAY_DEFENSE_HORIZON = 12` ターン) で、自惑星 `p` に到達する敵 fleet の総和 `incoming_enemy(p, h)` と、現状の駐留 + 生産から計算される手薄度を比較。`base_timeline` の `min_owned[p]` がすでに「この planet が一度でも持つ最小駐留」を返すので、それを再利用できる:

```
risk(p, h) = max(0, incoming_enemy(p, h) - (planet.ships + production*h - reserve_target[p]))
defense_score(src_p) = sum_{p in my_planets reachable from src} risk(p, h) * value(p)
```

`defense_score(src_p) >= STAY_DEFENSE_THRESHOLD` のとき、`held_ships = ceil(risk_total)` を STAY hold として登録。STAY された ship は次ターン以降に reinforcement / 自惑星防衛として使える。

実装上のポイント:

- `reserve` で既にカバーされている分を二重カウントしない: `risk` は `reserve` を控除した残差で計算する。
- 「敵が次ターン以降に **発射しうる**」量は OM (現状 default OFF) があれば `predicted_arrivals` で吸収できるが、case4 default 設定では OFF。**case6 では `OPPONENT_MODEL_ENABLED = True` にはせず**、純粋に「既に飛んでいる敵 fleet」(`arrivals_by_planet`) のみを使う ─ 仮説原文 ("敵の **出発済み** ship を元に") に忠実な実装にする。

### Burst-launch score (mission #2: 先ターン到着のための蓄積発射)

現行 capture/harass は「今 src にある ships」で `fleet_speed` を計算するため、`production = 5` の planet に 30 ship 残っているなら 1 ターン待って `(30+5)→35` で発射すれば速度が上がり ETA が早まる可能性がある。

定式化:

```
ships_now = available(src) - existing_holds
ships_next = ships_now + production
speed_now  = fleet_speed(ships_now)
speed_next = fleet_speed(ships_next)
ETA_now  = ceil(distance / speed_now)         + 0   (今ターン発射)
ETA_next = ceil(distance / speed_next)        + 1   (1ターン待ち)
burst_gain(src, target) = ETA_now - ETA_next  (>0 なら蓄積で先着)
```

`burst_gain >= STAY_BURST_MIN_GAIN (=1)` かつ ターゲットがその src の **best capture mission** に対して `value(target) > 0` のとき、`held_ships = ships_now` を STAY hold として登録 (= 今ターンその src からは何も発射しない)。

実装上のポイント:

- 1 src あたり 1 つの best target に対してだけ burst を発火 (狙わない target に巻き込まれて hold するのを避ける)。
- 連続 STAY を防ぐため `STAY_BURST_MAX_DELAY = 1` (= 1 ターン以上の連続待機は禁止)。連続待機状態は agent が stateless なので「self の last hold を覚える」必要があるが、最小実装では `step` の偶奇でガードするのではなく、**「待っても閾値ゲインが出る間は待つ」を許容**しつつ評価で副作用を見る (もし stalemate になるようなら iter2 で対処)。
- defense と burst が同じ src で競合する場合、`held_ships = max(defense_hold, burst_hold)` として大きい方を採用。

### Config

`baseline/core/config.py` に追加:

```python
STAY_ENABLED: bool = True
STAY_DEFENSE_ENABLED: bool = True
STAY_DEFENSE_HORIZON: int = 12
STAY_DEFENSE_THRESHOLD: float = 1.0   # risk を value 重み付けした単位での閾値
STAY_BURST_ENABLED: bool = True
STAY_BURST_MIN_GAIN: int = 1          # 1 ターン以上 ETA が短縮するときのみ burst
STAY_BURST_MIN_SHIPS: int = 8         # ships が少なすぎると速度差が出ないので skip
STAY_BURST_MAX_TARGET_TURNS: int = 30 # 遠すぎる target に対する burst は意味薄
```

`STAY_ENABLED = False` にすれば case6 = case4 完全等価に戻せる ablation スイッチ。

## 実装ステップ

1. `backend/pipeline/rulebase/case6/` を `case4` のコピーから派生 (copy済 — 直前のセットアップで実施)。
2. `case6/README.md` を case6 用に書き直す (派生元・差分・成績欄をプレースホルダで)。
3. `case6/baseline/core/config.py` に上記 STAY_* 定数を追加。
4. `case6/baseline/missions/stay.py` 新設: `StayDecision`, `build_stay_holds(world, planned_commitments, modes) -> dict[int, int]` を実装。
5. `case6/baseline/strategy.py` の `plan_moves` 冒頭で `build_stay_holds` を呼び、`source_attack_left` をラップする (差分は ~10 行)。`source_inventory_left` は変えない (reinforce 用なので STAY と独立)。
6. `case6/baseline/missions/__init__.py` で stay は **mission リストには追加しない** (`collect_missions` の戻り値に Mission として混ぜると `_process_*_mission` の枠に乗らない)。代わりに 5. の plan_moves での hook 経由で動かす。
7. `backend/src/dataset/selfplay/agents.py` の `AGENT_REGISTRY` に `"baseline_v6": "pipeline.rulebase.case6.baseline.agent:agent"` を追加。
8. `backend/pipeline/rulebase/README.md` の status table に case6 行を追記。
9. テスト追加: `backend/tests/pipeline/rulebase/case6/test_baseline_agent.py` (case4 のスモークテストを mirror) + `test_stay_decision.py` (build_stay_holds の単体: 敵 fleet 接近で defense hold 発火、ships 少+遠目 target で burst 発火、両方 OFF で hold = 0)。

## ローカル検証

1. `dev/test-backend` 全体 (format → lint → mypy → pytest) を pass させる。
2. dry-run import: `cd backend && uv run python -c "from pipeline.rulebase.case6.baseline.agent import agent; print(agent)"`
3. submit dry-run はあえて行わない (case6 は Kaggle に出さない、`.submitignore` 検証は不要)。

## 評価コマンド

`backend/pipeline/rulebase/case6/evaluation/compare_v4.py` を新設 (case4 の compare_v2.py を mirror、v4 ↔ v6 にリネーム)。

```bash
cd backend
uv run python -m pipeline.rulebase.case6.evaluation.compare_v4 -n 50 --seed 1000
```

`-n 50` × 2 seat = 100 戦。seed を 1000 起点にして case4 既存実験の seed と被らせない。

### 判定基準

- **case6 win rate ≥ 55%** (vs case4) → 採用候補、300 戦に伸ばして再評価を推奨。
- **45% ≤ case6 win rate < 55%** → noise の可能性大、`STAY_ENABLED=False` ablation も同条件で 100 戦回し、差分が seed variance を超えるか確認。
- **case6 win rate < 45%** → 設計に問題あり (例: STAY が長引きすぎてゲーム速度を失う、defense と burst の優先順位が逆など)、原因を result.md に記述して却下 / 再設計。

## リスク・既知の不確実性

- **stalemate リスク**: BURST が連続発火すると src が永遠に発射しない。最小実装では暗黙ループを許すが、評価で平均ターン長が著しく伸びていたら問題視。case4 の `compare_v2.py` 出力に `avg episode len` が含まれるので mirror した compare_v4.py でも同じく観測。
- **defense の二重カウント**: `reserve` が既に proactive_keep を計算しているので、`risk` を残差で取らないと過剰 hold になる。実装時に `world.reserve.get(src.id, 0)` を控除するロジックを必ず通す。
- **agent stateless 制約**: `_OM_STATE` のように module global を使うのは可能だが、STAY 用 state を持つと multi-process 評価で再現性が落ちる。今回は state を持たず、毎ターン obs だけから判定する純関数で済ませる。
- **case4 既存テストとの差分**: case4 の snapshot test は case6 にコピーしただけだと期待 action が壊れる可能性が高い (STAY が走ったら出力が変わる)。case6 の snapshot は **STAY 後の出力を初回キャプチャ**としてコミットする。

## 非ゴール (この実験では絶対にやらない)

- Kaggle 提出 (`dev/submit submit` / `kaggle competitions submit`)
- Vast.ai GPU 学習 (rulebase は学習なし)
- case4 / 既存 case の挙動変更
- Kaggle publicScore による評価 (memory 既知: publicScore は不安定指標)

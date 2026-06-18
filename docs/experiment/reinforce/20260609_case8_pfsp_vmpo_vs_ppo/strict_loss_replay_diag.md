# strict_v1 敗北の turn-level 診断 (replay分析)

> 関連: ladder17_result.md / hypotheses.md
> 方法: ladder11 best.pt (case8 agent, 全run最高 full 0.81) vs jax_v1 (=in-rollout strict_v1) を
>       kaggle_environments で N=4 試合、env.steps から序盤80手の planet/ship/fleet を seat別集計
> 日付: 2026-06-15

## 結果

outcomes: **0勝4敗** (held-out と整合)。
planet-lead 喪失 turn は表示上 turn0 だが、これは reset 直後 obs の全planet列挙アーティファクト
(無視)。turn5 以降の実体:

| turn | A_plt | S_plt | A_ship | S_ship | A_flt | S_flt |
|---|---|---|---|---|---|---|
| 5  | 1.0 | 1.2 | 7.0   | 7.8   | **2.0** | 1.0 |
| 10 | 1.2 | 1.8 | 6.5   | 8.0   | **3.0** | 1.8 |
| 20 | 2.0 | 2.5 | 13.2  | 29.2  | **10.0**| 2.8 |
| 30 | 2.8 | 2.5 | 25.0  | 19.2  | **18.0**| 7.5 |
| 40 | 3.0 | 4.2 | 54.8  | 37.5  | 15.0 | 11.2 |
| 50 | 4.0 | 7.0 | 146.0 | 100.2 | 12.8 | 12.2 |
| 60 | 5.0 | 9.5 | 267.2 | 187.8 | 10.2 | 15.5 |
| 70 | 5.2 | 13.2| 367.2 | 404.2 | 13.2 | 13.0 |

## 診断 (campaign の前提を覆す重要所見)

1. **agent は過剰に launch するが捕獲に変換できない**: turn20-30 で agent は strict の
   3-4倍の fleet を出している (10 vs 2.8 / 18 vs 7.5) = launch スパム。
2. **大量の fleet にもかかわらず planet 捕獲は少ない**: turn30 まで planet数はほぼ互角
   (2.8 vs 2.5) だが、turn40-70 で strict が引き離す (7 vs 4 → 13 vs 5)。agent の多数の
   fleet が territory に変換されない (target選択/ship配分が非効率)。
3. **差は序盤(0-20)でなく中盤(40-70)で開く**: turn30 まで planet 互角。strict の
   fleet効率 (少ない launch で確実な捕獲) が turn40以降 複利的に territory差へ。

## 結論: 「序盤で崩される」仮説は不正確

真の敗因は **fleet配分/target選択の非効率** — agent は小規模 fleet を乱発し捕獲に至らず、
strict は少数の決定的 launch で確実に捕獲。reverse-curriculum が「効いた」ように見えたのは、
strict の複利的効率優位が蓄積する中盤を warmup でスキップしていたから (序盤tempo問題では
なかった)。held-out が 0% なのは、この非効率が full strict 戦で最後まで覆らないため。

## 次の一手 (候補)

- **(本命) target-選択/ship-配分の reward 整形**: 「launch 数」でなく「捕獲成功 (planet
  奪取)」に報酬を寄せる。現 shaping は ratio (material比) で launch スパムを抑止しない。
  捕獲イベント報酬 + 無駄 launch ペナルティ で「少数精鋭 launch」を促す。
- **(代替) action空間/no-op bias 見直し**: no_op_bias=8.0 でも turn20で10 fleet/turn は過剰。
  launch 閾値を上げ「打つべき時だけ打つ」方向へ。
- **(代替) strict の fleet効率を模倣 (BC補助)**: strict の launch タイミング/規模を教師に。

## 追加診断: 角度ミス vs ship不足 の切り分け (N=3, 3031 fleet samples)

ユーザー質問「捕獲できないのは角度で外しているからか?」への定量回答:

| 指標 | 値 | 解釈 |
|---|---|---|
| **AIM error** (最寄り非所有planetへの方位差) | median **53.9°**, <15°(命中)=27%, >45°(大外し)=**55%** | 過半が目標から45°以上ずれ。角度も外している |
| **SHIP充足** (fleet ships / 目標 ships) | median **0.18**, ≥1.0(奪取可)=**13%**, <0.5(絶望)=**76%** | 76%が必要 shipの半分未満。圧倒的にship不足 |
| **LAUNCH規模** (1発のship数) | median 5, ≤2ships(極小)=31% | 小規模乱発も裏付け |

### 回答: 「両方、ただし ship不足が主因」
- **角度**: 55%が>45°ずれ = 確かに外している (target未確定のまま発射 or 軌道予測ミス)。
- **ship不足 (主因)**: 仮に角度が合っても **87%の fleet は目標を奪取できる ship を積んで
  いない** (median 0.18 = 目標の18%の戦力)。13%しか「取れる」規模がない。
- → agent は **「小規模 fleet を不正確な角度で乱発」** している。捕獲が成立しないのは
  当然。strict は逆に少数の fleet を正確な角度・十分な ship で送り確実に奪取。

### 設計含意 (次対策の根拠)
単純な「捕獲報酬」だけでは不足。**ship配分 (1目標に十分な戦力を集中) と角度精度の両方**を
改善する必要。candidate 船数 head が小さい値に張り付き、angle が target に解かれていない
可能性 → (a) 捕獲成功報酬で「奪取できる規模で送る」を学習させる + (b) 無駄(過小/大外し)
launch を罰する、を併せた reward 整形が筋。

## 角度の実装確認 (ユーザー指摘: 角度はモデルでなくルールベース計算)

確認結果、**角度はルールベースだが、訓練と提出で別ロジックを使う train/eval 不一致** がある:

| 経路 | 角度計算 | 軌道予測(intercept) |
|---|---|---|
| **訓練 rollout** (`policy/sampling_jax.py` `sampled_action_to_env_actions`, L146) | `atan2(target_now - source_now)` | **無し (現在位置へ直射)** |
| **提出/eval** (`policy/decoder.py`, L19-20) | `aim_with_prediction` | **有り (軌道先読み intercept)** |

- env はfleetを **固定方位で直進発射** (step.py L237: `start = planet + (cos,sin)·offset`、以後等速直進)。
  → 動く(orbit)planet には「現在位置へ直射」だと到着時にズレる = 系統的 miss。
- strict_v1 は `aim_with_prediction_jax` で **先読み intercept** (planet を flight time 分回して
  そこを狙う)。RL 訓練は先読み無しの直射。**これが 55% 大外し + 非捕獲の主因の一つ**。
- 重要: case8 policy には既に **pure-JAX 先読み解法 `geometry.aim_with_prediction`** があり
  提出経路 (decoder.py) は使っているが、**JAX 訓練 rollout (sampling_jax.py) はバイパス**して
  直射している。EnvState には必要な軌道情報 (planet_initial_xy, angular_velocity,
  planet_is_rotating, comet paths) が全て在り、JAX 先読み aim へ置換可能。

### 結論の更新: 真因は「train/eval の aim 不一致」
agent は **訓練中ずっと先読み無しの直射 aim で学習** (動く planet を外す) しているため、
target選択/ship配分をいくら学んでも捕獲に至らない。提出時だけ正しい intercept aim に
切り替わる = 訓練と評価で別物。これは reward/curriculum より優先度の高い構造バグ。

### 次の一手 (最優先・確信度高)
**rollout decoder (sampling_jax.py) の aim を先読み intercept 化** (python-to-jax):
`geometry.aim_with_prediction` 相当を JAX 化し `sampled_action_to_env_actions` の
naive atan2 を置換。これで訓練 aim が eval と一致し、agent は「動く planet に当たる」
前提で target/ship を学べる。parity test で eval decoder と一致を確認。これと並行して
ship配分 reward (捕獲報酬) は二次対策。

## Artifacts

- diag scripts: `/tmp/strict_opening_diag.py` (turn-level), `/tmp/strict_aim_diag.py` (aim/ship)
- model: ladder11 best.pt
- 関連コード: `policy/sampling_jax.py` (訓練aim, 要修正), `policy/decoder.py` +
  `policy/geometry.py:aim_with_prediction` (eval aim, 先読み有り)

# Reinforce/case6 — PFSP f_hard prioritized sampling (iter3)

> 作成日: 2026-05-28
> 仮説 ID: H4 (P2, depends on H1+H2)
> hypotheses.md: docs/experiment/reinforce/20260527_case6_pfsp/hypotheses.md
> 関連: iter2_plan.md / iter2_result.md / iter2_analysis.md
> スコープ: opponent 選択を一様 → f_hard=(1−x)^p の優先度 sampling に置換 (勝てない相手を優先)

## 仮説 (Hypothesis)

opponent 選択を一様抽出 (H2) から PFSP `f_hard(x)=(1−x)^p` (x=現 agent の対相手勝率) の
優先度 sampling に置換。勝てない相手 (baseline_jax_full や強い pool snapshot) を高確率で
選ぶ。— H2 は vs full が 0.359 で頭打ち。難敵への露出が一様 mix では薄いのが原因。
AlphaStar 主手法で難敵に学習を集中させ、vs full の到達点を H2 (0.359) より押し上げる。

## 既存コードの現状 (from Step 1 / iter2)

- `training/train_jax.py` (H2): `_OpponentPool` (FIFO cap 5) + late で pool snapshot
  (self_snapshot) / baseline_jax_full を `late_full_prob=0.5` の一様確率で選択。
  pool.sample も一様。各相手への勝率は追跡していない。
- iter2 所見: vs self_snapshot=0.828 / vs full=0.274、vs full は 0.138→0.359 と上昇するが
  頭打ち気味。難敵 (full) への露出を増やせば伸びしろを取れる (iter2_analysis.md)。
- opp_model 経路・rollout_jax は H1/H2 で完成済。H4 は **host 側の opponent 選択ロジック**
  のみ変更 (rollout_jax は不変)。

## スコープ (Scope)

- 変更ファイル:
  - `bot/pipeline/reinforce/case6/training/train_jax.py`
    — 各 opponent (各 pool snapshot + baseline_jax_full) の **直近勝率 EMA** を host 側で追跡。
      late iter の opponent 選択を一様 → `f_hard(x)=(1−x)^p` 重みの sampling に置換。
      `_OpponentPool` に per-entry 勝率トラッキングを追加 (または別 `_PrioritizedSelector`)。
      勝率は各 iter の win_rate を該当 opponent に帰属して EMA 更新。
  - `bot/pipeline/reinforce/case6/configs/kaggle_jax_train_h4.yaml` (新規)
    — H2 config 複製 + `opponent_pool.priority: f_hard`, `priority_p: 2.0`,
      `priority_ema: 0.7` を追加。iterations 100 / episodes 64 維持 (コスト方針)。
  - `bot/src/gpu/runpod/config/cases.py` — `reinforce_case6_kaggle_jax_train_h4` stage 登録。
- ハイパーパラメータ / config:
  - 新規: `priority: f_hard` / `priority_p: 2.0` (=(1−x)^2) / `priority_ema: 0.7`
  - H2 と共通: iterations 100, episodes_per_iter 64, switch_iter 5, pool cap 5, K=10
  - 比較基準: H2 (uniform mix) の vs full 0.359 / last10 0.661
- データセット / 特徴量変更: なし。

## 実装ステップ (Implementation outline)

1. `train_jax.py`: `_PrioritizedOpponentSelector` を追加 — entries =
   [baseline_jax_full] + pool snapshots。各 entry に勝率 EMA を保持。
   `sample(rng)` は `w_i = (1 − x_i)^p` (x_i=EMA勝率、未対戦は x=0.5 初期化) で確率抽出。
2. late iter: selector.sample() で opponent (full or pool snapshot) を選び、
   `_run_iter` 実行後に得た win_rate でその entry の EMA を更新。
3. pool push (K iter) 時に新 snapshot を selector に entry 追加 (初期勝率 0.5)。
4. config `kaggle_jax_train_h4.yaml` を priority=f_hard で新規作成。cases.py に stage 登録。
5. テスト: selector の重み計算 ((1−x)^p) と勝率更新の単体テストを case6 unit に追加。

## 検証方法 (Validation method)

### スキップする検証 (from hypotheses.md skip list)
- **ローカル self-play 300 対戦は行わない** — 採否は ① vs baseline_jax_full の last10 / 上昇
  トレンドが H2 (0.359) を超えるか、② entropy 収束を主軸。100 戦・20 戦は参考値。
- Kaggle publicScore / skill rating は引用しない。n<300 で結論を出さない。

### 実施する検証
- ローカル: `dev/test-bot` + `uv run --directory bot pytest tests/unit/pipeline/reinforce/case6 -x`
  (selector 重み・EMA 単体テスト)。
- smoke (必須): priority=f_hard の smoke config で 6-iter 完走、selector が難敵を優先
  抽出し reward NaN なしを確認。
- リモート: `dev/runpod train <commit> --case reinforce_case6_kaggle_jax_train_h4`。
  iterations 100 / episodes 64 で **~$1.0 目標 (RTX 4090)**。uptime 手動 cost 監視。
- 評価: 主軸 = vs baseline_jax_full の last10 win + slope が H2 を上回るか。
  ② vs 初期 snapshot・③ vs baseline_v1 20 戦は参考値。
- 分析: replay 分析は metrics 主体 skip mode (JAX rollout は in-memory)。

## リスク / 既知の不確実性

- **難敵集中の副作用**: full ばかり選ぶと弱い相手への対応を忘れる (catastrophic forgetting)。
  → full への偏りが強すぎたら H6 (混合比) で調整。EMA で動的に再配分されるので緩和は効く想定。
- **勝率推定の分散**: episodes 64 / iter だと per-opponent 勝率の EMA がノイジー。
  priority_ema=0.7 で平滑化するが、p を上げすぎると 1 相手に張り付く。
- **H2 比較の交絡**: H4 は選択ロジックのみ変更 (他 hyperparam は H2 と同一) なので、
  vs full トレンドの差は優先度 sampling の純効果と解釈できる。

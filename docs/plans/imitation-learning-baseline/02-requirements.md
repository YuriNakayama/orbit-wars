# Imitation Learning Baseline (case3) — Requirements Definition

## 背景と目的

Kaggle [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) 向けに、過去ログ (`data/kaggle_episodes/matches/replays/`) からの **行動クローニング (Behavior Cloning, BC)** で動く提出可能な PyTorch エージェントを `pipeline/case3/` に実装する。本ケースの位置付けは以下の三本柱:

1. **学習エージェントの土台確立** — case4 以降の RL/Self-play 実験のための再利用可能な学習基盤を構築する。
2. **case1 (rule-based) を上回る** — Kaggle 上位リプレイから学習した IL モデルが、手書き戦略 (case1 baseline) と互角以上の勝率 (≥50%) を達成することを示す。
3. **モデル設計パターンの実証** — DeepSets ベースの順不同 invariant な惑星 set encoder + 多ヘッド分類アーキテクチャが、Orbit Wars の連続座標点群環境で機能することを技術検証する。

提出のスコープは **ローカル評価まで** (Kaggle LB 提出は本 case では行わず、後続の意思決定に委ねる)。

## ユーザーストーリー

- **As a** ML エンジニア (このプロジェクトの開発者), **I want to** `data/kaggle_episodes/` のリプレイから自動で学習用テンソルを生成したい, **so that** 何度でも前処理を再実行できる。
- **As a** Kaggle 提出者, **I want to** `pipeline/case3/main.py` を Kaggle 提出 tar.gz にパッケージ化して、`agent(obs)` が actTimeout=1s 以内に動くことを担保したい, **so that** 必要な時にすぐ実提出に切り替えられる。
- **As a** 戦略実験者, **I want to** `case3_il_v1 vs case1_baseline_v1` のローカル N 戦勝率を 1 コマンドで取得したい, **so that** 学習モデルの品質を rule-based と並べて比較できる。
- **As a** 後続 case4/5 の実装者, **I want to** case3 のコード (データセット, モデル, 学習ループ) を **case3 内に閉じ込めた** まま参照・コピーしたい, **so that** case3 を改変しても他 case が壊れない。

## 機能要件

1. **データ前処理 CLI** (`pipeline/case3/training/preprocess.py`)
   - `data/kaggle_episodes/matches/index.parquet` から rating_mu 上位 25% かつ winner 側のフレームをフィルタ。
   - 1v1 / ffa4 両モード対応 (player perspective を winner 側に正規化)。
   - 各リプレイから (obs_features, action_targets) のテンソルを抽出し、`data/lake/case3/train.parquet` / `data/lake/case3/val.parquet` に保存 (90/10 split, episode 単位)。
   - 既存処理済み episode はスキップ (idempotent)。

2. **学習スクリプト** (`pipeline/case3/training/train.py`)
   - 設定は `pipeline/case3/configs/il_baseline.yaml` から読み込み (lr, batch_size, epochs, model dim 等)。
   - PyTorch CPU 学習 (mac で mps, Linux で cuda が見えれば自動利用、無くても動く)。
   - 学習中は loss/精度を stdout に構造化ログ出力 (wandb 等の外部依存なし)。
   - 各エポック終了時に検証 loss / top-1 accuracy を報告し、ベスト重みを `pipeline/case3/policy/weights.pt` に保存。

3. **推論エージェント** (`pipeline/case3/main.py` + `pipeline/case3/policy/`)
   - `pipeline/case3/main.py` は 20 行程度の Kaggle エントリポイント (`Path.cwd()` ベースの sys.path 注入、相対 import 規約準拠)。
   - `policy/agent.py` で `agent(obs)` を提供: weights.pt をモジュールロード時に 1 回だけ読み、推論時は torch.no_grad + greedy argmax + 有効ターゲットマスクで `[[from_id, angle, num_ships], ...]` を返す。
   - angle は `policy/decoder.py` の `aim_with_prediction(src, target)` で決定論的に再構成 (連続値回帰を回避)。

4. **vs baseline 評価スクリプト** (`pipeline/case3/evaluation/eval_vs_baseline.py`)
   - `src/env/runner.py` の `run_episodes()` を呼び、`case3_il_v1` を `case1_baseline_v1` と N 戦対戦させる。
   - 出力: 勝率, draw 率, 平均ターン数, p95 推論時間。
   - `src/env/agents.py` の `AGENT_REGISTRY` に `"case3_il_v1": "pipeline.case3.policy.agent:agent"` を追加 (既存 case と同じ規約)。

## 非機能要件

- **コード独立性 (重要)**: `pipeline/case3/` は `pipeline/case0/`, `pipeline/case1/`, `pipeline/case2/` のいずれにも import 依存しない。逆向きの依存も発生させない。共通必要な関数 (Planet/Fleet 型、aim_with_prediction の数学) は **case3 配下に独立コピー** する。`src/env/`, `src/submit/` の汎用基盤への依存のみ許容。
- **推論時間**: 1 ターンあたり推論 < 100ms (Kaggle actTimeout=1s に対し 10x 余裕)。`turn_p95` で計測。
- **モデルサイズ**: weights.pt < 5MB。Kaggle 提出 tar.gz 全体で < 10MB。
- **学習再現性**: torch / numpy / random の全 seed を `il_baseline.yaml` で固定。同 seed で同じ最終 loss が再現する。
- **推論決定性**: 同一 obs に対して同一 action を返す (snapshot test で fixate)。
- **Kaggle 提出規約**: `.claude/rules/pipeline.md` を満たす — `Path.cwd()` ベース sys.path、サブパッケージ内は相対 import、`pipeline/.submitignore` に `training/`, `evaluation/`, `configs/` を追記。
- **依存追加**: `pyproject.toml` に PyTorch (CPU 版) を追加。Kaggle ランタイムには既存同梱のため、追加コードは小さい。
- **テスト**: `dev/test-backend` (ruff + mypy + pytest) フルレーンを通す。`tests/pipeline/case3/` に最小スイート (snapshot, データセット形状, agent legality)。

## 評価ターゲット (合格基準)

- **vs case1_baseline_v1 勝率 ≥ 50%** (1v1, 100 戦, seed 0..99)。
- **BC 検証 loss が学習中に単調 (またはほぼ単調) に減少** することを学習曲線で確認。
- **推論決定性テスト**: snapshot 化した obs に対して action JSON が完全一致。
- **`dev/test-backend` 全レーン PASS**。

## スコープ外

- **RL fine-tuning, DAgger, self-play augmentation**: case4 以降に分離。MVP は純粋 BC のみ。
- **Kaggle 実提出 / LB チューニング**: クォータ消費を伴う実提出は本 case では行わない。`uv run python -m submit submit case3 --dry-run` での validation のみ任意で実施可能。
- **アンサンブル / 複数モデル切替**: シングルモデル 1 本に集中。
- **Hyperparameter 自動探索 (Optuna 等)**: 手動グリッド/ベスト案を yaml に固定するのみ。
- **可視化 / ノートブック / EDA**: `pipeline/case1/eda/` のような探索ツールは本 case では作らない (必要なら後続)。
- **共通ライブラリ抽出**: case0/1/2/3 共通の `src/features/` 等への抽出は行わず、case3 内自己完結。後続で必要が見えてからリファクタ。

## 用語集

| Term | Description |
|------|-------------|
| BC (Behavior Cloning) | 状態 → 行動の教師あり学習。報酬不要。本 case のメイン手法。 |
| Expert / Demonstrator | 学習教師となる Kaggle 上位プレイヤー (rating_mu top 25%)。 |
| DeepSets | 順不同集合に対する invariant ネットワーク (Zaheer 2017)。各要素を MLP で embed → mean/sum pool。 |
| from_planet / target_planet | action `[from_id, angle, num_ships]` の発射元・到達先惑星 ID (0〜35)。 |
| ships_bucket | num_ships を 5 段階 (例: min_capture, 25%, 50%, 75%, all) に量子化したクラス。 |
| `aim_with_prediction` | 既存の Orbit Wars geometry 関数: src と target から発射 angle を決定論的に算出。 |
| snapshot test | 固定 obs に対して action JSON 完全一致を assert する決定性テスト。 |
| `data/lake/case3/` | 本 case 専用の前処理済みテンソル parquet 出力先。 |

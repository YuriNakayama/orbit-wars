# Imitation Learning Baseline (case3) — Architecture Design

## 全体図

```
                         ┌─────────────────────────────────────────┐
                         │  data/kaggle_episodes/matches/          │
                         │   ├─ index.parquet (rating, winner)     │
                         │   └─ replays/*.json.gz (obs/action)     │
                         └─────────────────┬───────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ pipeline/case3/training/preprocess.py                                   │
│   1. parquet で rating_mu top25% × winner フィルタ                      │
│   2. replay.json.gz を stream read → (obs, action) フレーム抽出          │
│   3. Planet/Fleet → per-planet 特徴ベクトル (8ch) + global (6ch)         │
│   4. action → (from_id, target_id, ships_bucket) 3 ラベル               │
│   5. 1 フレーム = 1 行の parquet を data/lake/case3/{train,val}.parquet  │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ pipeline/case3/training/train.py                                        │
│   configs/il_baseline.yaml → PyTorch DeepSets model → BC 学習            │
│   ベスト重み → pipeline/case3/policy/weights.pt                         │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ pipeline/case3/main.py  ← Kaggle submission entry point                 │
│   sys.path.insert(0, str(Path.cwd()))                                   │
│   from policy.agent import agent                                        │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ pipeline/case3/policy/agent.py                                          │
│   weights.pt ロード (モジュール初期化時 1 回)                           │
│   agent(obs) → featurize → model.forward → decode → [[from, angle,     │
│   num_ships], ...]                                                      │
└─────────────────────────────────────────────────────────────────────────┘

評価時:
┌─────────────────────────────────────────────────────────────────────────┐
│ pipeline/case3/evaluation/eval_vs_baseline.py                           │
│   src/env/runner.run_episodes(RunSpec(                                  │
│     agents=("case3_il_v1", "baseline_v1"), mode="1v1", episodes=100))   │
│   → win rate, turn p95, draw rate                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

## ディレクトリ構成

```
pipeline/case3/
├── __init__.py
├── main.py                         # Kaggle entry (~20行, Path.cwd() ベース)
├── README.md                       # case3 独自の手順書
├── policy/                         # ★ 提出物 (.submitignore 対象外)
│   ├── __init__.py
│   ├── agent.py                    # agent(obs) エントリポイント
│   ├── featurizer.py               # obs → torch.Tensor (per-planet, global)
│   ├── model.py                    # DeepSets モデル (PyTorch)
│   ├── decoder.py                  # (from, target, ships_bucket) → action list
│   ├── geometry.py                 # aim_with_prediction (case3 独立コピー)
│   └── weights.pt                  # 学習済み重み (<5MB)
├── training/                       # 開発用 (.submitignore 対象)
│   ├── __init__.py
│   ├── preprocess.py               # replay → parquet tensor
│   ├── dataset.py                  # torch Dataset from parquet
│   ├── train.py                    # BC 学習ループ
│   └── losses.py                   # 3-head cross entropy + 重み
├── evaluation/                     # 開発用 (.submitignore 対象)
│   ├── __init__.py
│   └── eval_vs_baseline.py         # vs case1 勝率計測
└── configs/                        # 開発用 (.submitignore 対象)
    └── il_baseline.yaml            # model/train/data HP 設定
```

**重要原則**: `policy/` 配下の全ファイルは `pipeline/case0/`, `case1/`, `case2/` への import 依存を持たない。`geometry.py` は case1 の aim_with_prediction を **独立コピー** する (case 間結合を避けるため)。

## Backend (学習・推論モジュール) 設計

### policy/featurizer.py — 観測→テンソル

```python
@dataclass(frozen=True)
class BatchFeatures:
    planet_feats: torch.Tensor       # (B, P_max, 8)  P_max=36
    planet_mask: torch.Tensor        # (B, P_max) 真=存在
    global_feats: torch.Tensor       # (B, 6)
    my_planet_ids: torch.Tensor      # (B, P_max) bool: player 所有
    valid_target_mask: torch.Tensor  # (B, P_max) bool: 到達可能

def featurize(obs: dict, arrivals_by_planet: dict | None = None) -> BatchFeatures:
    ...
```

- **Per-planet 特徴 (8 次元)**: `x, y, ships, production, radius, is_self (0/1), is_enemy (0/1), is_neutral (0/1)`。`x/y` は `boardSize=100.0` で正規化。
- **ETA チャンネル (ships_in_{friendly,enemy,neutral}_fleet)**: 各惑星への到着 fleet を owner 別に集計 → 追加 3 次元。合計 **11 次元/惑星**。
- **Global 特徴 (6 次元)**: `step/TOTAL_STEPS, my_total_ships, enemy_total_ships, my_production, enemy_production, num_active_players`。全て -1〜+1 に正規化。
- 推論時は `batch_size=1` で呼ばれる。

### policy/model.py — DeepSets

```python
class DeepSetsPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # Per-planet encoder (shared MLP)
        self.phi = nn.Sequential(
            nn.Linear(cfg.planet_in_dim, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
        )
        # Global summary encoder
        self.psi = nn.Sequential(
            nn.Linear(cfg.global_in_dim, cfg.hidden),
            nn.GELU(),
        )
        # Joint head
        self.joint = nn.Sequential(
            nn.Linear(cfg.hidden * 3, cfg.hidden),  # [planet, pooled, global]
            nn.GELU(),
        )
        # 3 heads
        self.from_head   = nn.Linear(cfg.hidden, 1)          # per-planet logit
        self.target_head = nn.Linear(cfg.hidden, 1)          # per-planet logit
        self.ships_head  = nn.Linear(cfg.hidden, cfg.ships_buckets)  # per-planet
    def forward(self, feat: BatchFeatures) -> PolicyOutput:
        # phi: (B, P, H); pooled: (B, H) = masked mean
        # joint_in: concat(per_planet_phi, pooled_expanded, global_expanded)
        # from_logits: (B, P) masked by my_planet_ids
        # target_logits: (B, P) (no-op は virtual slot P=P_max+1)
        # ships_logits: (B, P, K)
        ...
```

- **パラメータ規模**: hidden=64, layers=2 → 重み約 30K params ≈ 120KB (fp32)。`weights.pt` < 1MB。
- **推論コスト**: P=36 の単発 forward は <5ms on CPU。

### policy/decoder.py — 推論時の action list 再構成

```python
def decode(output: PolicyOutput, world: WorldSnapshot) -> list[list[int | float]]:
    actions: list[list[int | float]] = []
    from_probs = torch.sigmoid(output.from_logits)  # (P,)
    for planet_idx in world.my_planet_indices:
        if from_probs[planet_idx].item() < FROM_THRESHOLD:
            continue
        target_idx = masked_argmax(output.target_logits, world.valid_target_mask[planet_idx])
        if target_idx == NO_OP_SLOT:
            continue
        ships_bucket = output.ships_logits[planet_idx].argmax()
        num_ships = ships_from_bucket(ships_bucket, world.planets[planet_idx].ships)
        angle = aim_with_prediction(
            world.planets[planet_idx], world.planets[target_idx],
            num_ships, world.initial_by_id, world.ang_vel, world.comets, world.comet_ids,
        )
        if angle is None:
            continue
        actions.append([planet_idx, angle, num_ships])
    return actions
```

- **決定性**: greedy argmax + 固定閾値 → 同一 obs で同一 action。
- **Kaggle 1s 制約**: P=36 に対し逐次 decode は 36回のループだが各々軽量 (<100μs)、全体 <5ms。

### training/preprocess.py — replay → parquet

```python
def preprocess(
    kaggle_index_path: Path,
    replay_dir: Path,
    output_train: Path, output_val: Path,
    rating_quantile: float = 0.75, val_split: float = 0.1,
    modes: tuple[str, ...] = ("1v1", "ffa4"),
) -> PreprocessReport:
    meta = pl.read_parquet(kaggle_index_path / "mode=*/*.parquet")
    cutoff = meta["agent_0_rating_mu"].quantile(rating_quantile)
    meta_top = meta.filter(pl.col("mode").is_in(modes))
    # winner 側の episode_id/player を抽出
    episode_specs = _select_winner_frames(meta_top, cutoff)

    rows_train: list[dict] = []
    rows_val: list[dict] = []
    for spec in episode_specs:
        frames = _extract_frames(replay_dir / f"{spec.match_id}.json.gz", spec.winner_player)
        target = rows_val if _is_val_episode(spec, val_split) else rows_train
        target.extend(frames)
    pl.DataFrame(rows_train).write_parquet(output_train)
    pl.DataFrame(rows_val).write_parquet(output_val)
```

- 1 row = 1 フレーム: `{match_id, step, player, planet_x: list[float], planet_y: list[float], ..., target_action: list[(from, target, ships_bucket)]}`。
- action が複数出た場合: データセット側で展開 (1 フレーム = 複数行、または list 保持のまま dataset で分岐)。**採用**: action 1 個ごとに 1 行に展開 (no-op フレームも 1 行残す)。

### training/dataset.py — torch Dataset

```python
class CaseThreeDataset(Dataset[Sample]):
    def __init__(self, parquet_path: Path) -> None:
        self.df = pl.read_parquet(parquet_path)
    def __len__(self) -> int: return self.df.height
    def __getitem__(self, idx: int) -> Sample:
        row = self.df.row(idx, named=True)
        return Sample(
            planet_feats=torch.tensor(row["planet_feats"], dtype=torch.float32),
            planet_mask=torch.tensor(row["planet_mask"], dtype=torch.bool),
            global_feats=torch.tensor(row["global_feats"], dtype=torch.float32),
            from_label=row["from_label"],
            target_label=row["target_label"],
            ships_label=row["ships_label"],
        )

def collate(batch: list[Sample]) -> BatchedSample:
    # 惑星数は 36 で固定なので pad は不要 (zero-fill)
    ...
```

### training/train.py — BC 学習ループ

```python
def train(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    train_ds = CaseThreeDataset(cfg.train_parquet)
    val_ds = CaseThreeDataset(cfg.val_parquet)
    loader_train = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate)
    loader_val = DataLoader(val_ds, batch_size=cfg.batch_size, collate_fn=collate)
    model = DeepSetsPolicy(cfg.model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float("inf")
    for epoch in range(cfg.epochs):
        model.train()
        for batch in loader_train:
            out = model(batch.features)
            loss = compute_loss(out, batch.labels, weights=(1.0, 1.0, 0.5))
            opt.zero_grad(); loss.backward(); opt.step()
        val_loss, top1 = _validate(model, loader_val)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), cfg.weights_out)
        _log_epoch(epoch, loss.item(), val_loss, top1)
```

## データモデル

### data/lake/case3/train.parquet / val.parquet (新規)

| 列 | 型 | 説明 |
|----|----|------|
| match_id | str | 元 episode ID |
| step | int | ターン番号 (0〜499) |
| player | int | フレームのプレイヤー視点 (winner) |
| planet_feats | list[list[float]] | shape (36, 11) の per-planet 特徴 |
| planet_mask | list[bool] | shape (36,) 存在マスク |
| global_feats | list[float] | shape (6,) |
| my_planet_mask | list[bool] | shape (36,) 自陣惑星 |
| valid_target_mask | list[bool] | shape (36,) 有効ターゲット |
| from_label | int | 0〜35 (発射元 planet_id) |
| target_label | int | 0〜36 (36 = no-op, 0〜35 = target planet) |
| ships_label | int | 0〜4 (5 buckets) |

- **ships_bucket 定義**: `[exact_need, all_available, 50%, 75%, 25%]` の 5 クラス。デモ data の分布を Step 6 の Step 2 で確認してチューニング。
- **エピソード数**: rating top25% × 2 mode 対応で約 100-150 episodes × 平均 250 turns × 平均 2.5 actions/turn ≈ **60K-100K 行**。
- **サイズ見積**: 1 行 ~2KB × 100K ≈ **200MB**。`data/lake/` は gitignore なので許容。

### pipeline/case3/policy/weights.pt

- PyTorch state_dict (fp32, torch.save で pickle)。
- 推定サイズ: hidden=64, 2-layer DeepSets → **< 500KB**。

## Infrastructure

### pyproject.toml への追加

```toml
[project]
dependencies = [
    ...,
    "torch>=2.3.0",       # CPU 版、Kaggle ランタイムと整合
]
```

- **影響**: `uv sync` 時間が +30 秒程度。`dev/setup` への変更なし。
- mac M1/M2 では自動で mps 利用、それ以外の環境で CPU fallback。

### pipeline/.submitignore への追記

```
# case3 開発用
training/
evaluation/
configs/
```

(既に `eda/`, `notebook/`, `evaluation/`, `configs/` は記載済み → `training/` のみ新規追加)

### src/env/agents.py への追記

```python
AGENT_REGISTRY = {
    ...,
    "case3_il_v1": "pipeline.case3.policy.agent:agent",
}
```

### Kaggle ランタイム依存

- `torch` はプリインストール済み (Kaggle standard container)。tar.gz に同梱しない。
- `numpy` / `polars` も既存同梱。
- **モデル重みのみ** `pipeline/case3/policy/weights.pt` として tar.gz に含める。

## 外部 API

本 case では **新規の外部 API 依存なし**。Kaggle scraper は既存 `src/env/kaggle/` を必要に応じて流用 (データ追加取得時のみ)。

## 主要インターフェース

### agent(obs) の契約 (既存と同じ)

```python
def agent(obs: Any) -> list[list[int | float]]:
    """Return [[from_planet_id (int), angle (float, radian), num_ships (int)], ...]"""
```

### case3 内の層間契約

- `featurizer.featurize(obs) → BatchFeatures` (frozen dataclass of torch.Tensor)
- `model.DeepSetsPolicy.forward(BatchFeatures) → PolicyOutput`
- `decoder.decode(PolicyOutput, WorldSnapshot) → list[list[int | float]]`
- `training.dataset.CaseThreeDataset.__getitem__ → Sample` (frozen dataclass)

これらは case3 の 3-layer (feature → model → decode) を明示化し、単体テストを書きやすくする。

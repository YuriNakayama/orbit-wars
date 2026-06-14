"""Training case registry and path helpers for RunPod launches."""

from __future__ import annotations

from pathlib import Path

import typer

CASE_DEFAULTS: dict[str, dict[str, str]] = {
    "case1": {
        "stage": "train_imitation_case1",
        "train_module": "pipeline.imitation.case1.training.train",
        "config_arg": "",
        "preprocess_cmd": "",
        "canonical_weights": "bot/pipeline/imitation/case1/policy/weights.pt",
    },
    "case3": {
        "stage": "train_imitation_case3",
        "train_module": "pipeline.imitation.case3.training.train",
        "config_arg": "--config pipeline/imitation/case3/configs/il_phase2.yaml",
        "preprocess_cmd": (
            "pipeline.imitation.case3.training.preprocess "
            "--config pipeline/imitation/case3/configs/il_phase2.yaml"
        ),
        "canonical_weights": ("bot/pipeline/imitation/case3/policy/weights.pt"),
    },
    "case4": {
        "stage": "train_imitation_case4",
        "train_module": "pipeline.imitation.case4.training.train",
        "config_arg": "--config pipeline/imitation/case4/configs/il_case4.yaml",
        "preprocess_cmd": (
            "pipeline.imitation.case4.training.preprocess "
            "--config pipeline/imitation/case4/configs/il_case4.yaml"
        ),
        "canonical_weights": "bot/pipeline/imitation/case4/policy/weights.pt",
    },
    "case5": {
        "stage": "train_imitation_case5",
        "train_module": "pipeline.imitation.case5.training.train",
        "config_arg": "--config pipeline/imitation/case5/configs/il_case5.yaml",
        "preprocess_cmd": (
            "pipeline.imitation.case5.training.preprocess "
            "--config pipeline/imitation/case5/configs/il_case5.yaml"
        ),
        "canonical_weights": "bot/pipeline/imitation/case5/policy/weights.pt",
    },
    "case6": {
        "stage": "train_imitation_case6",
        "train_module": "pipeline.imitation.case6.training.train",
        "config_arg": "--config pipeline/imitation/case6/configs/il_case6.yaml",
        "preprocess_cmd": (
            "pipeline.imitation.case6.training.preprocess "
            "--config pipeline/imitation/case6/configs/il_case6.yaml"
        ),
        "canonical_weights": "bot/pipeline/imitation/case6/policy/weights.pt",
    },
    "case7": {
        "stage": "train_imitation_case7",
        "train_module": "pipeline.imitation.case7.training.train",
        "config_arg": "--config pipeline/imitation/case7/configs/il_case7.yaml",
        "preprocess_cmd": (
            "pipeline.imitation.case7.training.preprocess "
            "--config pipeline/imitation/case7/configs/il_case7.yaml"
        ),
        "canonical_weights": "bot/pipeline/imitation/case7/policy/weights.pt",
    },
    "case8": {
        "stage": "train_imitation_case8",
        "train_module": "pipeline.imitation.case8.training.train",
        "config_arg": "--config pipeline/imitation/case8/configs/il_case8.yaml",
        "preprocess_cmd": (
            "pipeline.imitation.case8.training.preprocess "
            "--config pipeline/imitation/case8/configs/il_case8.yaml"
        ),
        "canonical_weights": "bot/pipeline/imitation/case8/policy/weights.pt",
    },
    "case9_three_head": {
        "stage": "train_imitation_case9_three_head",
        "train_module": "pipeline.imitation.case9.training.train",
        "config_arg": (
            "--config pipeline/imitation/case9/configs/il_case9_three_head.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": (
            "bot/pipeline/imitation/case9/policy/weights_three_head.pt"
        ),
    },
    # case9_rulebase: 実 rulebase baseline_v1 の蒸留 (selfplay ミラー棋譜から BC)。
    # mart は local preprocess + dvc push 済みを pull する (preprocess_cmd なし)。
    # 蒸留クローンは reinforce/case8 の distilled opponent / bc_warmstart 教師に使う。
    "case9_rulebase": {
        "stage": "train_imitation_case9_rulebase",
        "train_module": "pipeline.imitation.case9.training.train",
        "config_arg": (
            "--config pipeline/imitation/case9/configs/il_case9_rulebase.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # case9_rulebase_v8: baseline_v8 (v1 に勝率 ~70-75%) の蒸留。v8 vs v1 混合
    # 対戦から teacher_agents=[baseline_v8] で v8 側のみ教師化 (Phase 1)。
    "case9_rulebase_v8": {
        "stage": "train_imitation_case9_rulebase_v8",
        "train_module": "pipeline.imitation.case9.training.train",
        "config_arg": (
            "--config pipeline/imitation/case9/configs/il_case9_rulebase_v8.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    "case9_candidate": {
        "stage": "train_imitation_case9_candidate",
        "train_module": "pipeline.imitation.case9.training.train",
        "config_arg": (
            "--config pipeline/imitation/case9/configs/il_case9_candidate.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": (
            "bot/pipeline/imitation/case9/policy/weights_candidate.pt"
        ),
    },
    "case9_candidate_ships": {
        "stage": "train_imitation_case9_candidate_ships",
        "train_module": "pipeline.imitation.case9.training.train",
        "config_arg": (
            "--config pipeline/imitation/case9/configs/il_case9_candidate_ships.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": (
            "bot/pipeline/imitation/case9/policy/weights_candidate_ships.pt"
        ),
    },
    "case9_template_ships": {
        "stage": "train_imitation_case9_template_ships",
        "train_module": "pipeline.imitation.case9.training.train",
        "config_arg": (
            "--config pipeline/imitation/case9/configs/il_case9_template_ships.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": (
            "bot/pipeline/imitation/case9/policy/weights_template_ships.pt"
        ),
    },
    "case9_dual": {
        "stage": "train_imitation_case9_dual",
        "train_module": "pipeline.imitation.case9.training.train",
        "config_arg": "--config pipeline/imitation/case9/configs/il_case9_dual.yaml",
        "preprocess_cmd": "",
        "canonical_weights": "bot/pipeline/imitation/case9/policy/weights_dual.pt",
    },
    "case9_per_planet": {
        "stage": "train_imitation_case9_per_planet",
        "train_module": "pipeline.imitation.case9.training.train",
        "config_arg": (
            "--config pipeline/imitation/case9/configs/il_case9_per_planet.yaml"
        ),
        # mart は事前に local で preprocess + DVC push 済 (7GB + 745MB)。
        # pod 側は dvc pull のみで mart を取得し、preprocess を再実行しない
        # (約 50 分節約 + stall watcher 回避)。
        "preprocess_cmd": "",
        "canonical_weights": (
            "bot/pipeline/imitation/case9/policy/weights_per_planet.pt"
        ),
    },
    "case10_candidate": {
        "stage": "train_imitation_case10_candidate",
        "train_module": "pipeline.imitation.case10.training.train",
        "config_arg": (
            "--config pipeline/imitation/case10/configs/il_case10_candidate.yaml"
        ),
        "preprocess_cmd": (
            "pipeline.imitation.case10.training.preprocess "
            "--config pipeline/imitation/case10/configs/il_case10_candidate.yaml"
        ),
        "canonical_weights": (
            "bot/pipeline/imitation/case10/policy/weights_candidate_ships.pt"
        ),
    },
    "case10_template": {
        "stage": "train_imitation_case10_template",
        "train_module": "pipeline.imitation.case10.training.train",
        "config_arg": (
            "--config pipeline/imitation/case10/configs/il_case10_template.yaml"
        ),
        "preprocess_cmd": (
            "pipeline.imitation.case10.training.preprocess "
            "--config pipeline/imitation/case10/configs/il_case10_template.yaml"
        ),
        "canonical_weights": (
            "bot/pipeline/imitation/case10/policy/weights_template_ships.pt"
        ),
    },
    # case10 data_volume_sweep iter1 (5 sweep points). All share the same
    # base mart (built once) and apply train_top_submission_limit in-memory
    # via index_filter. preprocess_cmd is empty: dvc pull supplies the mart
    # and the *_index.parquet files; no per-pod preprocess needed.
    "case10_sweep_top20": {
        "stage": "train_imitation_case10_sweep_top20",
        "train_module": "pipeline.imitation.case10.training.train",
        "config_arg": (
            "--config "
            "pipeline/imitation/case10/configs/data_volume_sweep/il_case10_sweep_top20.yaml"
        ),
        # preprocess は事前に専用 pod で base mart を生成 + DVC push 済。
        # 各 sweep pod は dvc pull のみで mart を取得して preprocess を skip。
        "preprocess_cmd": "",
        "canonical_weights": (
            "bot/pipeline/imitation/case10/policy/data_volume_sweep/weights_top20.pt"
        ),
    },
    "case10_sweep_top40": {
        "stage": "train_imitation_case10_sweep_top40",
        "train_module": "pipeline.imitation.case10.training.train",
        "config_arg": (
            "--config "
            "pipeline/imitation/case10/configs/data_volume_sweep/il_case10_sweep_top40.yaml"
        ),
        # preprocess は事前に専用 pod で base mart を生成 + DVC push 済。
        # 各 sweep pod は dvc pull のみで mart を取得して preprocess を skip。
        "preprocess_cmd": "",
        "canonical_weights": (
            "bot/pipeline/imitation/case10/policy/data_volume_sweep/weights_top40.pt"
        ),
    },
    "case10_sweep_top80": {
        "stage": "train_imitation_case10_sweep_top80",
        "train_module": "pipeline.imitation.case10.training.train",
        "config_arg": (
            "--config "
            "pipeline/imitation/case10/configs/data_volume_sweep/il_case10_sweep_top80.yaml"
        ),
        # preprocess は事前に専用 pod で base mart を生成 + DVC push 済。
        # 各 sweep pod は dvc pull のみで mart を取得して preprocess を skip。
        "preprocess_cmd": "",
        "canonical_weights": (
            "bot/pipeline/imitation/case10/policy/data_volume_sweep/weights_top80.pt"
        ),
    },
    "case10_sweep_top160": {
        "stage": "train_imitation_case10_sweep_top160",
        "train_module": "pipeline.imitation.case10.training.train",
        "config_arg": (
            "--config "
            "pipeline/imitation/case10/configs/data_volume_sweep/il_case10_sweep_top160.yaml"
        ),
        # preprocess は事前に専用 pod で base mart を生成 + DVC push 済。
        # 各 sweep pod は dvc pull のみで mart を取得して preprocess を skip。
        "preprocess_cmd": "",
        "canonical_weights": (
            "bot/pipeline/imitation/case10/policy/data_volume_sweep/weights_top160.pt"
        ),
    },
    "case10_sweep_topall": {
        "stage": "train_imitation_case10_sweep_topall",
        "train_module": "pipeline.imitation.case10.training.train",
        "config_arg": (
            "--config "
            "pipeline/imitation/case10/configs/data_volume_sweep/il_case10_sweep_topall.yaml"
        ),
        # preprocess は事前に専用 pod で base mart を生成 + DVC push 済。
        # 各 sweep pod は dvc pull のみで mart を取得して preprocess を skip。
        "preprocess_cmd": "",
        "canonical_weights": (
            "bot/pipeline/imitation/case10/policy/data_volume_sweep/weights_topall.pt"
        ),
    },
    # case10: data_volume_sweep recovery entry — the `<CASE>` placeholder in
    # onstart_lib expands to the registry key verbatim (not case_subdir), so
    # we need a registry key whose name matches the on-disk mart subdir
    # `data/mart/imitation/case10/` for the skip-preprocess path to work
    # after a base_preprocess pod left the mart on /persist but couldn't
    # DVC-push it (mart_dvc_persist looked at the wrong subdir and skipped).
    # preprocess_cmd="" so 50_preprocess.sh entirely skips that
    # block; train runs the template_ships head against the existing mart.
    "case10": {
        "stage": "train_imitation_case10",
        "train_module": "pipeline.imitation.case10.training.train",
        "config_arg": (
            "--config pipeline/imitation/case10/configs/il_case10_template.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": (
            "bot/pipeline/imitation/case10/policy/weights_template_ships.pt"
        ),
    },
    # case10_base_preprocess: produce the shared base mart in a dedicated
    # pod and DVC-push it. Reuses the case0 dummy train so the onstart
    # pipeline still has a valid train step; the actual deliverable is the
    # mart parquets uploaded via the standard 70_artifacts_and_dvc.sh path
    # (which dvc-pushes both the run_dir and the touched data/mart entries).
    "case10_base_preprocess": {
        "stage": "train_imitation_case10_base_preprocess",
        "train_module": "pipeline.imitation.case0.training.train",
        "config_arg": "--config pipeline/imitation/case0/configs/smoke.yaml",
        "preprocess_cmd": (
            "pipeline.imitation.case10.training.preprocess "
            "--config "
            "pipeline/imitation/case10/configs/data_volume_sweep/il_case10_base_full.yaml"
        ),
        "canonical_weights": "",
    },
    # case11: per_planet-only successor of case9_per_planet, refreshed on
    # the latest Kaggle episode lake. winners_only + top_team_rank=80 +
    # max_episodes=null (1v1, ~30,160 matches, 5.94M frames, ~21GB parquet).
    #
    # mart は事前に local で preprocess + DVC push 済。pod 側で parquet ->
    # per-column .npy 変換を onstart 内で実行 (preprocess_cmd で指定)。
    # 変換は 1-2 分で完了し、Dataset は mmap_mode="r" で読み込むため
    # OS page cache に委譲できる。174GB の npy は S3 push せず pod 内で
    # 都度生成 (push 10-40h を回避)。
    "case11": {
        "stage": "train_imitation_case11",
        "train_module": "pipeline.imitation.case11.training.train",
        "config_arg": (
            "--config pipeline/imitation/case11/configs/"
            "il_case11_per_planet_recent10d_rank30.yaml"
        ),
        "preprocess_cmd": "pipeline.imitation.case11.training.parquet_to_npy",
        "canonical_weights": "bot/pipeline/imitation/case11/policy/weights.pt",
    },
    # case11 smoke variant: 100 episodes (~25k train rows / ~3k val rows)
    # × 1 epoch で kernel e2e パイプラインを 10-15 分で検証する用。
    # mart は data/mart/imitation/case11_smoke/ に置かれている。
    "case11_smoke": {
        "stage": "train_imitation_case11",
        "train_module": "pipeline.imitation.case11.training.train",
        "config_arg": (
            "--config pipeline/imitation/case11/configs/"
            "il_case11_per_planet_recent10d_rank30_smoke.yaml"
        ),
        "preprocess_cmd": "pipeline.imitation.case11.training.parquet_to_npy",
        "canonical_weights": "",
    },
    # case0 = RunPod E2E smoke pipeline. NOT a real training case — the model
    # is a 200-param MLP on synthetic data, designed to finish in minutes so
    # the GPU basis itself can be verified end-to-end.
    "case0": {
        "stage": "train_imitation_case0",
        "train_module": "pipeline.imitation.case0.training.train",
        "config_arg": "--config pipeline/imitation/case0/configs/smoke.yaml",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # smoke_gpu: bare GPU-acquisition smoke test. NOT a training case — runs
    # pipeline/reinforce/_bench/gpu_smoke.py which prints CUDA availability /
    # GPU name / VRAM, does one tiny CUDA tensor op, and exits 0. Designed to
    # validate the RunPod basis end-to-end in seconds (no DVC pull / preprocess).
    "smoke_gpu": {
        "stage": "smoke_gpu",
        "train_module": "pipeline.reinforce._bench.gpu_smoke",
        "config_arg": "",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case1: PPO + BC warm-start on case9 per_planet weights.
    # `family` differentiates the on-disk run dir (data/output/models/reinforce/...).
    "reinforce_case1": {
        "family": "reinforce",
        "stage": "train_reinforce_case1",
        "train_module": "pipeline.reinforce.case1.training.train",
        "config_arg": "--config pipeline/reinforce/case1/configs/train.yaml",
        # BC weights are pulled via DVC by the standard 50_dvc_pull stage; no
        # per-pod preprocess needed (RL collects rollouts in-process).
        "preprocess_cmd": "",
        "canonical_weights": "bot/pipeline/reinforce/case1/policy/weights.pt",
    },
    # reinforce_case1_kaggle_smoke: Kaggle Kernel smoke variant. BC warm-start
    # disabled because the Kaggle Kernel template has no DVC pull cell to
    # fetch case9 per_planet BC weights. 6 iter x 4 ep — verifies that
    # reinforce/case1 boots and PPO update / artifact collection survive on
    # Kaggle GPU.
    "reinforce_case1_kaggle_smoke": {
        "family": "reinforce",
        "stage": "train_reinforce_case1_kaggle_smoke",
        "train_module": "pipeline.reinforce.case1.training.train",
        "config_arg": "--config pipeline/reinforce/case1/configs/kaggle_smoke.yaml",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce_case1_kaggle_jax_smoke: JAX-accelerated smoke variant of the
    # above. Uses train_jax.py end-to-end (rollout + PPO update inside a
    # single jit) so we can measure the README claim (17-18x speedup on RTX
    # 4090) on Kaggle T4x2. BC warm-start stays OFF for the same dataset
    # transport reason. Kaggle template auto-installs jax-cuda12 plugin +
    # cuDNN 9.8 wheels for reinforce_*_jax_* cases.
    "reinforce_case1_kaggle_jax_smoke": {
        "family": "reinforce",
        "stage": "train_reinforce_case1_kaggle_jax_smoke",
        "train_module": "pipeline.reinforce.case1.training.train_jax",
        "config_arg": "--config pipeline/reinforce/case1/configs/kaggle_jax_smoke.yaml",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # 50-iter Kaggle Kernel training variant (~27 min on T4x2).
    "reinforce_case1_kaggle_jax_train": {
        "family": "reinforce",
        "stage": "train_reinforce_case1_kaggle_jax_train",
        "train_module": "pipeline.reinforce.case1.training.train_jax",
        "config_arg": "--config pipeline/reinforce/case1/configs/kaggle_jax_train.yaml",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce_case1_jax_train: RunPod 用の JAX 学習 case。Kaggle 30h quota
    # を回避するため、kaggle_jax_train.yaml と同じ AA 構成 (300 iter, shaping=0.50)
    # を RunPod GPU で走らせる。RTX 3090 (Medium stock, $0.22/h) を想定。
    "reinforce_case1_jax_train": {
        "family": "reinforce",
        "stage": "train_reinforce_case1_jax_train",
        "train_module": "pipeline.reinforce.case1.training.train_jax",
        "config_arg": "--config pipeline/reinforce/case1/configs/kaggle_jax_train.yaml",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case2: head 構造変更実験用 case (case1 ベース + from_head 追加)。
    # per_planet head の NO_OP slot に per-source 発射推奨スコア (from_logit) を
    # bias として加算し、発射判断を明示的に学習する。最初の head 変更実験。
    "reinforce_case2_kaggle_jax_train": {
        "family": "reinforce",
        "stage": "train_reinforce_case2_kaggle_jax_train",
        "train_module": "pipeline.reinforce.case2.training.train_jax",
        "config_arg": "--config pipeline/reinforce/case2/configs/kaggle_jax_train.yaml",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case3: case2 の有用な学習レシピ (planets shaping=0.50 /
    # curriculum switch_iter=5 / lr 3e-5 線形 decay / 200 iter 長期学習) を
    # 継承しつつ from_head を除去した case (= case1 純正 head)。case2 の
    # head 変更効果を切り分ける ablation 兼、学習レシピの統合ベースライン。
    "reinforce_case3_kaggle_jax_train": {
        "family": "reinforce",
        "stage": "train_reinforce_case3_kaggle_jax_train",
        "train_module": "pipeline.reinforce.case3.training.train_jax",
        "config_arg": "--config pipeline/reinforce/case3/configs/kaggle_jax_train.yaml",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce_case3_kaggle_jax_smoke: short 6-iter JAX smoke for case3 used to
    # verify the RunPod / Kaggle wiring end-to-end (launch → train → artifact
    # collection) without spending a full 200-iter run. ~minutes on GPU.
    "reinforce_case3_kaggle_jax_smoke": {
        "family": "reinforce",
        "stage": "train_reinforce_case3_kaggle_jax_smoke",
        "train_module": "pipeline.reinforce.case3.training.train_jax",
        "config_arg": "--config pipeline/reinforce/case3/configs/kaggle_jax_smoke.yaml",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case6: PFSP (Prioritized Fictitious Self-Play). case3 の学習
    # レシピを継承しつつ対戦相手構成のみ変更。H1 = curriculum late を
    # self_snapshot に差し替え、学習開始時点の凍結 self snapshot と自己対戦。
    "reinforce_case6_kaggle_jax_train_h1": {
        "family": "reinforce",
        "stage": "train_reinforce_case6_kaggle_jax_train_h1",
        "train_module": "pipeline.reinforce.case6.training.train_jax",
        "config_arg": (
            "--config pipeline/reinforce/case6/configs/kaggle_jax_train_h1.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce_case6 H2: PFSP snapshot pool (K=10 周期更新 + baseline_jax_full
    # 混合 late)。H1 の飽和を解消する狙い。iterations 100 / episodes 64 に軽量化。
    "reinforce_case6_kaggle_jax_train_h2": {
        "family": "reinforce",
        "stage": "train_reinforce_case6_kaggle_jax_train_h2",
        "train_module": "pipeline.reinforce.case6.training.train_jax",
        "config_arg": (
            "--config pipeline/reinforce/case6/configs/kaggle_jax_train_h2.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce_case6 H4: PFSP f_hard=(1−x)^p 優先 sampling。難敵 (full / 強い
    # pool snapshot) を優先選択し vs full の伸びしろを取る。iter 100 / ep 64。
    "reinforce_case6_kaggle_jax_train_h4": {
        "family": "reinforce",
        "stage": "train_reinforce_case6_kaggle_jax_train_h4",
        "train_module": "pipeline.reinforce.case6.training.train_jax",
        "config_arg": (
            "--config pipeline/reinforce/case6/configs/kaggle_jax_train_h4.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce_case6 python_v1: 本物 baseline_v1 を pure_callback で seat1 に
    # 注入 (opponent=python_v1, sha dc13a25)。train/eval gap を原理的に消す試行。
    # H4 (近似 rule 相手) が live v1 戦で 0/30 だった現状を打開する狙い。
    "reinforce_case6_kaggle_jax_train_python_v1": {
        "family": "reinforce",
        "stage": "train_reinforce_case6_kaggle_jax_train_python_v1",
        "train_module": "pipeline.reinforce.case6.training.train_jax",
        "config_arg": (
            "--config pipeline/reinforce/case6/configs/kaggle_jax_train_python_v1.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce_case6 pool_v1_h50: H6d (horizon 500→50, iter 30)。pool_v1 の 0/30
    # 棄却後の次仮説。reward 密度 10x で PPO approx_kl ~ 0 停止問題に対処。
    "reinforce_case6_kaggle_jax_train_pool_v1_h50": {
        "family": "reinforce",
        "stage": "train_reinforce_case6_kaggle_jax_train_pool_v1_h50",
        "train_module": "pipeline.reinforce.case6.training.train_jax",
        "config_arg": (
            "--config "
            "pipeline/reinforce/case6/configs/kaggle_jax_train_pool_v1_h50.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce_case6 pool_v1_shaping5: H6c (shaping_coef 0.5 → 5.0)。pool_v1 の
    # 0/30 棄却後、horizon は 500 のまま dense reward signal を 10x 強化。
    # H6d 失敗時の保険として用意、現時点では起動しない。
    "reinforce_case6_kaggle_jax_train_pool_v1_shaping5": {
        "family": "reinforce",
        "stage": "train_reinforce_case6_kaggle_jax_train_pool_v1_shaping5",
        "train_module": "pipeline.reinforce.case6.training.train_jax",
        "config_arg": (
            "--config "
            "pipeline/reinforce/case6/configs/kaggle_jax_train_pool_v1_shaping5.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce_case6 pool_v1_3stage: H6b 3 段 curriculum
    # (noop → baseline_jax_lite → python_v1)。pool_v1 系の「v1 切替で 0/8 全敗」
    # を中間に勝てる相手を挟んで解消する狙い。
    "reinforce_case6_kaggle_jax_train_pool_v1_3stage": {
        "family": "reinforce",
        "stage": "train_reinforce_case6_kaggle_jax_train_pool_v1_3stage",
        "train_module": "pipeline.reinforce.case6.training.train_jax",
        "config_arg": (
            "--config "
            "pipeline/reinforce/case6/configs/kaggle_jax_train_pool_v1_3stage.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce_case6 pool_v8: noop → python_v8 curriculum (case8 production champion)。
    # pool_v1 の v8 版。最強 baseline で学習し v1/v4 への汎化を試す。
    "reinforce_case6_kaggle_jax_train_pool_v8": {
        "family": "reinforce",
        "stage": "train_reinforce_case6_kaggle_jax_train_pool_v8",
        "train_module": "pipeline.reinforce.case6.training.train_jax",
        "config_arg": (
            "--config pipeline/reinforce/case6/configs/kaggle_jax_train_pool_v8.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce_case6 pool_v1: noop → python_v1 curriculum (5c8725cd)。
    # iter 0-4 noop で free 学習開始 → iter 5+ で本物 v1 戦。
    # python_v1 単独より cost は嵩むが、学習信号薄問題を回避する狙い。
    "reinforce_case6_kaggle_jax_train_pool_v1": {
        "family": "reinforce",
        "stage": "train_reinforce_case6_kaggle_jax_train_pool_v1",
        "train_module": "pipeline.reinforce.case6.training.train_jax",
        "config_arg": (
            "--config pipeline/reinforce/case6/configs/kaggle_jax_train_pool_v1.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce_case6_kaggle_jax_smoke: short JAX wiring smoke for case6 (verifies
    # the self_snapshot opponent path launches and collects artifacts on RunPod).
    "reinforce_case6_kaggle_jax_smoke": {
        "family": "reinforce",
        "stage": "train_reinforce_case6_kaggle_jax_smoke",
        "train_module": "pipeline.reinforce.case6.training.train_jax",
        "config_arg": (
            "--config pipeline/reinforce/case6/configs/kaggle_jax_smoke.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8: PFSP f_var + held-out=python_v8 (real rulebase case8) +
    # Elo の case7 コピー。`algo: ppo|vmpo` フラグで PPO (既存流用) と V-MPO (H1
    # 新規) を同一 harness で A/B する。pool=case1(lite/full)+case8(python_v8)+self。
    # h0_ppo_short: H0 wiring 検証 (PPO 短 run, held-out vs python_v8 + Elo 記録)。
    "reinforce_case8_kaggle_jax_train": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_kaggle_jax_train",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/h0_ppo_short.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce_case8_kaggle_jax_smoke: short wiring smoke (2 iter, algo=ppo).
    # Verifies the case7→case8 copy + algo flag + f_var 3-opp pool launch on RunPod.
    "reinforce_case8_kaggle_jax_smoke": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_kaggle_jax_smoke",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/h0_smoke.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 Phase 1 R1: PPO 実験条件探索。f_var priority_p を {1.0,2.0,4.0}
    # で A/B (他は ppo_base と同一: pool=full+lite+self, 20 iter,
    # held-out=baseline_jax_full every=1)。目標 = pool 勝率~0.5 / held-out 0→
    # 滑らか増加 / 20 iter ≤30分。
    "reinforce_case8_phase1_r1_p1": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_phase1_r1_p1",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/phase1_r1_p1.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    "reinforce_case8_phase1_r1_p2": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_phase1_r1_p2",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/phase1_r1_p2.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    "reinforce_case8_phase1_r1_p4": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_phase1_r1_p4",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/phase1_r1_p4.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 Phase 1 R5: iterations 20->50 (base=p4, priority_p=4.0)。
    # R1 で priority_p 単独では held-out が伸びなかったため学習量を拡大し、
    # 固定相手勝率が 0 から滑らかに増加するかを検証。A/B 軸は iterations のみ。
    "reinforce_case8_phase1_r5_iter50": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_phase1_r5_iter50",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": (
            "--config pipeline/reinforce/case8/configs/phase1_r5_iter50.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 Phase 1 観測性検証: 10 iter (~8min)。本番 (iter50) 前に
    # REQ1-5 (incremental local logs/metrics, S3 per-iter durability, resource
    # telemetry, oneshot SSH) が GPU 上で成立するか確認する事前検証 run。
    "reinforce_case8_phase1_validate": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_phase1_validate",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": (
            "--config pipeline/reinforce/case8/configs/phase1_validate.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 Phase 1 jit速度検証: iterations=30。rollout を eqx.filter_jit
    # 化した後の GPU util 上昇 / rollout_secs 短縮を実証する。学習結果は jit 前と
    # 不変 (同計算を GPU で一気通貫実行するだけ)。
    "reinforce_case8_phase1_jit_iter30": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_phase1_jit_iter30",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": (
            "--config pipeline/reinforce/case8/configs/phase1_jit_iter30.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 Phase 1 FROZEN: PPO baseline の確定実験条件 (algo=ppo,
    # 50 iter, f_var p=4.0, pool=full+lite+self, held-out=baseline_jax_full)。
    # Phase 2/3 (V-MPO) はこの config を algo (+V-MPO HP) 以外不変で流用する。
    "reinforce_case8_ppo_frozen": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_ppo_frozen",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/ppo_frozen.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 Phase 2: V-MPO arm。vmpo_frozen.yaml は ppo_frozen.yaml と
    # algo (ppo→vmpo) 以外完全同一。PPO arm と同一 PFSP/held-out/rollout harness で
    # 無調整 (論文デフォルト HP) A/B。安定性・収束性を比較する。
    "reinforce_case8_vmpo_frozen": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_vmpo_frozen",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/vmpo_frozen.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 Phase 3: V-MPO eps_alpha sweep。vmpo_frozen (eps_alpha=0.01)
    # と eps_alpha のみ差分。Phase 2 で trust-region KL 0.0003 ≪ ε_α=0.01 (α 過拘束)
    # だったので ε_α を緩めて学習加速するか検証。条件は凍結、V-MPO HP のみ変更の A/B。
    "reinforce_case8_phase3_eps_alpha_005": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_phase3_eps_alpha_005",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": (
            "--config pipeline/reinforce/case8/configs/phase3_eps_alpha_005.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    "reinforce_case8_phase3_eps_alpha_01": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_phase3_eps_alpha_01",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": (
            "--config pipeline/reinforce/case8/configs/phase3_eps_alpha_01.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 strict-opponent V-MPO: held-out + pool に strict JAX rulebase
    # (jax_v1/v2/v3 = case1/2/3 baseline_jax.strict、本物 parity port を in-JAX 化した
    # vmappable agent) を使う。rollout_jax の lax.switch に mode 7/8/9 を追加し、
    # python_v* host-callback を使わず GPU rollout 内で strict 相手と対戦する。
    # held-out=strict_v1 固定、pool=strict_v1/v2/v3 + self、f_var p=4.0、30 iter。
    "reinforce_case8_vmpo_strict": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_vmpo_strict",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/vmpo_strict.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 蒸留クローン版: 実 baseline_v1 の BC 蒸留 (case9_rulebase)
    # を bc_warmstart 教師 + 固定 NN 相手 (distilled) + held-out yardstick に使う。
    # PoC (Kaggle棋譜BC, flat 0% vs rulebase) の対策。
    "reinforce_case8_vmpo_distilled": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_vmpo_distilled",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": (
            "--config pipeline/reinforce/case8/configs/vmpo_distilled.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 ladder-pool iter11: 経験量拡大 episodes 96→192 (ladder9構成)。
    "reinforce_case8_vmpo_ladder11": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_vmpo_ladder11",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/vmpo_ladder11.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 ladder-pool iter10: フロンティア低 T0 シフト (終端断絶を埋める)。
    "reinforce_case8_vmpo_ladder10": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_vmpo_ladder10",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/vmpo_ladder10.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 ladder-pool iter9: T0崖の密刻み細分化 (ε段全廃、汚染遮断)。
    "reinforce_case8_vmpo_ladder9": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_vmpo_ladder9",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/vmpo_ladder9.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 ladder-pool iter8: 2軸 (T0, ε) ハイブリッド + 素strict定期照射。
    "reinforce_case8_vmpo_ladder8": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_vmpo_ladder8",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/vmpo_ladder8.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 ladder-pool iter7: フロンティア再構成 [300..100,0] 9段。
    "reinforce_case8_vmpo_ladder7": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_vmpo_ladder7",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/vmpo_ladder7.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 ladder-pool iter6: switch_iter 0 (noopウォームアップ廃止)。
    "reinforce_case8_vmpo_ladder6": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_vmpo_ladder6",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/vmpo_ladder6.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 ladder-pool iter5: episodes 96 (バッチ次元で経験量+50%)。
    "reinforce_case8_vmpo_ladder5": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_vmpo_ladder5",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/vmpo_ladder5.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 ladder-pool iter4: iterations 70 (サイクル当たり経験量1.4x)。
    "reinforce_case8_vmpo_ladder4": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_vmpo_ladder4",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/vmpo_ladder4.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 ladder-pool iter3: mix_strict 0.6 (フロンティア対戦密度up)。
    "reinforce_case8_vmpo_ladder3": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_vmpo_ladder3",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/vmpo_ladder3.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 ladder-pool iter2: フロンティア (300-200) 周辺を細分化した
    # 9段 ladder + iter1 best.pt からの resume。
    "reinforce_case8_vmpo_ladder2": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_vmpo_ladder2",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/vmpo_ladder2.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 ladder-pool: 時間窓弱体化 strict の7段を独立 pool エントリ化
    # (AlphaStar checkpoint-PFSP 流)。50/35/15 混合 + f_var 段選択 + v2 継続学習。
    "reinforce_case8_vmpo_ladder": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_vmpo_ladder",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/vmpo_ladder.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 ハンディキャップ・カリキュラム (蒸留不要): 本物 strict_v1
    # と強制対戦 (every:5)、seat0 初期 ships を ladder 3.0→1.0 で勝率連動降下。
    # held-out は h=1.0 の strict_v1 + baseline_jax_full 両観測。
    "reinforce_case8_vmpo_handicap": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_vmpo_handicap",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": (
            "--config pipeline/reinforce/case8/configs/vmpo_handicap.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 合流構成: 蒸留クローン (教師+常設NN相手) + strict_v1
    # (10iter毎の強制 pool 対戦 + held-out yardstick)。実証済み3部品の合流。
    "reinforce_case8_vmpo_combo": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_vmpo_combo",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/vmpo_combo.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case8 strict-held-out 軽量版: vmpo_strict は pool+held-out 両方に
    # strict を入れ iter0 が >13分で打ち切り (strict 実行が重い)。対策として
    # pool は in-JAX baseline (full+lite+self) に戻し、strict_v1 を held-out
    # 進捗 yardstick のみ (every:5, episodes:16) で実行する。
    # reinforce/case8 PoC: BC warm-start × V-MPO。vmpo_frozen から bc_warmstart
    # のみ有効化 (case9_per_planet BC 重みで初期化)。rulebase への勝率軌跡が
    # 学習で向上するかを held-out 曲線 + 学習後の実 baseline_v1 offline 対戦で検証。
    "reinforce_case8_vmpo_poc_bc": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_vmpo_poc_bc",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": ("--config pipeline/reinforce/case8/configs/vmpo_poc_bc.yaml"),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    "reinforce_case8_vmpo_strict_heldout": {
        "family": "reinforce",
        "stage": "train_reinforce_case8_vmpo_strict_heldout",
        "train_module": "pipeline.reinforce.case8.training.train_jax",
        "config_arg": (
            "--config pipeline/reinforce/case8/configs/vmpo_strict_heldout.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case4: case3 の 2-head 分解 (per_planet (P+1) categorical で
    # target+NO_OP を一括 P(Y) + ship regression P(X|Y)) を、3-head 分解
    # (launch head の 2値 P(Z) / target head の P-class P(Y|Z) / ship head
    # P(X|Y,Z)) に置き換えた case。学習レシピ・backbone は case3 と完全同一で、
    # head 構造のみを変えて性能・収束速度の改善を検証する純粋 ablation。
    "reinforce_case4_kaggle_jax_train": {
        "family": "reinforce",
        "stage": "train_reinforce_case4_kaggle_jax_train",
        "train_module": "pipeline.reinforce.case4.training.train_jax",
        "config_arg": "--config pipeline/reinforce/case4/configs/kaggle_jax_train.yaml",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce_case4_kaggle_jax_smoke: short 6-iter JAX smoke for case4 used to
    # verify the 3-head wiring end-to-end before the full 200-iter run.
    "reinforce_case4_kaggle_jax_smoke": {
        "family": "reinforce",
        "stage": "train_reinforce_case4_kaggle_jax_smoke",
        "train_module": "pipeline.reinforce.case4.training.train_jax",
        "config_arg": "--config pipeline/reinforce/case4/configs/kaggle_jax_smoke.yaml",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case5: case3 を丸ごと継承し support (補助) reward を検証する case。
    # H1 = ship 差分と planet 差分を個別係数で同時加算する combined shaping。
    # docs/experiment/reinforce/20260527_case5_support_reward/ 参照。
    "reinforce_case5_kaggle_jax_train": {
        "family": "reinforce",
        "stage": "train_reinforce_case5_kaggle_jax_train",
        "train_module": "pipeline.reinforce.case5.training.train_jax",
        "config_arg": (
            "--config "
            "pipeline/reinforce/case5/configs/kaggle_jax_train_h1_combined.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case5 H2: 保持割合 mine/(mine+enemy) の Δ を shaping (ratio mode)。
    "reinforce_case5_kaggle_jax_train_h2_ratio": {
        "family": "reinforce",
        "stage": "train_reinforce_case5_kaggle_jax_train_h2_ratio",
        "train_module": "pipeline.reinforce.case5.training.train_jax",
        "config_arg": (
            "--config pipeline/reinforce/case5/configs/kaggle_jax_train_h2_ratio.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case5 H4: ratio mode の shaping_coef sweep (0.50 → 1.0)。
    "reinforce_case5_kaggle_jax_train_h4_ratio_coef1": {
        "family": "reinforce",
        "stage": "train_reinforce_case5_kaggle_jax_train_h4_ratio_coef1",
        "train_module": "pipeline.reinforce.case5.training.train_jax",
        "config_arg": (
            "--config "
            "pipeline/reinforce/case5/configs/kaggle_jax_train_h4_ratio_coef1.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case5 H5: planet 信号を production 加重保持割合に (ratio_prod mode)。
    "reinforce_case5_kaggle_jax_train_h5_ratio_prod": {
        "family": "reinforce",
        "stage": "train_reinforce_case5_kaggle_jax_train_h5_ratio_prod",
        "train_module": "pipeline.reinforce.case5.training.train_jax",
        "config_arg": (
            "--config "
            "pipeline/reinforce/case5/configs/kaggle_jax_train_h5_ratio_prod.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case5 H7: ratio per-turn shaping を band clip (序盤 spike 抑制)。
    "reinforce_case5_kaggle_jax_train_h7_ratio_clip": {
        "family": "reinforce",
        "stage": "train_reinforce_case5_kaggle_jax_train_h7_ratio_clip",
        "train_module": "pipeline.reinforce.case5.training.train_jax",
        "config_arg": (
            "--config "
            "pipeline/reinforce/case5/configs/kaggle_jax_train_h7_ratio_clip.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case5 H3: 絶対保持数の非差分 dense 加算 (非 PBRS 対照群)。
    "reinforce_case5_kaggle_jax_train_h3_dense": {
        "family": "reinforce",
        "stage": "train_reinforce_case5_kaggle_jax_train_h3_dense",
        "train_module": "pipeline.reinforce.case5.training.train_jax",
        "config_arg": (
            "--config pipeline/reinforce/case5/configs/kaggle_jax_train_h3_dense.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce/case5 H6: 勝ちターン短縮 time bonus + 引き伸ばし penalty。
    "reinforce_case5_kaggle_jax_train_h6_time_bonus": {
        "family": "reinforce",
        "stage": "train_reinforce_case5_kaggle_jax_train_h6_time_bonus",
        "train_module": "pipeline.reinforce.case5.training.train_jax",
        "config_arg": (
            "--config "
            "pipeline/reinforce/case5/configs/kaggle_jax_train_h6_time_bonus.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # reinforce_case1_bench_workers: short 5-iter run to measure rollout
    # parallelization speedup vs iter1 serial baseline (7h / 100 iter).
    # Same hyperparams, only iterations + rollout_workers differ.
    "reinforce_case1_bench_workers": {
        "family": "reinforce",
        "stage": "train_reinforce_case1_bench_workers",
        "train_module": "pipeline.reinforce.case1.training.train",
        "config_arg": "--config pipeline/reinforce/case1/configs/bench_workers.yaml",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # bench_jax_env_gpu: GPU benchmark for the JAX env. Not a training case.
    # Installs jax[cuda12] inside the pod, runs vendor vs jax(cpu/gpu) timings,
    # writes bench_results.json. Uses family=reinforce so the onstart pipeline
    # builds the Rust simulator and pulls BC weights (~5 min overhead, harmless
    # — we just need the JAX env imports to resolve, plus a default-shape run dir).
    "bench_jax_env_gpu": {
        "family": "reinforce",
        "stage": "bench_jax_env_gpu",
        "train_module": "pipeline.reinforce._bench.jax_env_gpu.run_bench",
        "config_arg": "",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # bench_featurizer_gpu: GPU benchmark for the JAX featurizer
    # (W1+W2a-d, f, final). Measures wall-clock for PyTorch baseline vs
    # JAX (cpu wheel) vs JAX (GPU, post jax[cuda12] install) at
    # vmap=1/16/32/64. Same family/onstart shape as bench_jax_env_gpu so
    # the artifacts uploader and `dev/runpod pull` work unchanged.
    "bench_featurizer_gpu": {
        "family": "reinforce",
        "stage": "bench_featurizer_gpu",
        "train_module": "pipeline.reinforce._bench.featurizer_gpu.run_bench",
        "config_arg": "",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # bench_rollout_gpu: GPU bench for the W4 JAX rollout (featurize +
    # model forward + env step end-to-end). Same shape as
    # bench_featurizer_gpu so artifacts upload + dev/runpod pull work.
    "bench_rollout_gpu": {
        "family": "reinforce",
        "stage": "bench_rollout_gpu",
        "train_module": "pipeline.reinforce._bench.rollout_gpu.run_bench",
        "config_arg": "",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # bench_baseline_jax_gpu: GPU bench for the rulebase/case2 baseline_jax
    # numeric port (geometry/physics). Times the Python scalar-loop original
    # vs JAX (cpu wheel) vs JAX (GPU after jax[cuda12]) across a vmap batch
    # sweep. Same family/onstart shape as the other bench_*_gpu cases so the
    # artifacts uploader and `dev/runpod pull` work unchanged.
    "bench_baseline_jax_gpu": {
        "family": "reinforce",
        "stage": "bench_baseline_jax_gpu",
        "train_module": "pipeline.reinforce._bench.baseline_jax_gpu.run_bench",
        "config_arg": "",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # bench_agent_full_jax_gpu: GPU bench for the faithful rulebase/case1 JAX
    # agent (core_jax.agent_full_jax). Times vmapped JAX-vs-JAX self-play
    # throughput (env-steps/sec) across batch sizes vs Python baseline_v1 single.
    # Headline = GPU figure justifying the rulebase→JAX port.
    "bench_agent_full_jax_gpu": {
        "family": "reinforce",
        "stage": "bench_agent_full_jax_gpu",
        "train_module": "pipeline.reinforce._bench.agent_full_jax_gpu.run_bench",
        "config_arg": "",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # bench_aim_jax_gpu: GPU bench for the case2 intercept solver — both the
    # (src,target) grid vmap (vs Python double loop) AND the act()-level turn
    # time (Python vs JAX aim backend, on jax_env obs). Same onstart shape as
    # the other bench_*_gpu cases.
    "bench_aim_jax_gpu": {
        "family": "reinforce",
        "stage": "bench_aim_jax_gpu",
        "train_module": "pipeline.reinforce._bench.aim_jax_gpu.run_bench",
        "config_arg": "",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # tournament_rulebase_jax_smoke: ~10-min observability pre-validation for the
    # round-robin JAX tournament. 3 agents, few seeds, short horizon — confirms
    # incremental tournament.json/leaderboard.md flush + S3 upload + nvidia-smi /
    # system_monitor sampling + SSH all work BEFORE the full run (the mandatory
    # dry-run from pipeline.md "Dry-run before the real run"). Same family/onstart
    # shape as bench_*_gpu so the artifacts uploader + `dev/runpod pull` work.
    "tournament_rulebase_jax_smoke": {
        "family": "reinforce",
        "stage": "tournament_rulebase_jax_smoke",
        "train_module": "pipeline.rulebase._bench.tournament.run_tournament",
        "config_arg": (
            "--config pipeline/rulebase/_bench/tournament/configs/smoke.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # tournament_rulebase_jax: full round-robin over the 7 in-JAX rule agents
    # (jax_v1/2/3/4/6/8/9), 300 seeds/half × both seat assignments, horizon 500.
    # Win-rate + Wilson CI leaderboard. Run ONLY after the smoke case validates
    # observability. Same onstart shape as bench_*_gpu.
    "tournament_rulebase_jax": {
        "family": "reinforce",
        "stage": "tournament_rulebase_jax",
        "train_module": "pipeline.rulebase._bench.tournament.run_tournament",
        "config_arg": "--config pipeline/rulebase/_bench/tournament/configs/full.yaml",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # ranking_rulebase_jax_smoke: ~10-min pre-validation for the RANKING driver
    # (ranking.py). 3 agents, few seeds, short horizon — confirms incremental
    # ranking.json/ranking.md flush + S3 + samplers + SSH before the full run.
    "ranking_rulebase_jax_smoke": {
        "family": "reinforce",
        "stage": "ranking_rulebase_jax_smoke",
        "train_module": "pipeline.rulebase._bench.tournament.ranking",
        "config_arg": (
            "--config pipeline/rulebase/_bench/tournament/configs/ranking_smoke.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
    # ranking_rulebase_jax: minimal-cost total-order ranking via adjacent-pair
    # verification of the case-number prior [v8,v6,v4,v3,v2,v1]. 5 adjacent
    # comparisons (+repair on inversion), 100*2=200 games each, horizon 300.
    # Tie-merge for near-equal agents. Run ONLY after the smoke case validates.
    "ranking_rulebase_jax": {
        "family": "reinforce",
        "stage": "ranking_rulebase_jax",
        "train_module": "pipeline.rulebase._bench.tournament.ranking",
        "config_arg": (
            "--config pipeline/rulebase/_bench/tournament/configs/ranking.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
}


def case_subdir(case: str) -> str:
    """Map a registry key to the on-disk case subdirectory.

    case9 has head variants registered as separate keys (case9_three_head /
    case9_candidate / case9_candidate_ships / case9_template_ships /
    case9_dual) but they all share the same
    `data/output/models/imitation/case9/` tree on disk (same training
    pipeline, different head_mode). Strip the `_<variant>` suffix.

    Reinforce cases use a `reinforce_<caseN>` prefix that maps to
    `data/output/models/reinforce/<caseN>/`.
    """
    if case.startswith("case9_"):
        return "case9"
    if case.startswith("case10_"):
        return "case10"
    if case.startswith("case11_"):
        return "case11"
    if case.startswith("reinforce_"):
        return case[len("reinforce_") :]
    if case.startswith("bench_"):
        return case[len("bench_") :]
    return case


def case_family(case: str) -> str:
    """Return the top-level family directory for a registry key."""
    defaults = CASE_DEFAULTS.get(case, {})
    family = defaults.get("family", "imitation")
    return family


def runs_root_for(case: str) -> Path:
    return Path(f"data/output/models/{case_family(case)}/{case_subdir(case)}/runs")


def case_defaults(case: str) -> dict[str, str]:
    if case not in CASE_DEFAULTS:
        raise typer.BadParameter(
            f"unknown case={case!r}; supported: {sorted(CASE_DEFAULTS)}"
        )
    return CASE_DEFAULTS[case]


__all__ = [
    "CASE_DEFAULTS",
    "case_defaults",
    "case_family",
    "case_subdir",
    "runs_root_for",
]

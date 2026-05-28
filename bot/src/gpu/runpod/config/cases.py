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

# Reinforce/case6 — PFSP pool (iter2) ANALYSIS

> 対象: iter2 (H2, snapshot pool + 周期更新) / run_id: 20260528-005806__...__36982a3__seed0
> 関連: iter2_plan.md / iter2_result.md / iter1_analysis.md / hypotheses.md
> モード: skip mode (replay JSON なし — JAX rollout は in-memory)

metrics.json (100 iter) ベース。H1 との対比で PFSP pool の効果を読む。

## what_worked (機能した点)

1. **win_rate 飽和の解消** — H1 は frozen 相手で last10=0.988 に張り付き。H2 は
   last10=0.661 / overall=0.614。pool 周期更新 (K=10) + baseline_jax_full 混合で
   相手が常に手強く、勝ち切れない状態を維持できた。
2. **baseline_jax_full が学習圧の主エンジン** — opponent 内訳 noop:5 / self_snapshot:55 /
   full:40。vs self_snapshot=0.828 (過去自分は surpass 済) に対し **vs full=0.274**。
   full が「勝てない相手」として持続的な勾配を供給。
3. **vs full の上昇トレンド** — early5 0.138 → last5 0.359、slope +0.0027/iter。
   100 iter で強いルール相手への勝率が 2.6 倍に。「より強い agent になった」直接証拠。
4. **entropy が有界** — H1 は 46→97 と暴走 (policy 拡散) したが H2 は 38→47。
   勝てない相手の存在が policy を絞り続け、健全な学習。
5. **コスト 1/10** — iterations 100 / episodes 64 + RTX 4090 で $0.70 (H1 $7.1)。

## where_to_focus_next (H4 への示唆)

- **vs full の伸びしろが本丸**: 0.359 で頭打ち気味。H2 は uniform mix (pool/full 50:50、
  pool 内も一様) なので、難敵 (full・強い pool snapshot) への露出が薄い。
  → **H4: PFSP `f_hard(x)=(1−x)^p`** で勝率の低い相手を優先 sampling すれば、full への
    学習が加速し vs full 到達点が上がるはず。注視メトリクス = vs full の last5 mean と slope。
- **pool 内優先度も検討**: 現状 pool.sample は一様。H4 では pool snapshot 各々の勝率を
  追跡し、手強い (= 最近の) snapshot を優先するのが筋。
- **H5 (f_var=x(1-x)) との A/B**: full が強すぎて勾配消失する場合は同レベル優先が有利な
  可能性。H4 で vs full が伸び悩んだら H5 を試す。

## why_not_yet_conclusive (n<300)

- win_rate は相手構成依存の相対値。vs full 0.359 は「baseline_jax_full に 36% 勝つ」だが、
  これが「Kaggle で通用する強さ」かは別問題。H4/H5 完了後に rl_v6 vs baseline_v1 /
  baseline_jax_full / rl_v3 を 300 戦 (例外条件) で測って初めて絶対的強度を確定できる。
- entropy 47 はまだやや高め。さらなる収束余地あり (H4 の優先 sampling が効けば下がる想定)。

## NEXT ACTION

1. H4 (PFSP f_hard) を実装 → vs full の到達点が H2 (0.359) を超えるか検証。
2. H4 でも iterations 100 / episodes 64 / RTX 4090 優先のコスト方針を継続。
3. H4 or H5 完了後、最良 iter の weights で 300 戦 (vs baseline_v1 / full / rl_v3) を実施。

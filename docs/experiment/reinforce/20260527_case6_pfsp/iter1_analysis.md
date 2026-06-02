# Reinforce/case6 — PFSP self-snapshot (iter1) ANALYSIS

> 対象: iter1 (H1, self_snapshot opponent) / run_id: 20260527-145442__...__fb36504__seed0
> 関連: iter1_plan.md / iter1_result.md / hypotheses.md
> モード: skip mode (replay JSON なし — JAX rollout は in-memory、selfplay runner 不経由)

分析は metrics.json (200 iter) + iter1_result.md のみ。対戦 replay は存在しないため
turn-level の tactical 分析は不可。学習ダイナミクスから 3 観点を抽出する。

## why_inconclusive (信号が枯れた理由)

win_rate trajectory (opponent 切替を跨いで):

| iter | win | opponent | 解釈 |
|---|---|---|---|
| 0-4 | 0.98-1.00 | noop | 何もしない相手、自明に勝つ |
| 5 | 0.844 | self_snapshot | **switch 直後、frozen iter0 が一時的に手強い** |
| 6 | **0.766** | self_snapshot | 最小値。snapshot が実質的な学習圧を供給 |
| 7-15 | 0.84→0.91 | self_snapshot | 急速に snapshot を上回る |
| 50 | 0.945 | self_snapshot | ほぼ勝ち切り |
| 150-199 | ~1.000 | self_snapshot | 完全飽和 |

- **frozen iter0 snapshot は iter6 で win 0.766 まで押し下げた** = 短期的には本物の相手。
  しかし相手が固定なので agent が surpass した時点 (~iter15-50) で信号が枯れ、以降は
  「勝てる相手に勝ち続ける」だけの無圧状態に。entropy が 41→97 と単調増加 (policy が
  絞られず拡散) なのは、勝ちが保証され探索を抑える勾配が消えたことの裏付け。
- value_loss は 0.104→0.052 と半減 = 価値推定そのものは学習継続。配線は健全。

## what_worked (機能した点)

1. **配線が完全に成立**: OPPONENT_SELF_SNAPSHOT=3 + lax.switch 4 分岐 + opp_model の
   vmap broadcast が 200 iter / 25600 episodes を NaN なく完走。unit test 5 + smoke + 本番。
2. **switch 設計が機能**: noop early (勝ち体験) → self_snapshot late の curriculum が
   意図通り iter5 で発火し、相手が強くなる遷移を再現できた (iter6 dip が証拠)。
3. **frozen snapshot が一時的に学習信号を出した** — iter5-15 の dip&recovery は、
   pool 化で相手を更新し続ければ持続的な学習圧になりうる手応え。

## where_to_focus_next (H2 への示唆)

- **相手を学習に追従させる** = H2 の核心。iter6 の dip (0.766) は「相手が agent と
  同程度に強い」瞬間に学習圧が最大化することを示す。K iter ごとに最新 snapshot を
  pool 追加 + late をそこからサンプリングすれば、win_rate が中間域 (0.5-0.7) に
  留まり reward trend が意味を持つはず。
- **注視メトリクス**: H2 では win_rate が 1.0 に張り付かず 0.4-0.7 帯を維持するか、
  entropy が単調増加せず収束に転じるか。これらが PFSP 成立の判定軸。
- **コスト制約を先に潰す**: self_snapshot 系は rollout ~2倍重。H2 は iterations を
  100 に半減 or episodes 64 で開始し、A100 fallback 時は uptime 手動監視
  ([[project_reinforce_self_snapshot_cost]])。

## 確認に必要なこと (n<300 のため断定不可)

- 本 iter の win_rate は「相手 = frozen iter0」前提の値で、強さの絶対指標ではない。
  H1 単体では「より強い agent になったか」は判定不能 (verdict=inconclusive)。
- H2 完了後に rl_v6 vs baseline_v1 / rl_v3 を 300 戦 (例外条件) で測って初めて
  「PFSP がより強い agent を生むか」に答えられる。

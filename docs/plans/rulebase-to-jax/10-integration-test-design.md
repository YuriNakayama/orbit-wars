# rulebase-to-jax — 結合テスト設計 (ローカル CPU)

## 現状認識 (TDD の観点)

- 確認済 = **単体相当の構造検証** (PoC0/1/2)。しかも本物 Python helper を呼んでおり、JAX 実装の単体テストですらない (「JAX 化可能な構造である」ことの実証)。
- **未着手 = 結合テスト**: JAX port を end-to-end で動かし、本物 Python と**等価**であることをローカル CPU で示す。
- TDD 原則: 結合テストを**先に**書き (RED)、それを GREEN にするよう Step1-4 を実装する。

## 「結合」の 2 レベル (tests.md の分類に従う)

tests.md は integration (workflow が動くか、black-box) と e2e (self-play / 実 episode 実行) を区別。本件の等価性検証は両方にまたがるので 2 段に分ける:

### レベルA: action 等価性 (e2e/pipeline、最重要)
**同一 obs で JAX port の action == 本物 Python の action** を、**実 self-play で到達する盤面列**に対して検証。

- 既存 `case2/test_agent_jax_identity.py` が precedent だが**2つの欠陥**があり、本件では修正する:
  1. ⚠️ `step(empty_actions())` で両 seat noop 進行 → **realistic な contested 盤面に到達しない**。
     → **修正**: 実際に JAX agent (seat0) と本物 Python (seat1) の action で `step` し、**本物がプレイした盤面列**を辿る。
  2. ⚠️ tolerance が緩い (`mismatches <= 1`, `angle < 1e-2`)。memory `project_rulebase_jax_parity_failure_mode` の温床。
     → **修正**: **action 完全一致 (一致率 100%)** を assert。許容は float 位置の rtol のみ、選択 (from_pid/ships int) は exact。tie-break 統一。

検証手順 (1 ゲーム/seed):
```
state = reset(seed)
for turn in range(500):
    a_jax = compute_actions_jax(state, seat=0)       # JAX port
    a_py  = agent(state_to_obs(state, player=0))     # 本物 Python、同一 obs
    assert actions_equal(a_jax, a_py)                # ★ 完全一致 (tie-break 統一後)
    # 本物がプレイした盤面を辿る (両 seat 本物 Python で進行)
    state = step(state, both_python_actions)
    if done: break
```
- seat0 の比較に専念し、盤面進行は本物同士で行う (到達分布を本物に合わせる)。
- 配置: `bot/tests/e2e/pipeline/rulebase/case1/test_agent_jax_identity.py`

### レベルB: self-play が破綻なく回るか (e2e/pipeline、smoke)
JAX port 同士 / JAX vs Python で 1 ゲーム完走し、不正 action (NaN・範囲外・shape 不一致) を出さないこと。等価性ではなく**実行健全性**の black-box smoke。
- `make_orbit_wars_env` ではなく JAX env (`reset`/`step`) で回す (JAX port は EnvState 入力のため)。
- 配置: 同上ファイル内の別テスト。

## 等価性の契約 (actions_equal の定義)

JAX port 出力 `(MAX_LAUNCHES, 3)` float32 tensor と Python `list[[pid, angle, ships]]` を比較:

```python
def actions_equal(jax_row, py_moves, *, angle_tol=1e-4):
    jax_moves = [(int(r[0]), float(r[1]), int(r[2])) for r in jax_row if r[0] >= 0]
    j = sorted(jax_moves);  p = sorted((int(m[0]), float(m[1]), int(m[2])) for m in py_moves)
    if len(j) != len(p): return False
    return all(a[0]==b[0] and abs(a[1]-b[1])<angle_tol and a[2]==b[2] for a,b in zip(j,p))
```
- `from_planet_id`: exact。`ships`: exact (int)。`angle`: rtol 1e-4 (aim solver の float32 差のみ許容)。
- 発射本数も exact (launch/hold の判定一致)。

## TDD 進行 (RED → GREEN)

1. **いま RED にする**: 上記 e2e テストを書く。`compute_actions_jax` は現状 lite なので **0% 一致で FAIL** する (= 正しい RED、実測済 0% を裏付け)。
2. Step1-4 を実装するたびにこのテストの一致率が上がる (turn0 → 中盤 → 全 turn)。
3. **GREEN = 全 seed・全 turn で 100% 一致**。これが「JAX と Python が等価」の最終定義。
4. CI では重いので `@pytest.mark.slow`、seed 数を絞る (例 5 seed × 500 turn)。

## ローカル CPU 実行

```bash
cd bot
# RED 確認 (実装前、FAIL するはず)
uv run pytest tests/e2e/pipeline/rulebase/case1/test_agent_jax_identity.py -x -q
# 全テスト
dev/test-bot
```
JAX は CPU 既定 (`JAX_PLATFORM_NAME` 不要、PoC で実証済)。x64 は parity 単体テスト側で使い、e2e は本番同様 float32 で一致率を見る (float32 で 100% にするのが最終目標、tie-break 統一で吸収)。

## まとめ

| レベル | 内容 | 合格条件 | 配置 |
|--------|------|----------|------|
| A: action 等価 | 同一 obs で JAX==Python、本物プレイ盤面列 | **全 turn 100% 一致** (tie-break 統一) | e2e/.../test_agent_jax_identity.py |
| B: self-play smoke | JAX 同士で 1 ゲーム完走 | NaN/範囲外/shape 異常なし | 同上 |

既存 case2 test の「緩い tolerance + noop 進行」は**反面教師**。本件は完全一致 + 本物プレイ盤面で厳格化する。

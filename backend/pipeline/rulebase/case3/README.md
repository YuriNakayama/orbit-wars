# case3 — baseline_v3

case2 の後継。`baseline/lookahead/rollout.py` (325 行) を追加した内蔵ロールアウト。

## 採用戦略

- case2 の構成 + rollout.py
- mission scoring に短期ロールアウトの結果を反映

## 構造

case2 とほぼ同型。違いは:
- `baseline/lookahead/rollout.py` 追加
- `baseline/strategy.py` 内で rollout を mission scoring に反映

## 備考

`baseline/core/` は case1 / case2 と高い類似性を持つが、case 完全独立原則に従い copy 維持。

"""Replay viewer for Orbit Wars matches.

VS Code / Jupyter で "Run Cell" ボタンから各 `# %%` セルを実行する。
`env.render(mode="ipython")` で盤面を可視化する。
"""

# %%
from pathlib import Path

from env import list_matches, load_replay

DATA_ROOT = Path("data")

# %%
df = list_matches(DATA_ROOT, mode="1v1", limit=10)
df

# %%
match_id = df["match_id"][0]
env = load_replay(match_id, data_root=DATA_ROOT)
env.render(mode="ipython", width=800, height=600)

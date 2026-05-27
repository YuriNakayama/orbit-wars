"""Interactive mode: S3-mediated command channel for live Kaggle Kernel control.

Kaggle Notebook には SSH / live tail がないため、長時間 sleep する notebook を
push し、その中で S3 prefix を **command queue** として polling させる方式で
RunPod の ``dev/ssh/sync/destroy`` と機能的に等価な体験を実現する。

Flow:
    Claude (local) ──PUT cmd──▶ S3 inbox/  ◀─poll─ Kaggle Kernel
                                S3 outbox/ ──poll──▶ Claude (local)

詳細: ``docs/plans/kaggle-kernel-basis/07_interactive_mode.md`` (作成予定)
"""

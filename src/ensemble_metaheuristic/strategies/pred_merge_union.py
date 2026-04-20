"""Union of two flat prediction dicts (label-set OR)."""
from __future__ import annotations

from typing import Dict, List


def merge_preds_union(
    a: Dict[int, List[str]],
    b: Dict[int, List[str]],
    all_pids: List[int],
) -> Dict[int, List[str]]:
    return {pid: sorted(set(a.get(pid, [])) | set(b.get(pid, []))) for pid in all_pids}

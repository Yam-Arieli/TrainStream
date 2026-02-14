__all__ = ["select_random", "select_stratified"]

import random
import heapq
from typing import Any, List, Dict, Callable
from collections import defaultdict

def select_random(data, m, train_info=None, **kwargs):
    """Reservoir Sampling: Randomly keeps m items."""
    if len(data) <= m: return data
    return random.sample(data, m)

def select_stratified(data: List[Any], m: int, train_info: Any = None, **kwargs) -> List[Any]:
    """
    Stratified Sampling Strategy.
    Groups data by 'stratify_by', calculates a quota for each group, 
    and then uses 'base_select_fn' to pick items within that group.

    Args:
        data: List of data items.
        m: Total number of items to select.
        train_info: Passed to the base selector.
        **kwargs: Must contain:
            - 'stratify_by': Key/Attribute to group by.
            - 'base_select_fn' (Optional): Function to select items within a group. 
                                           Defaults to select_random.

    Returns:
        The stratified subset of size m.
    """
    strat_key = kwargs.get('stratify_by')
    base_selector = kwargs.get('base_select_fn', select_random)

    if not strat_key:
        raise ValueError("select_stratified requires 'stratify_by' in **kwargs")

    # 1. Group Data
    groups = defaultdict(list)
    for item in data:
        if isinstance(item, dict):
            label = item.get(strat_key)
        else:
            label = getattr(item, strat_key, None)
        groups[label].append(item)

    # 2. Calculate Quotas & Select
    total_n = len(data)
    if total_n <= m: return data

    selected_data = []
    
    # Sort keys for deterministic behavior
    sorted_labels = sorted(groups.keys(), key=lambda x: str(x))
    
    for label in sorted_labels:
        group_items = groups[label]
        group_size = len(group_items)
        
        if group_size == 0: continue

        # Proportional Allocation: (Group / Total) * m
        # We ensure at least 1 item per group if quota > 0
        raw_quota = (group_size / total_n) * m
        quota = max(1, int(raw_quota))
        
        # --- THE MAGIC ---
        # We call the USER'S function on just this group!
        # We pass the same kwargs down, so train_info/etc are preserved.
        selection_from_group = base_selector(
            data=group_items, 
            m=min(len(group_items), quota), # Don't ask for more than exists
            train_info=train_info,
            **kwargs
        )
        
        selected_data.extend(selection_from_group)

    # 3. Final Trim (to ensure exact m size)
    # Because of rounding up (max(1, ...)), we might overshoot m.
    if len(selected_data) > m:
        # If we have too many, we just trim the end 
        # (or shuffle first if we want fairness, but simple trim is faster)
        random.shuffle(selected_data) 
        return selected_data[:m]

    return selected_data
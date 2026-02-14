import random
import heapq
from typing import Any, List, Dict

def select_random(data, m, train_info=None, **kwargs):
    """Reservoir Sampling: Randomly keeps m items."""
    if len(data) <= m: return data
    return random.sample(data, m)

def select_fifo(data, m, train_info=None, **kwargs):
    """First-In-First-Out: Keeps the most recent m items."""
    return data[-m:]

def select_highest_loss(data, m, train_info, **kwargs):
    """
    Active Learning: Keeps items with highest loss.
    Requires train_info['losses'] to be a list of floats matching data order.
    """
    losses = train_info.get('losses')
    if not losses or len(losses) != len(data):
        raise ValueError("select_highest_loss requires train_info['losses'] of same length as data.")
    
    # Sort by loss (descending) and take top m
    # (Using zip/sort is safer than kwargs assumptions)
    paired = sorted(zip(data, losses), key=lambda x: x[1], reverse=True)
    return [item[0] for item in paired[:m]]
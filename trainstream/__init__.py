"""
TrainStream — Streaming Coreset Training for Large-Scale Datasets.

Train neural networks on datasets too large to fit in memory by streaming
chunks, training with a coreset buffer, and selecting the most informative
samples using training dynamics (confidence, AUM, forgetting events).

Quick start:
    from trainstream import StreamLoop
    from trainstream.streaming import from_numpy
    from trainstream.selection import select_top

    engine = StreamLoop(
        stream_factory=from_numpy(X, y, chunk_size=2000),
        buffer_size=500,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
    )
    result = engine.run()
"""

__version__ = "0.2.0"

from .core import StreamLoop
from .training import (
    default_train_fn,
    train_one_epoch,
    train_one_batch,
    auto_detect_device,
)
from .selection import (
    select_random,
    select_top,
    select_by_forgetting,
    weighted_sample_from_confidence,
    select_stratified,
    compute_class_budget,
)
from .streaming import generator_from_args, from_numpy
from .preprocessing import make_scanpy_preprocessor, compute_global_gene_list
from .tracking import CoresetTracker
from .evaluation import make_eval_fn

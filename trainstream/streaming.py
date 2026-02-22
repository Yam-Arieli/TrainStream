"""
Stream factory helpers for TrainStream.

A stream factory is a callable that returns an iterator of (X, y) tuples,
where X is np.ndarray (n_samples, n_features) and y is np.ndarray (n_samples,).

The user defines how raw data is loaded (files, API, database) and optionally
provides a preprocess_fn to convert raw data into (X, y) numpy arrays.
"""

import numpy as np
from typing import Callable, List, Any, Optional, Tuple, Iterator


def generator_from_args(
    loader_func: Callable[[Any], Any],
    args_list: List[Any],
    preprocess_fn: Optional[Callable[[Any], Tuple[np.ndarray, np.ndarray]]] = None,
) -> Callable[[], Iterator[Tuple[np.ndarray, np.ndarray]]]:
    """
    Creates a stream factory by mapping a loader function over a list of arguments.

    Args:
        loader_func: Callable that takes one argument and returns raw data.
                     Examples: sc.read_h5ad, pd.read_csv, custom API fetcher.
        args_list: List of arguments to map over (file paths, URLs, etc.).
                   Must be a list (not a generator) so the factory can be
                   called multiple times.
        preprocess_fn: Optional callable(raw_data) -> (X, y).
                       Converts the raw output of loader_func into numpy arrays.
                       If None, loader_func must directly return (X, y) tuples.

    Returns:
        A callable that returns a fresh iterator of (X, y) tuples each time
        it is called.

    Example:
        # With scanpy preprocessing
        factory = generator_from_args(
            sc.read_h5ad,
            ["chunk_0.h5ad", "chunk_1.h5ad", "chunk_2.h5ad"],
            preprocess_fn=my_preprocessor,
        )
        stream = factory()  # fresh iterator
        X, y = next(stream)
    """
    def factory():
        for arg in args_list:
            raw = loader_func(arg)
            if preprocess_fn is not None:
                yield preprocess_fn(raw)
            else:
                yield raw

    return factory


def from_numpy(
    X: np.ndarray,
    y: np.ndarray,
    chunk_size: int,
) -> Callable[[], Iterator[Tuple[np.ndarray, np.ndarray]]]:
    """
    Factory that streams chunks from in-memory numpy arrays.
    Useful for testing or small datasets.

    Args:
        X: np.ndarray (n_samples, n_features) — full feature matrix.
        y: np.ndarray (n_samples,) — full label array.
        chunk_size: Number of samples per chunk.

    Returns:
        A callable that returns a fresh iterator of (X_chunk, y_chunk) tuples.

    Example:
        factory = from_numpy(X_train, y_train, chunk_size=500)
        engine = StreamLoop(stream_factory=factory, ...)
    """
    n = len(X)

    def factory():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            yield X[start:end], y[start:end]

    return factory

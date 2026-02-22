# TrainStream 🚂🌊

Train neural networks on datasets too large to fit in memory — practically and with strong performance.

## The Problem

You have a massive dataset (e.g., 10M cells × 20K genes). You can't load it all at once, and you can't train on it in full. Standard approaches either discard most of the data or require expensive distributed infrastructure.

## The Idea

TrainStream implements a **Streaming Coreset** loop:

```
For each chunk from the stream:
    1. Merge chunk (k samples) with a small coreset buffer (m samples)
    2. Train for a few epochs on the merged data
    3. Use training dynamics to select the best m samples as the new coreset
    4. Repeat — next chunk loads in parallel with current training
```

**Memory requirement:** `2k + m + model_size` — independent of total dataset size.

**Why coresets help performance:** The model sees all the data (once), but the most informative samples are retained in the buffer and keep influencing training across chunks. The selection is driven by *training dynamics* — metrics like confidence, margin, and forgetting events that reveal which samples are most valuable.

## Installation

```bash
git clone https://github.com/Yam-Arieli/TrainStream.git
cd TrainStream
pip install .                # core
pip install ".[scanpy]"      # + scanpy/AnnData support for single-cell RNA-seq
```

**Dependencies:** `numpy`, `torch`, `scikit-learn`, `tqdm`
**Optional:** `scanpy`, `scipy` (for single-cell preprocessing)

## Quick Start

```python
import torch
import torch.nn as nn
from trainstream import StreamLoop
from trainstream.streaming import from_numpy

# Stream from in-memory arrays (use generator_from_args for files/APIs)
factory = from_numpy(X_train, y_train, chunk_size=2000)

engine = StreamLoop(
    stream_factory=factory,
    buffer_size=500,           # coreset size (m)
    model=my_model,
    optimizer=torch.optim.Adam(my_model.parameters(), lr=1e-3),
    criterion=nn.CrossEntropyLoss(),
    epochs_per_step=5,
    score_key="aum",           # use AUM (margin) to drive selection
)

result = engine.run()

# Result contains:
# result["X"]            — final coreset features
# result["y"]            — final coreset labels
# result["tracker"]      — provenance of every coreset sample
# result["train_history"]— per-step loss summaries
# result["eval_history"] — per-step metrics (if eval_fn provided)
```

## Selection Methods

TrainStream computes per-sample training dynamics during each training phase and uses them to select the coreset. All methods are stratified by class by default.

| Method | `score_key` | `sample_fn` | Keeps |
|---|---|---|---|
| AUM (default) | `"aum"` | `select_top` | Highest margin samples — well-learned class backbone |
| Confidence | `"confidence"` | `select_top` | Highest mean P(correct) across epochs |
| Weighted confidence | `"confidence"` | `weighted_sample_from_confidence` | Logistic-weighted quantile sampling biased toward high confidence |
| Forgetting | `"forgetting"` | `select_by_forgetting` | Samples near decision boundaries that flip most |
| Random | any | `select_random` | Baseline — ignores scores |

```python
from trainstream import weighted_sample_from_confidence, select_by_forgetting

# Logistic-weighted confidence sampling
engine = StreamLoop(..., score_key="confidence", sample_fn=weighted_sample_from_confidence)

# Forgetting-based selection
engine = StreamLoop(..., score_key="forgetting", sample_fn=select_by_forgetting)
```

## Staleness Prevention

Without intervention, early chunks can dominate the coreset because their samples accumulate high scores over time. Two independent mechanisms address this:

**1. Chunk fraction cap** — hard limit on any single chunk's share:
```python
engine = StreamLoop(..., max_chunk_fraction=0.3)
# No chunk can occupy more than 30% of each class's budget
```

**2. Confidence decay** — erodes old samples' scores gradually:
```python
engine = StreamLoop(..., confidence_decay=0.9)
# Old buffer scores are multiplied by 0.9 each step
```

Use either, both, or neither.

## Evaluation

Track model performance on a held-out test set after each streaming step:

```python
from trainstream import make_eval_fn

eval_fn = make_eval_fn(
    X_test, y_test,
    label_map=CLASS_TO_INT,   # for per-class F1 reporting
)

engine = StreamLoop(..., eval_fn=eval_fn)
result = engine.run()

# result["eval_history"] is a list of dicts:
# [{"step": 0, "accuracy": 0.72, "f1_macro": 0.70, "f1_per_class": {...}}, ...]
```

## Coreset Tracking

Every sample in the coreset carries its provenance — which chunk it came from and its index within that chunk. This enables post-hoc analysis.

```python
tracker = result["tracker"]

# Which chunks are represented in the final coreset?
print(tracker.chunk_distribution())
# Counter({3: 24, 2: 13, 1: 12, 0: 11})

# Export full history as a DataFrame
df = tracker.to_dataframe()
# columns: step, chunk_id, within_chunk_idx, score
```

## Single-Cell RNA-seq Example (Scanpy)

TrainStream was built for large-scale scRNA-seq classification. The default scanpy preprocessor handles gene alignment, normalization, and densification.

```python
import scanpy as sc
import torch.nn as nn
from trainstream import StreamLoop, make_eval_fn
from trainstream.streaming import generator_from_args
from trainstream.preprocessing import make_scanpy_preprocessor, compute_global_gene_list
from trainstream.selection import weighted_sample_from_confidence

# 1. Compute a fixed gene list from a reference (run once)
ref = sc.concat([sc.read_h5ad(f) for f in chunk_files[:5]])
global_genes = compute_global_gene_list(ref, n_top_genes=2000)

# 2. Build label map from all classes
CLASS_TO_INT = {name: i for i, name in enumerate(all_classes)}

# 3. Create preprocessor: AnnData → (X, y) numpy arrays
preprocessor = make_scanpy_preprocessor(
    global_gene_list=global_genes,
    label_column="cell_type",
    label_map=CLASS_TO_INT,
)

# 4. Stream factory: loads and preprocesses each chunk on demand
factory = generator_from_args(sc.read_h5ad, chunk_files, preprocess_fn=preprocessor)

# 5. Run
engine = StreamLoop(
    stream_factory=factory,
    buffer_size=4096,
    model=MyClassifier(input_dim=2000, num_classes=len(CLASS_TO_INT)),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    criterion=nn.CrossEntropyLoss(),
    epochs_per_step=10,
    batch_size=256,
    score_key="confidence",
    sample_fn=weighted_sample_from_confidence,
    max_chunk_fraction=0.3,
    eval_fn=make_eval_fn(X_test, y_test, label_map=CLASS_TO_INT),
)
result = engine.run()
```

The preprocessing pipeline per chunk: filter genes → normalize (1e4) → log1p → align to global gene list (zero-pad missing genes) → densify → scale.

## Custom Training Function

You can replace the default training loop with your own. Your function must return a dict containing at least the `score_key` you intend to use for selection.

```python
def my_train_fn(X, y, model, optimizer, criterion, epochs, batch_size, device, binary):
    # ... your training logic ...
    return {
        "confidence": ...,  # np.ndarray (n_samples,)
        "aum":        ...,  # np.ndarray (n_samples,)
        "forgetting": ...,  # np.ndarray (n_samples,) int
        "avg_loss":   ...,  # float
    }

engine = StreamLoop(..., train_fn=my_train_fn)
```

The `train_one_epoch` and `train_one_batch` functions are also exported for building custom loops:

```python
from trainstream import train_one_epoch, train_one_batch
```

## Architecture

```
trainstream/
├── core.py           # StreamLoop — the engine
├── training.py       # PyTorch training: train_one_batch, train_one_epoch, default_train_fn
├── selection.py      # Selection methods + stratified wrapper
├── streaming.py      # Stream factory helpers
├── preprocessing.py  # Default scanpy preprocessing
├── tracking.py       # CoresetTracker — sample provenance
└── evaluation.py     # Evaluation hooks
```

## License

MIT

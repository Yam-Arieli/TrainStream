# TrainStream 🚂🌊

**TrainStream** is a lightweight library for training Machine Learning models on massive, streaming datasets without forgetting.

It implements a **Streaming Coreset** approach:
1. Load a chunk of data ($k$) from a stream.
2. Merge it with a memory buffer ($m$).
3. Train a model on the merged data.
4. Select the best $m$ samples to keep for the next step.

## Installation

```bash
git clone [https://github.com/yourusername/TrainStream.git](https://github.com/yourusername/TrainStream.git)
cd TrainStream
pip install .
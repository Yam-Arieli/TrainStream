"""
Default preprocessing for TrainStream.

Provides a ready-to-use preprocessor for scanpy AnnData objects (single-cell RNA-seq),
and a utility to compute a global gene list from a reference dataset.

Users working with non-scanpy data should write their own preprocess_fn that
returns (X: np.ndarray, y: np.ndarray) and pass it to generator_from_args.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable

try:
    import scanpy as sc
    from scipy.sparse import issparse, csr_matrix
    _HAS_SCANPY = True
except ImportError:
    _HAS_SCANPY = False


def _require_scanpy():
    if not _HAS_SCANPY:
        raise ImportError(
            "scanpy is required for the default preprocessor. "
            "Install it with: pip install scanpy"
        )


def compute_global_gene_list(reference_adata, n_top_genes: int = 2000, min_cells: int = 3) -> List[str]:
    """
    Compute a fixed list of highly variable genes (HVGs) from a reference dataset.
    Run this once before the streaming loop to lock in the gene set.

    Args:
        reference_adata: AnnData object (can be a concatenation of several chunks).
        n_top_genes: Number of top HVGs to select.
        min_cells: Minimum number of cells a gene must appear in.

    Returns:
        List of gene names to use across all chunks.

    Example:
        ref = sc.concat([sc.read_h5ad(f) for f in chunk_files[:5]])
        global_genes = compute_global_gene_list(ref, n_top_genes=2500)
    """
    _require_scanpy()

    adata = reference_adata.copy()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

    return adata.var_names[adata.var["highly_variable"]].tolist()


def make_scanpy_preprocessor(
    global_gene_list: List[str],
    label_column: str,
    label_map: Dict[str, int],
    target_sum: float = 1e4,
    max_scale: float = 10,
    min_cells: int = 3,
) -> Callable:
    """
    Returns a preprocess_fn for scanpy AnnData objects.

    The returned function converts an AnnData chunk into (X, y) numpy arrays,
    applying: gene filtering → normalize → log1p → align to global gene list
    → densify → scale → encode labels.

    Args:
        global_gene_list: Fixed gene names (from compute_global_gene_list).
        label_column: Column in adata.obs containing string class labels.
        label_map: Dict mapping label strings to integers, e.g. {"B_cell": 0, "T_cell": 1}.
        target_sum: Normalization target (default 1e4).
        max_scale: Max value after scaling (default 10).
        min_cells: Minimum cells for gene filtering (default 3).

    Returns:
        Callable[[AnnData], Tuple[np.ndarray, np.ndarray]]

    Example:
        preprocessor = make_scanpy_preprocessor(
            global_gene_list=global_genes,
            label_column="cytokine",
            label_map={"IL-2": 0, "IL-6": 1, ...},
        )
        factory = generator_from_args(sc.read_h5ad, file_list, preprocess_fn=preprocessor)
    """
    _require_scanpy()

    target_genes = set(global_gene_list)

    def preprocess(adata) -> Tuple[np.ndarray, np.ndarray]:
        adata = adata.copy()

        # Save labels before any operations that might drop metadata
        labels_backup = adata.obs[label_column].copy()

        # Standard preprocessing
        sc.pp.filter_genes(adata, min_cells=min_cells)
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)

        # Align genes to global gene list
        batch_genes = set(adata.var_names)
        missing_genes = list(target_genes - batch_genes)

        if len(missing_genes) > 0:
            zero_data = csr_matrix(
                (adata.n_obs, len(missing_genes)), dtype=adata.X.dtype
            )
            adata_missing = sc.AnnData(X=zero_data)
            adata_missing.var_names = missing_genes
            adata_missing.obs_names = adata.obs_names
            adata = sc.concat(
                [adata, adata_missing], axis=1, join="outer", index_unique=None
            )

        # Reorder and slice to match global gene list exactly
        adata = adata[:, global_gene_list].copy()

        # Restore labels
        adata.obs[label_column] = labels_backup.values

        # Densify sparse matrix
        if issparse(adata.X):
            adata.X = adata.X.toarray()

        # Scale
        sc.pp.scale(adata, max_value=max_scale)

        # Extract numpy arrays
        X = adata.X.astype(np.float32)
        y = np.array([label_map[label] for label in adata.obs[label_column].values])

        return X, y

    return preprocess

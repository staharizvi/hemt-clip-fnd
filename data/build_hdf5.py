"""Pack staged Fakeddit images + texts + labels into a single HDF5 file.

Why HDF5: loading 15K loose JPEGs from Google Drive is dominated by small-file I/O
overhead. A single HDF5 file with sequential reads is roughly 10× faster.

Schema:
    images : uint8  [N, 3, 224, 224]
    texts  : variable-length UTF-8 strings [N]
    labels : int8   [N]              # 0=real, 1=fake
    alpha  : float32 [N]             # precomputed CLIP text-image cosine similarity
    ids    : variable-length strings [N]   # Fakeddit post id, useful for joins

Outputs: data/fakeddit.h5, plus train/val/test CSV splits in data/splits/.
"""

# TODO: implement HDF5 packing + stratified split (see Blueprint §5)

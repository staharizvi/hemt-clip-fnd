"""PyTorch Dataset reading from the packed HDF5 file.

Returns dicts with:
    input_ids, attention_mask  (from RoBERTa tokenizer, max_len=128)
    pixel_values               (CLIP-normalised image tensor)
    alpha                      (precomputed CLIP cosine similarity, scalar)
    label                      (0=real, 1=fake)

Note: open the HDF5 handle lazily inside __getitem__ (after worker fork),
otherwise PyTorch DataLoader workers crash on h5py.File handles.
"""

# TODO: implement HEMTClipDataset(torch.utils.data.Dataset)

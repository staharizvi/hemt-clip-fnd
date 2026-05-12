"""Parallel Fakeddit image downloader.

Reads Fakeddit metadata TSVs, samples ~18K rows (stratified by 2_way_label),
downloads images concurrently with 5s timeout + 1 retry, resizes to 224×224 RGB,
and writes them to a local staging directory. Broken URLs are skipped and logged.

Inputs:  Fakeddit metadata TSVs
Outputs: data/images/<id>.jpg (staged for build_hdf5.py), data/raw/sample.csv
"""

# TODO: implement download pipeline (see Blueprint §5.4)

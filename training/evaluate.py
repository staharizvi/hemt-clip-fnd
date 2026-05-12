"""Test-set evaluation + metrics.

Computes accuracy, precision/recall/F1 (per-class and macro), AUC-ROC,
and confusion matrix. Saves a metrics JSON and renders the report plots:
    - confusion-matrix heatmap
    - ROC curve
    - per-class P/R bar chart
into outputs/.
"""

# TODO: implement evaluate(checkpoint_path, split="test")

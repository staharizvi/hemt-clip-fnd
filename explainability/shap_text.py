"""Post-hoc explainability — SHAP for text tokens.

Wraps a HuggingFace text-classification pipeline around the text branch of
the trained model and runs shap.Explainer on a fixed subset of test samples
(mix of correct + misclassified).

Outputs colour-coded HTML via shap.plots.text(), plus selected PNG snapshots
for the report. Multimodal SHAP is intentionally skipped — too complex,
marginal value for the FYP scope.
"""

# TODO: implement SHAP wrapper + sample-export helpers

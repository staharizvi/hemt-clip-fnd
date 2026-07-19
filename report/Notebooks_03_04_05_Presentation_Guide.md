# Notebooks 03, 04, and 05: Presentation Guide

This guide explains what happens in the three notebooks, why each step exists, and how to present it confidently.

## The Overall Story

Use this simple narrative:

1. **Notebook 03 trains and compares the models.**
2. **Notebook 04 evaluates the trained models on unseen test data.**
3. **Notebook 05 explains how the best model made its decisions.**

Together, they answer three questions:

- Can we train a multimodal fake-news detector?
- Does it perform better than simpler alternatives?
- Can we understand and inspect its decisions?

---

# Notebook 03: Full Training

## One-Sentence Explanation

Notebook 03 trains multiple versions of the model under the same conditions to determine which architecture works best.

## What Happens in the Notebook

### 1. Colab and Data Setup

The notebook:

- Mounts Google Drive.
- Clones or updates the GitHub repository.
- Installs the required Python packages.
- Copies `fakeddit.h5` from Drive to Colab's local storage.
- Changes the dataset path in `configs/base.yaml`.

The HDF5 file is copied locally because reading it repeatedly from Google Drive would make training slower.

### 2. TensorBoard Is Started

TensorBoard displays:

- Training and validation loss.
- Training and validation accuracy.
- Validation F1 score.
- Learning rate.
- Weight and gradient distributions.

Say:

> TensorBoard lets us monitor learning and identify problems such as overfitting, unstable gradients, or a model that has stopped improving.

### 3. The Model Receives Five Inputs

Each dataset sample contains:

| Input | Meaning |
|---|---|
| `input_ids` | Tokenized title for RoBERTa |
| `attention_mask` | Identifies real tokens and padding |
| `pixel_values` | CLIP-normalized image |
| `alpha` | Precomputed CLIP text-image cosine similarity |
| `label` | `0 = real`, `1 = fake` |

### 4. The Main Model Architecture

```text
Title -> RoBERTa -> 512-dimensional text vector

Image -> CLIP ViT-B/16 -> 196 image-patch vectors

Text vector queries image patches through cross-attention

Alpha controls the image contribution in gated_fusion

Fused vector -> MLP classifier -> Real/Fake logits
```

Important dimensions:

- RoBERTa initially produces a 768-dimensional representation.
- CLIP ViT-B/16 produces a 14 x 14 grid, giving 196 patch tokens.
- Text and image features are projected to 512 dimensions.
- Cross-attention uses 8 attention heads.
- The classifier maps `512 -> 256 -> 2`.

### 5. What Cross-Attention Does

The text representation is the **query**, while image patches are the **keys and values**:

```text
Query = text vector
Keys/Values = image patch vectors
```

This allows the title to search the image for relevant visual evidence.

Say:

> Instead of combining two independent summaries, cross-attention lets the text ask which image regions are relevant to its meaning.

### 6. What Alpha Means

`alpha` is the cosine similarity between CLIP's text and image embeddings. It measures how semantically aligned the title and image are.

In the headline `gated_fusion` model:

```text
fused = alpha * attended + (1 - alpha) * text
```

- High alpha gives more influence to image-attended evidence.
- Low alpha makes the model rely more on text.
- The gate adds no trainable parameters.

Important clarification:

> Alpha is not directly treated as a fake-news score. It controls the mixture of text and image evidence, while the classifier learns how that mixture relates to the label.

### 7. Two-Stage Fine-Tuning

The model is trained in two stages.

#### Stage 1: Head Warm-Up

- Duration: 1 epoch.
- Learning rate: `1e-4`.
- RoBERTa and CLIP backbones are frozen.
- Only projections, fusion, and classifier are trained.

Why:

> The newly initialized layers must first learn reasonable behavior before their noisy gradients are allowed to modify the pretrained encoders.

#### Stage 2: Partial Encoder Fine-Tuning

- Maximum duration: 6 epochs.
- Learning rate: `2e-5`.
- Last 4 RoBERTa layers and last 4 CLIP blocks are unfrozen.
- Early stopping monitors validation F1 with patience 3.

Why:

> A smaller learning rate gently adapts pretrained knowledge to fake-news detection without destroying it.

### 8. Training Stability and Efficiency

The training script uses:

- **AdamW:** optimizer with weight decay.
- **Learning-rate warm-up and linear decay:** avoids unstable early updates.
- **Mixed precision:** reduces GPU memory usage and speeds up training.
- **Gradient accumulation:** simulates a larger batch size.
- **Gradient clipping:** prevents excessively large updates.
- **Gradient checkpointing:** saves GPU memory by recomputing some activations.
- **Atomic checkpoints:** protects checkpoints from interrupted writes.
- **Full resume support:** restores the model, optimizer, scheduler, scaler, epoch, and random state.

### 9. Ablation Study

An ablation study changes or removes components to measure their contribution.

| Variant | What It Tests |
|---|---|
| `text_only` | Performance using only the title |
| `image_only` | Performance using only the image |
| `concat_fusion` | Simple concatenation of text, pooled image, and alpha |
| `hemt_clip` | Cross-attention with alpha appended as a feature |
| `gated_fusion` | Cross-attention with alpha controlling the fusion |

The key controlled comparison is:

```text
hemt_clip vs gated_fusion
```

They use the same cross-attention backbone. The main difference is whether alpha is appended as a feature or used as a gate.

### 10. Validation Results

| Variant | Best Validation F1 |
|---|---:|
| `text_only` | 0.7702 |
| `image_only` | 0.8012 |
| `concat_fusion` | 0.8204 |
| `hemt_clip` | 0.8229 |
| `gated_fusion` | **0.8464** |

Main conclusion:

> Multimodal models outperform unimodal models, and using alpha as a gate gives the strongest validation result.

### 11. Seed Robustness

The `hemt_clip` alpha-feature variant was trained using seeds 42, 7, and 123:

```text
Validation F1 = 0.8218 +/- 0.0017
```

This shows that its result is stable across different random initializations and data-shuffle orders.

Be honest:

> The final gated model currently has a single-seed result, so multi-seed evaluation of that model remains future work.

## How to Present Notebook 03

> In notebook 03, we train five controlled variants using the same dataset splits and training schedule. We use two-stage fine-tuning to protect pretrained RoBERTa and CLIP representations. The ablation shows that combining text and image is better than using either alone, and the alpha-gated cross-attention model achieves the best validation F1 of 0.8464.

---

# Notebook 04: Evaluation

## One-Sentence Explanation

Notebook 04 loads the best checkpoint of every model and evaluates each one on 2,573 test posts that were never used for training or model selection.

## Why a Separate Test Set Matters

- The training set teaches model parameters.
- The validation set selects checkpoints and guides development.
- The test set estimates final generalization.

Say:

> We do not select the best model using test performance. The test set is held back so it can provide a fair final evaluation.

## What Happens in the Notebook

### 1. Discover the Best Checkpoints

The evaluation script automatically finds the latest canonical `best.pt` checkpoint for every variant.

It avoids seed-specific checkpoints when selecting the canonical model, preventing an accidental mismatch during comparison.

### 2. Run Test Inference

For every model:

1. Build the correct architecture.
2. Load its saved weights.
3. Run inference on all test samples.
4. Convert logits into probabilities using softmax.
5. Select the class with the highest logit.
6. Compute evaluation metrics and save predictions.

### 3. Understand the Outputs

- **Logits:** raw model scores before softmax.
- **Probabilities:** normalized confidence for real and fake.
- **Predictions:** class with the highest score.
- **Labels:** ground-truth classes.

Predictions are saved in `.npz` files because notebook 05 reuses them for explainability sample selection.

## Metrics You Must Be Able to Explain

### Accuracy

The fraction of all predictions that are correct.

```text
accuracy = correct predictions / all predictions
```

Accuracy is meaningful here because the dataset is nearly balanced.

### Precision

Of all posts predicted as fake, how many were actually fake?

```text
precision = TP / (TP + FP)
```

High precision means fewer real posts are falsely accused of being fake.

### Recall

Of all actual fake posts, how many did the model detect?

```text
recall = TP / (TP + FN)
```

High recall means fewer fake posts are missed.

### F1 Score

The harmonic mean of precision and recall:

```text
F1 = 2 * precision * recall / (precision + recall)
```

F1 is the primary metric because it rewards a useful balance between detecting fake posts and avoiding false alarms.

### AUC-ROC

AUC measures how well the model ranks fake posts above real posts across every possible decision threshold.

- `0.5` is random ranking.
- `1.0` is perfect ranking.

Say:

> AUC is useful because it is threshold-independent. It tells us whether the model separates the classes well even if the deployment threshold later changes.

### Confusion Matrix

```text
                 Predicted Real    Predicted Fake
Actual Real      True Negative     False Positive
Actual Fake      False Negative    True Positive
```

It reveals the types of mistakes hidden by a single score.

## Final Test Results

| Variant | Accuracy | F1 | Precision | Recall | AUC |
|---|---:|---:|---:|---:|---:|
| `text_only` | 0.7773 | 0.7827 | 0.7522 | 0.8158 | 0.8545 |
| `image_only` | 0.8061 | 0.8135 | 0.7716 | 0.8601 | 0.8824 |
| `concat_fusion` | 0.8286 | 0.8319 | **0.8034** | 0.8625 | 0.9042 |
| `hemt_clip` | 0.8189 | 0.8277 | 0.7776 | 0.8846 | 0.8919 |
| `gated_fusion` | **0.8313** | **0.8393** | 0.7895 | **0.8957** | **0.9122** |

## What the Results Mean

### Finding 1: Multimodal Learning Helps

`concat_fusion` beats both `text_only` and `image_only`.

This shows that text and images provide complementary information.

### Finding 2: Images Are Stronger Than Titles on This Dataset

`image_only` beats `text_only`.

Likely reason:

> Fakeddit titles are short, while images often contain strong visual clues about the post category.

### Finding 3: The Alpha Gate Is the Best Overall Design

`gated_fusion` achieves the best:

- Accuracy.
- F1.
- Recall.
- AUC.

It beats the alpha-feature `hemt_clip` model by:

- About **1.2 F1 points**.
- About **2.0 AUC points**.

It beats plain concatenation by:

- About **0.7 F1 points**.
- About **0.8 AUC points**.

### Finding 4: Different Models Have Different Error Profiles

`concat_fusion` has the highest precision, while `gated_fusion` has higher recall.

Say:

> The gated model catches more fake posts, but this comes with slightly more false alarms than the concat model.

### Finding 5: Validation Transfers to Test

The best validation model remains the best test model, and validation-to-test drops are small.

This suggests healthy generalization rather than severe validation overfitting.

## Figures Produced

- Per-model confusion matrices.
- ROC curve overlay.
- F1 comparison bar chart.
- Per-class precision and recall chart.
- CSV, Markdown, JSON, and NPZ result files.

## How to Present Notebook 04

> Notebook 04 performs the final fair comparison on 2,573 unseen test samples. The alpha-gated model remains the strongest model, reaching F1 0.8393 and AUC 0.9122. The results also show that multimodal learning consistently beats either modality alone.

---

# Notebook 05: Explainability

## One-Sentence Explanation

Notebook 05 investigates why the best model makes its predictions using image attention, text attribution, and modality-removal experiments.

## Why Explainability Matters

A fake-news detector can affect real people and content. A prediction without a reason is difficult to trust, audit, or improve.

Notebook 05 uses four explanation methods:

| Method | Question Answered |
|---|---|
| Cross-attention heatmaps | Which image regions did the title attend to? |
| SHAP | Which text tokens contributed to the verdict? |
| LIME | Which whole words locally influenced the verdict? |
| Modality contribution | Did text or image matter more for this sample? |

## Shared Sample Selection

The methods select examples from the test predictions generated in notebook 04.

Samples include:

- Correct and incorrect predictions.
- Real and fake labels.
- Low-, medium-, and high-confidence cases.

This avoids showing only easy, successful examples.

## Method 1: Cross-Attention Heatmaps

### How They Are Generated

The cross-attention layer produces weights with shape:

```text
batch x 8 heads x 1 text query x 196 image patches
```

The script:

1. Averages attention over the 8 heads.
2. Reshapes 196 patch scores into a 14 x 14 grid.
3. Upsamples the grid to 224 x 224.
4. Overlays it on the original image.

The heatmap shows which image regions received high attention in relation to the text query.

### Sample Buckets

The notebook selects 12 examples:

- `correct_hi`: correct and highly confident.
- `correct_lo`: correct but uncertain.
- `wrong_hi`: wrong and highly confident.
- `wrong_lo`: wrong and uncertain.

`wrong_hi` examples are especially valuable because they reveal confident failure modes.

### Important Limitation

> Attention shows where the model focused, but attention alone does not prove causal importance.

## Method 2: SHAP Text Attribution

SHAP repeatedly masks or changes text tokens and measures how the prediction changes.

It estimates each token's marginal contribution relative to a baseline.

For the multimodal model:

- The real image is held fixed.
- Alpha is held fixed.
- Only the text is perturbed.

Therefore, SHAP explains which text tokens affected the actual multimodal verdict, not a separate text-only proxy.

RoBERTa uses subword tokenization, so SHAP may display pieces of words.

## Method 3: LIME Text Attribution

LIME creates many nearby versions of the title by removing words, observes the model's predictions, and fits a simple local model.

Its word weights explain the behavior around one specific example.

Difference from SHAP:

- SHAP estimates marginal feature contributions.
- LIME approximates local sensitivity using a simple surrogate model.

They can disagree because they answer related but different questions.

Say:

> Agreement between SHAP and LIME increases confidence in an explanation, while disagreement is itself useful because it identifies unstable or context-dependent behavior.

## Efficient Multimodal Text Explanations

SHAP and LIME require many model calls.

To reduce cost:

1. Encode the sample's image once.
2. Cache its image-patch features.
3. Reuse the cached patches for every perturbed title.
4. Re-run only the text encoder, fusion, and classifier.

This preserves the multimodal explanation while avoiding repeated CLIP image encoding.

## Method 4: Modality Contribution

This method compares three predictions:

```text
full     = real text + real image + real alpha
no_image = real text + zero image + alpha set to zero
no_text  = empty text + real image + real alpha
```

For predicted class `c`:

```text
image contribution = full logit[c] - no_image logit[c]
text contribution  = full logit[c] - no_text logit[c]
```

A large drop after removing a modality means that modality strongly supported the original prediction.

The method also checks whether removing text or image flips the final predicted class.

### Important Limitation

Zero images and empty titles are not normal dataset examples. Therefore:

> These values are directional evidence of reliance, not exact causal or Shapley attributions.

## What Notebook 05 Establishes

- The cross-attention model can provide intrinsic image-side explanations.
- SHAP and LIME provide complementary text-side explanations.
- The model's decisions can be inspected for successes, uncertainty, confident errors, and bias.
- Modality occlusion verifies whether individual decisions genuinely depend on both text and image.

## How to Present Notebook 05

> Notebook 05 moves beyond performance and investigates model behavior. Cross-attention heatmaps show image regions linked to the title, SHAP and LIME show influential words, and modality-removal experiments measure whether text or image carried each verdict. We also explicitly acknowledge that attention and perturbation explanations are useful diagnostic evidence, not perfect causal proof.

---

# The Full Presentation Script

Use this as a compact spoken explanation:

> Notebook 03 is the training and model-selection stage. We train five variants under the same conditions using two-stage fine-tuning. First, the pretrained encoders are frozen while the new projection, fusion, and classifier layers warm up. Then, the last four layers of RoBERTa and CLIP are gently fine-tuned. The ablation study shows that multimodal models outperform text-only and image-only models, and using CLIP similarity alpha as a gate produces the best validation F1.
>
> Notebook 04 is the final evaluation stage. We load each best checkpoint and test it on 2,573 samples that were never used during training or validation. The gated model remains the strongest, reaching F1 0.8393 and AUC 0.9122. It also has the highest recall, meaning it catches the largest proportion of fake posts, while concat fusion has slightly better precision.
>
> Notebook 05 is the explainability stage. We use cross-attention heatmaps to inspect image focus, SHAP and LIME to inspect word influence, and modality occlusion to measure whether text or image contributed more to each decision. This makes the system easier to audit and helps identify both successful reasoning and failure modes.

---

# Likely Questions and Strong Answers

## Why use two-stage training?

The new fusion and classifier layers begin randomly initialized. Training them first while freezing the encoders prevents unstable gradients from damaging pretrained RoBERTa and CLIP knowledge.

## Why fine-tune only the last four encoder layers?

Lower layers contain general pretrained features, while upper layers are more task-specific. Partial fine-tuning reduces compute and overfitting while still adapting the encoders.

## Why is F1 the main metric?

F1 balances precision and recall for the fake class. It penalizes a model that catches many fakes by falsely accusing too many real posts, or one that avoids false alarms by missing many fakes.

## Why report AUC as well?

AUC measures ranking quality across all thresholds. It confirms that the model's advantage is not only caused by the default decision threshold.

## What is the project's strongest result?

The controlled ablation shows that the alpha-gated cross-attention model is the strongest tested design on both validation and held-out test data.

## Does high alpha mean a post is real?

No. Alpha only measures text-image semantic similarity. In this dataset, fake posts can even have higher alignment because satire or misleading posts often use very relevant images. Alpha controls fusion; the classifier learns the relationship with the label.

## Is the gated model statistically proven to be better?

It wins across F1, accuracy, recall, and AUC, but the gated model currently has a single-seed result. The margin should be presented as promising rather than statistically conclusive.

## Is attention a complete explanation?

No. Attention shows where the model focused, but it does not guarantee causal importance. That is why the project also uses SHAP, LIME, and modality occlusion.

## Why use both SHAP and LIME?

They explain predictions differently. Agreement provides stronger evidence, while disagreement reveals unstable or context-dependent reasoning.

## Why evaluate incorrect predictions?

Correct examples show how the system succeeds, but incorrect and highly confident examples reveal bias, shortcuts, and failure modes that are more useful for improvement.

## What is the main limitation of modality occlusion?

An empty title or zero image is off-distribution. The measured changes indicate relative reliance but should not be treated as exact causal contributions.

---

# Numbers to Memorize

```text
Dataset:
17,149 total posts
12,003 train
2,573 validation
2,573 test

Architecture:
RoBERTa-base
CLIP ViT-B/16
196 image patches = 14 x 14
512-dimensional projected features
8-head cross-attention
Classifier: 512 -> 256 -> 2

Training:
Stage 1: 1 epoch, learning rate 1e-4, encoders frozen
Stage 2: up to 6 epochs, learning rate 2e-5, last 4 layers/blocks trainable
Early-stopping patience: 3

Best model:
gated_fusion
Validation F1: 0.8464
Test accuracy: 0.8313
Test F1: 0.8393
Test recall: 0.8957
Test AUC: 0.9122
```

# Final Closing Line

> The project does not only produce a fake-news prediction. It experimentally shows that gated multimodal fusion improves performance, verifies that improvement on unseen data, and provides multiple ways to inspect why the model reached each decision.

# Findings and Discussion

For the caption **"a photo of a dog"**, dog images ranked **1 out of 10 classes** by average cosine similarity (dog mean = **0.260**), so dog was the highest-ranked class. The three highest-scoring non-dog classes were **cat** (0.240), **deer** (0.233), and **frog** (0.227). CLIP achieved **89.5% overall zero-shot accuracy** on **1,000 CIFAR-10 test images** (100 per class). The best-performing class was **horse** (98.0%), while the lowest-performing class was **deer** (76.0%). The most common directional confusion(s) were **deer predicted as horse (19 images); truck predicted as automobile (11 images)**; these errors may reflect visually overlapping categories. Because CIFAR-10 images are only **32×32 pixels**, small objects and distinguishing details may be hard to recognize even after resizing; CLIP results also depend on prompt wording.

## Experimental details
- **Model:** openai/clip-vit-base-patch32
- **Dataset:** CIFAR-10 test split
- **Caption:** a photo of a dog
- **Images per class:** 100
- **Total images:** 1000
- **Zero-shot accuracy:** 89.50%

## Interpretation
Cosine similarity measures alignment of normalized image and text embeddings, not a probability.
Class-wise similarity and classification accuracy measure different things.

## Figures and tables
See the displayed top/bottom caption matches, best image in each class,
similarity distribution, per-class accuracy chart, confusion matrix, and misclassifications.

# Aircraft Damage Classification and Automated Captioning

### Deep Learning for Aviation Maintenance Inspection

`Python 3.11` `TensorFlow 2.x` `Keras` `PyTorch` `Hugging Face Transformers` `VGG16` `BLIP` `Transfer Learning` `Computer Vision` `Vision-Language Models`

---

## Overview

An end-to-end computer-vision pipeline for **aircraft surface inspection**. The project pairs a fine-tuned VGG16 classifier for binary defect detection (`dent` vs. `crack`) with a custom Keras layer that wraps the **Salesforce BLIP** vision-language model to generate natural-language captions and summaries of inspected imagery. The two components are designed to operate together: the classifier triages incoming inspection images, and the captioning layer produces standardized descriptive text for downstream maintenance documentation.

---

## Business Value

Visual inspection of aircraft surfaces is a safety-critical, labor-intensive activity. Manual review is slow, costly, and inconsistent across inspectors and hangars. This pipeline targets four operational outcomes:

- **Reduced inspection cycle time.** Automated first-pass triage of damage imagery accelerates the path from photograph to maintenance decision.
- **Standardized defect labelling.** A single classifier eliminates inter-inspector variability across fleets and facilities.
- **Auto-generated inspection narratives.** BLIP-generated captions and summaries are ready for ingestion into maintenance records, work orders, and compliance reports.
- **Lower operational risk.** Earlier and more consistent defect surfacing reduces the probability of structural issues progressing undetected.

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.11 |
| Deep Learning Frameworks | TensorFlow 2.x, Keras, PyTorch |
| Pretrained Models | VGG16 (ImageNet), BLIP (`Salesforce/blip-image-captioning-base`) |
| Model Hub | Hugging Face Transformers |
| Image Processing | Pillow, NumPy, Matplotlib |
| Notebook Environment | Jupyter |

---

## Architecture

### End-to-End Workflow

```
                         +-----------------------------+
   Inspection Image  --> |  Preprocessing (224x224)    |
                         |  rescale, augment           |
                         +--------------+--------------+
                                        |
                       +----------------+----------------+
                       |                                 |
                       v                                 v
        +---------------------------+      +-----------------------------+
        |   VGG16 Classifier        |      |   BlipCaptionSummaryLayer    |
        |   (frozen base + head)    |      |   wraps BLIP via             |
        |                           |      |   tf.py_function             |
        |   Output: dent / crack    |      |   Output: caption / summary  |
        +---------------------------+      +-----------------------------+
                       |                                 |
                       +---------------+-----------------+
                                       |
                                       v
                         +-----------------------------+
                         |   Inspection Record         |
                         |   (label + narrative)       |
                         +-----------------------------+
```

### Classifier Architecture

```
Input (224 x 224 x 3)
   |
   v
VGG16 (ImageNet, frozen)
   |
   v
Flatten
   |
   v
Dense(512, ReLU) -> Dropout(0.3)
   |
   v
Dense(512, ReLU) -> Dropout(0.3)
   |
   v
Dense(1, Sigmoid)  ->  P(crack)
```

- **Loss:** Binary cross-entropy
- **Optimizer:** Adam, learning rate `1e-4`
- **Callbacks:** EarlyStopping (`patience=5`, restore best weights), ModelCheckpoint
- **Input:** RGB images rescaled to `[0, 1]` with light rotation augmentation on the training split only

### Captioning Layer

The custom `BlipCaptionSummaryLayer` subclasses `tf.keras.layers.Layer` and bridges the TensorFlow graph to the PyTorch BLIP backbone via `tf.py_function`. It exposes two operating modes selected by an input task tensor:

- **`caption`** &mdash; concise descriptive caption, prompted with `"This is a picture of"`.
- **`summary`** &mdash; longer descriptive narrative, prompted with `"This is a detailed photo showing"`.

This design keeps the entire inspection workflow expressible as Keras layers while leveraging a state-of-the-art vision-language model with no retraining required.

---

## Dataset

The project uses the public **Aircraft Damage Dataset**, a curated collection of labelled aircraft surface images partitioned into `train`, `valid`, and `test` splits across two classes.

```
aircraft_damage_dataset_v1/
├── train/
│   ├── dent/
│   └── crack/
├── valid/
│   ├── dent/
│   └── crack/
└── test/
    ├── dent/
    └── crack/
```

- **Source:** [Roboflow Aircraft Damage Dataset](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-tnxzm)
- **Classes:** `dent`, `crack`
- **Input resolution:** 224 x 224 (matching VGG16 expectations)

---

## Repository Structure

```
.
├── Aircraft_Damage_Classification_and_Captioning.ipynb   # Main notebook
├── README.md                                              # This file
└── best_aircraft_model.keras                              # Saved checkpoint (generated at runtime)
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A CUDA-capable GPU is strongly recommended for training; CPU is acceptable for BLIP inference.

### Installation

```bash
git clone https://github.com/<your-username>/aircraft-damage-pipeline.git
cd aircraft-damage-pipeline

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install tensorflow keras torch transformers pillow matplotlib numpy
```

### Run

```bash
jupyter notebook Aircraft_Damage_Classification_and_Captioning.ipynb
```

The notebook downloads the dataset automatically on first run, trains the classifier, evaluates on the test split, and demonstrates BLIP captioning end-to-end.

---

## Results

| Metric | Value |
| --- | --- |
| Test Accuracy | ~87.5% |
| Test Loss | ~0.61 |
| Backbone | VGG16 (frozen, ImageNet weights) |
| Trainable Parameters | Custom dense head only |

Sample BLIP outputs on test imagery:

```
Caption: this is a picture of a plane
Summary: this is a detailed photo showing the engine of a boeing 747
```

Quantitative results vary across runs due to GPU non-determinism and stochastic data augmentation.

---

## Design Decisions

- **Frozen VGG16 base.** With limited domain-specific data, freezing the convolutional base preserves rich ImageNet features and reduces the risk of overfitting. Selective unfreezing of upper blocks is a natural next step.
- **Conservative learning rate (`1e-4`).** Standard practice when fine-tuning a head on top of a frozen pretrained backbone.
- **EarlyStopping with restore-best-weights.** Guards against overfitting and removes the need to manually pick a stopping epoch.
- **Custom Keras layer over a standalone script.** Encapsulating BLIP as a `tf.keras.layers.Layer` keeps the entire pipeline composable inside Keras and aligned with the rest of the codebase.

---

## Roadmap

- Unfreeze the upper VGG16 blocks and fine-tune for additional accuracy gains.
- Replace VGG16 with a modern backbone (EfficientNet, ConvNeXt, ViT) and benchmark.
- Extend to multi-class defect taxonomies beyond `dent` / `crack`.
- Fine-tune BLIP on aviation-specific captions for domain-aligned vocabulary.
- Wrap the pipeline in a REST or gRPC service for production deployment.
- Add Grad-CAM visualizations for classifier explainability.

---

## License

Released under the MIT License. The underlying pretrained models retain their original licenses (VGG16 weights via Keras applications; BLIP weights under the Salesforce BLIP license on Hugging Face).

---

## Acknowledgements

- **Salesforce Research** for the BLIP vision-language model.
- **Hugging Face** for distribution and tooling around pretrained transformers.
- **Roboflow** community for the source aircraft damage imagery.

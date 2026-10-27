<div align="center">

# 🐶🐱 Dog vs Cat Image Classification with a CNN

**A three-layer Convolutional Neural Network, built with TensorFlow/Keras, that tells dogs from cats — trained on a Hugging Face dataset and ready to run in Google Colab.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Open in Colab](https://img.shields.io/badge/Google%20Colab-ready-F9AB00?logo=googlecolab&logoColor=white)

</div>

---

## 📖 Overview

This project develops a **Convolutional Neural Network (CNN) with exactly three convolutional layers** to solve a classic computer-vision task: **binary image classification of cats and dogs**.

The single script, `cnn_cats_dogs_colab.py`, covers the whole pipeline:

> download dataset → preprocess images → build CNN → train with early stopping → evaluate → visualize → predict → save model & predictions

## ✨ Highlights

- 🤗 **Loads data straight from Hugging Face** — [`louiecerv/cats_dogs_dataset`](https://huggingface.co/datasets/louiecerv/cats_dogs_dataset), no manual download
- 🧱 **Robust image loader** that handles PIL images, byte/path dictionaries, NumPy arrays and lists
- 🧠 **3 × (Conv2D → MaxPooling)** feature extractor, followed by a dense classifier with **Dropout (0.5)**
- ⏹️ **EarlyStopping** on validation loss (patience 3, best weights restored) to avoid overfitting
- 📊 Complete evaluation: accuracy, precision/recall/F1, confusion matrix, learning curves
- 🔍 Single-image prediction with **confidence score**
- 💾 Saves the trained model (`dog_cat_cnn.keras`) and per-image test predictions (`dog_cat_predictions.csv`)
- 🔁 **Reproducible** — fixed seeds for NumPy, TensorFlow and the validation split

## 🏗️ Model Architecture

```
Input (128 × 128 × 3)
      │
Rescaling (1/255)                       → pixels scaled to [0, 1]
      │
Conv2D(32,  3×3, ReLU, same) → MaxPool(2×2)     Block 1
      │
Conv2D(64,  3×3, ReLU, same) → MaxPool(2×2)     Block 2
      │
Conv2D(128, 3×3, ReLU, same) → MaxPool(2×2)     Block 3
      │
Flatten
      │
Dense(128, ReLU)
      │
Dropout(0.5)
      │
Dense(1, Sigmoid)  →  P(dog)
```

| Setting          | Value                                    |
| ---------------- | ---------------------------------------- |
| Input size       | 128 × 128 RGB                            |
| Parameters       | ≈ 4.29 million                           |
| Optimizer        | Adam                                     |
| Loss             | Binary cross-entropy                     |
| Batch size       | 32                                       |
| Max epochs       | 15 (with early stopping, patience = 3)   |
| Validation split | 20% of the training set (seeded shuffle) |
| Decision rule    | predict **dog** if probability ≥ 0.5     |
| Labels           | `0` = cat, `1` = dog                     |

## 🔬 Pipeline

| Step | Description |
| :--: | ----------- |
| 1    | Install `datasets`, import libraries, set random seeds |
| 2    | Load `louiecerv/cats_dogs_dataset` from Hugging Face |
| 3–4  | Convert every image to RGB, resize to 128 × 128 (Lanczos), stack into arrays |
| 5    | Display a 3 × 3 grid of sample images |
| 6–7  | Build and compile the CNN |
| 8    | Carve a reproducible 80/20 train/validation split |
| 9    | Train with `EarlyStopping` |
| 10–12| Evaluate on the unseen test set; print the classification report |
| 13   | Plot the confusion matrix |
| 14   | Plot training vs. validation accuracy and loss |
| 15   | Predict one test image and show the confidence |
| 16–17| Save `dog_cat_cnn.keras` and `dog_cat_predictions.csv` |

## 📈 Results

Run the script to reproduce the metrics; they are printed at the end of training and shown in the plots.

| Metric              | Value |
| ------------------- | ----- |
| Test accuracy       | _add after running_ |
| Test loss           | _add after running_ |
| Epochs trained      | _add after running_ |

Add your plots here after a run — for example, save them into an `assets/` folder and embed them:

```markdown
![Confusion matrix](assets/confusion_matrix.png)
![Training curves](assets/training_curves.png)
```

## 🚀 Getting Started

### Option 1 — Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Optional but faster: **Runtime → Change runtime type → GPU**.
3. Paste the contents of `cnn_cats_dogs_colab.py` into a cell and run it.

The script begins with `!pip install -q datasets`, a notebook command that works in Colab and Jupyter.

### Option 2 — Run locally

The `!pip` line is notebook-only syntax, so remove or comment it out for a plain Python run (the dependencies come from `requirements.txt` instead).

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Comment out the first line of the script:  # !pip install -q datasets
# 5. Run
python cnn_cats_dogs_colab.py
```

Plot windows open one after another — close each to continue. An internet connection is needed on the first run to download the dataset. Training is much faster on a GPU.

## 🔮 Using the Trained Model

```python
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model("dog_cat_cnn.keras")

img = Image.open("my_pet.jpg").convert("RGB").resize((128, 128))
x = np.asarray(img, dtype=np.uint8)[np.newaxis, ...]   # model rescales internally

p = float(model.predict(x, verbose=0)[0][0])
print("dog" if p >= 0.5 else "cat", f"(P(dog) = {p:.2f})")
```

> Pixel scaling is part of the model (`Rescaling` layer), so feed raw 0–255 pixel values.

## 📦 Outputs

| File                      | Contents                                                   |
| ------------------------- | ---------------------------------------------------------- |
| `dog_cat_cnn.keras`       | Trained Keras model                                        |
| `dog_cat_predictions.csv` | `Actual_Label`, `Predicted_Label`, `Dog_Probability` per test image |

## 📁 Project Structure

```
.
├── cnn_cats_dogs_colab.py   # Complete pipeline: data → CNN → evaluation → export
├── requirements.txt         # Python dependencies
└── README.md
```

## 🛠️ Tech Stack

- **[TensorFlow / Keras](https://www.tensorflow.org/)** — CNN
- **[Hugging Face Datasets](https://huggingface.co/docs/datasets)** — data loading
- **[scikit-learn](https://scikit-learn.org/)** — metrics
- **[Pillow](https://python-pillow.org/)**, **[NumPy](https://numpy.org/)**, **[pandas](https://pandas.pydata.org/)** — image and data handling
- **[Matplotlib](https://matplotlib.org/)** — visualization

## 🎛️ Experiment Ideas

- Add **data augmentation** (`RandomFlip`, `RandomRotation`, `RandomZoom`) to improve generalization
- Add **BatchNormalization** or more filters per block
- Swap the flatten-and-dense head for `GlobalAveragePooling2D`
- Try **transfer learning** (MobileNetV2, ResNet50) for a big accuracy jump
- Tune the learning rate, dropout rate, or early-stopping patience
- Use `tf.data` pipelines to reduce memory use on large datasets

## 🙏 Acknowledgements

Dataset: [`louiecerv/cats_dogs_dataset`](https://huggingface.co/datasets/louiecerv/cats_dogs_dataset) on Hugging Face.

## 🤝 Contributing

Suggestions and pull requests are welcome. Fork the repo, create a feature branch, and open a PR describing your change.

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

---

<div align="center">

If you found this useful, consider giving the repo a ⭐

</div>

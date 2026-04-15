# 🌿 CropID — AI Crop Identification & Nutrition System

A multimodal AI system that identifies crops from photos using a local TensorFlow model
and generates detailed human diet & nutrition reports using Groq LLaMA Vision.

---

## 📁 Project Structure

```
CropID/
│
├── dataset/                  ← YOUR TRAINING DATA (you already have this)
│   ├── wheat/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── rice/
│   │   └── ...
│   └── (one folder per crop class)
│
├── app.py                    ← Streamlit web app
├── train.py                  ← Model training script
├── requirements.txt          ← Python dependencies
├── .env.example              ← Copy to .env and add your Groq API key
│
│   ─── Generated after training ───
├── crop_model.h5             ← Trained TensorFlow model
├── labels.txt                ← Class names in order
├── confusion_matrix.png      ← Per-class confusion heatmap
└── training_curves.png       ← Accuracy / loss graphs
```

---

## 🚀 Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Set up your API key
```bash
# Copy the example file
cp .env.example .env

# Open .env and paste your Groq API key
# Get a free key at: https://console.groq.com
```

### Step 3 — Train the model
Make sure your `dataset/` folder is ready (one subfolder per crop class with images inside).
```bash
python train.py
```
This will print **Accuracy, Precision, Recall, F1** to the terminal and save:
- `crop_model.h5` — the trained model
- `labels.txt` — class names
- `confusion_matrix.png` — visual heatmap of predictions
- `training_curves.png` — accuracy and loss graphs

### Step 4 — Run the app
```bash
streamlit run app.py
```

---

## 📊 Training Output (terminal)

After training you will see something like:

```
════════════════════════════════════════════════════════════
  📊  EVALUATION METRICS
════════════════════════════════════════════════════════════

  ✅  Overall Accuracy   : 94.32 %
  ✅  Weighted Precision : 93.87 %
  ✅  Weighted Recall    : 94.32 %
  ✅  Weighted F1-Score  : 93.98 %

──────────────────────────────────────────────────────────
  Per-Class Classification Report
──────────────────────────────────────────────────────────
              precision    recall  f1-score   support

       wheat       0.96      0.97      0.97       120
        rice       0.93      0.94      0.93       115
       maize       0.91      0.90      0.90       108
         ...
```

---

## ✨ Features

| Feature | Details |
|---|---|
| 🤖 Local ML Identification | MobileNetV2 fine-tuned on your dataset |
| 🌫️ Unclear Image Handling | Auto-sharpening + brightness correction + retry |
| 📊 Confidence Score | Top-5 predictions with % confidence |
| 🥗 Diet & Nutrition Report | 7-section report incl. weekly meal plan |
| 💬 AI Chatbot | LLaMA 70B crop & nutrition expert |
| 📸 Multiple Input Sources | Upload, URL, or Webcam |
| 📈 Training Metrics | Accuracy, Precision, Recall, F1, Confusion Matrix |

---

## 🧠 Model Architecture

```
MobileNetV2 (ImageNet pretrained)
    └── GlobalAveragePooling2D
    └── BatchNormalization
    └── Dense(512, relu)  + Dropout(0.45)
    └── Dense(256, relu)  + Dropout(0.35)
    └── Dense(num_classes, softmax)
```

**Training strategy:**
- **Phase 1** (15 epochs): Only the classification head is trained; MobileNetV2 base is frozen.
- **Phase 2** (15 epochs): Top 40 layers of MobileNetV2 are unfrozen and fine-tuned with a very low learning rate (1e-5).
- `EarlyStopping` + `ReduceLROnPlateau` + `ModelCheckpoint` used throughout.

---

## 🗂️ Dataset Format

```
dataset/
├── wheat/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...          (minimum 50 images per class recommended)
├── rice/
│   └── ...
└── maize/
    └── ...
```
- Images can be `.jpg`, `.jpeg`, or `.png`
- At least **50–100 images per class** for good accuracy
- More images = better accuracy

---

## 💡 Tips for Better Accuracy

1. **More data** — 200+ images per class is ideal
2. **Diverse images** — different angles, lighting, backgrounds
3. **Clean labels** — make sure images are in the correct folder
4. **Clear photos** — well-lit, crop fills most of the frame

---

## 🔑 API Keys

| Service | Purpose | Free Tier |
|---|---|---|
| [Groq](https://console.groq.com) | Vision + Chat AI | ✅ Yes |

---

## 📦 Dependencies

- `tensorflow` — local crop identification model
- `streamlit` — web interface
- `groq` — LLaMA Vision & Chat API
- `Pillow` — image processing & enhancement
- `scikit-learn` — accuracy/precision/F1 metrics
- `matplotlib` / `seaborn` — training graphs & confusion matrix

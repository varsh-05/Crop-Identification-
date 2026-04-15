# 🌿 CropID — AI Crop Identification & Nutrition System

A multimodal AI system that identifies crops from photos using a local TensorFlow model
and Groq LLaMA Vision AI, then generates detailed human diet & nutrition reports.

---

## 📁 Project Structure

```
ML_crop/
│
├── dataset/                    ← YOUR TRAINING DATA
│   ├── wheat/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── rice/
│   │   └── ...
│   └── (one folder per crop class)
│
├── cropenv/                    ← Python virtual environment
├── app.py                      ← Streamlit web app
├── train.py                    ← Model training script
├── validate.py                 ← Prediction validator
├── evaluate.py                 ← Local ML model evaluator
├── evaluate_groq.py            ← Groq Vision AI evaluator
├── download_images.py          ← Multi-source image downloader
├── requirements.txt            ← Python dependencies
├── .env                        ← Your Groq API key (never share this)
├── CropID_Colab.ipynb          ← Google Colab notebook (GPU training)
│
│   ─── Generated after training ───
├── crop_model.h5               ← Trained TensorFlow model
├── best_crop_model.h5          ← Best checkpoint during training
├── labels.txt                  ← Class names in order
├── confusion_matrix.png        ← Per-class confusion heatmap
└── training_curves.png         ← Accuracy / loss graphs
```

---

## 🚀 Quick Start (macOS — VS Code)

### Step 1 — Activate virtual environment
```bash
cd ~/Documents/Projects/ML_crop
source cropenv/bin/activate
```

### Step 2 — Set up your API key
Create a `.env` file in the project folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at: **https://console.groq.com**

### Step 3 — Train the model
```bash
python train.py
```
This prints **Accuracy, Precision, Recall, F1** to the terminal and saves:
- `crop_model.h5` — trained model
- `labels.txt` — class names
- `confusion_matrix.png` — visual heatmap
- `training_curves.png` — accuracy and loss graphs

### Step 4 — Run the app
```bash
python -m streamlit run app.py
```
Opens at **http://localhost:8501**

> ⚠️ Always use `python -m streamlit` not just `streamlit` to ensure the correct Python environment is used.

---

## ⚡ Train Faster with Google Colab (Recommended)

Use `CropID_Colab.ipynb` for **10x faster GPU training**:

1. Go to **https://colab.research.google.com**
2. Upload `CropID_Colab.ipynb`
3. Set runtime: **Runtime → Change runtime type → T4 GPU**
4. Run all cells in order
5. Download `crop_model.h5` and `labels.txt`
6. Copy both files to your `ML_crop` folder
7. Run the app locally

| | Mac (VS Code) | Google Colab |
|---|---|---|
| GPU | ❌ CPU only | ✅ Free T4 GPU |
| Training time | 30–60 mins | ⚡ 5–10 mins |
| Cost | Free | Free |

---

## 📊 Training Output (terminal)

```
══════════════════════════════════════════════════════════════
  📊  EVALUATION METRICS
══════════════════════════════════════════════════════════════

  ✅  Overall Accuracy   : 86.00 %
  ✅  Weighted Precision : 85.50 %
  ✅  Weighted Recall    : 86.00 %
  ✅  Weighted F1-Score  : 85.70 %

──────────────────────────────────────────────────────────────
  Per-Class Classification Report
──────────────────────────────────────────────────────────────
              precision    recall  f1-score   support

       wheat       0.96      0.97      0.97       120
        rice       0.93      0.94      0.93       115
       maize       0.91      0.90      0.90       108
         ...
```

---

## ✅ Validating Predictions

Use `validate.py` to prove the model is predicting correctly against known labels.

### Validate a single image:
```bash
python validate.py --image dataset/wheat/img1.jpg --true_label wheat
```

### Validate all images in a class folder:
```bash
python validate.py --folder dataset/wheat --true_label wheat
```

### Validate entire dataset:
```bash
python validate.py --all --samples 10
```

### Validate with both Local ML + Groq Vision:
```bash
python validate.py --all --samples 5 --groq
```

**Output example:**
```
══════════════════════════════════════════════
  VALIDATION RESULT
══════════════════════════════════════════════
  Image       : img1.jpg
  True Label  : wheat        ← what it SHOULD be
  Local ML:
    Predicted : wheat        ← what model PREDICTED
    Confidence: 86%
    Result    : ✅ CORRECT
══════════════════════════════════════════════
```

---

## 📈 Evaluating Model Accuracy

### Local ML model only (fast):
```bash
python evaluate.py
```

### Groq Vision AI accuracy (sends images to API):
```bash
python evaluate_groq.py
```

---

## 🖼️ Downloading More Training Images

To improve accuracy, download more images per class:
```bash
caffeinate -i python download_images.py
```
Downloads **100 images per class** from Google + Bing automatically.
Then retrain: `python train.py`

---

## ✨ Features

| Feature | Details |
|---|---|
| 🤖 Local ML Identification | EfficientNetB0 fine-tuned on your dataset |
| 🧠 Groq Vision AI | Primary identifier — LLaMA Vision (very accurate) |
| 🌫️ Unclear Image Handling | Auto-sharpening + brightness correction + retry |
| 📊 Top 5 Predictions | Groq Vision alternatives with confidence % |
| 🥗 Diet & Nutrition Report | 7-section report including weekly meal plan |
| 💬 AI Chatbot | LLaMA 70B crop & nutrition expert |
| 📸 Multiple Input Sources | Upload file, Image URL, or Webcam |
| 🔗 URL Image Loading | Works with any image URL (SSL + browser headers) |
| ✅ Chat Auto-Clear | Chat resets automatically when new image is uploaded |
| 📈 Training Metrics | Accuracy, Precision, Recall, F1, Confusion Matrix |
| 🔍 Prediction Validator | Validates predictions against ground truth labels |

---

## 🧠 Model Architecture

```
EfficientNetB0 (ImageNet pretrained)
    └── GlobalAveragePooling2D
    └── BatchNormalization
    └── Dense(512, relu)  + Dropout(0.50)
    └── Dense(256, relu)  + Dropout(0.40)
    └── Dense(num_classes, softmax)
```

**Training strategy:**
- **Phase 1** (25 epochs): Head only trained, EfficientNetB0 base frozen
- **Phase 2** (25 epochs): Top 60 layers unfrozen and fine-tuned (lr=2e-6)
- `EarlyStopping` + `ReduceLROnPlateau` + `ModelCheckpoint` throughout
- **Class weights** used to handle imbalanced datasets
- **Batch size 16** for better learning on small datasets

**Identification flow:**
```
Upload Image
     ↓
Groq Vision AI (primary — very accurate)
     ↓ if fails
Local ML Model (fallback)
     ↓
Diet & Nutrition Report generated
```

---

## 🗂️ Dataset Format

```
dataset/
├── wheat/
│   ├── 001.jpg
│   └── ...    (100 images per class recommended)
├── rice/
│   └── ...
└── maize/
    └── ...
```

| Images per class | Expected Accuracy |
|---|---|
| 5–10 | ~10–30% |
| 30–50 | ~70–78% |
| 80–100 | ~82–88% |
| 150+ | ~90–95% |

---

## 💡 Tips for Better Accuracy

1. **More data** — 100+ images per class is ideal
2. **Diverse images** — different angles, lighting, backgrounds
3. **Clean labels** — make sure images are in the correct folder
4. **Clear photos** — well-lit, crop fills most of the frame
5. **Use Colab** — GPU training gives same accuracy 10x faster

---

## 🔑 API Keys

| Service | Purpose | Free Tier | Link |
|---|---|---|---|
| Groq | Vision + Chat AI | ✅ Yes | https://console.groq.com |
| ngrok | Run app publicly from Colab | ✅ Yes | https://dashboard.ngrok.com |

---

## 📦 Dependencies

```
streamlit==1.32.0       — web interface
tensorflow-macos==2.13  — local crop identification model
groq==0.5.0             — LLaMA Vision & Chat API
Pillow>=10.0.0          — image processing & enhancement
scikit-learn>=1.3.0     — accuracy/precision/F1 metrics
matplotlib>=3.8.0       — training graphs
seaborn>=0.13.0         — confusion matrix
python-dotenv>=1.0.0    — API key management
icrawler                — image downloading
```

---

## ⚠️ Troubleshooting

| Error | Fix |
|---|---|
| `tensorflow not found` | Run `source cropenv/bin/activate` first |
| `GROQ_API_KEY not found` | Check `.env` file has your key |
| `crop_model.h5 not found` | Run `python train.py` first |
| `labels.txt not found` | Run `python train.py` first |
| `use_container_width error` | Use `python -m streamlit run app.py` |
| Wrong predictions | Add more images + retrain |
| Low accuracy | Run `python download_images.py` then `python train.py` |
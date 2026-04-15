"""
CropID — Model Training Pipeline (EfficientNetB0 — higher accuracy)
=====================================================================
Usage:  python train.py

Outputs:
    crop_model.h5          ← trained model
    labels.txt             ← class names
    confusion_matrix.png   ← heatmap
    training_curves.png    ← accuracy / loss graphs

Terminal prints: Accuracy, Precision, Recall, F1 + per-class report.
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D,
                                     Dropout, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════
DATASET_PATH    = 'dataset/'
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 16
EPOCHS_FROZEN   = 30   # Phase 1 — head only
EPOCHS_FINETUNE = 30   # Phase 2 — fine-tune top layers

print("\n" + "═" * 62)
print("    🌿  CropID — EfficientNetB0 Training Pipeline  🌿")
print("═" * 62 + "\n")

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset folder '{DATASET_PATH}' not found.\n"
        "Place images under:  dataset/<class_name>/image.jpg"
    )

# ══════════════════════════════════════════════
#  DATA GENERATORS
#  EfficientNet needs pixel values in [0,255]
#  — do NOT rescale, use preprocess_input instead
# ══════════════════════════════════════════════
from tensorflow.keras.applications.efficientnet import preprocess_input

def efficientnet_preprocess(x):
    return preprocess_input(x)

train_aug = ImageDataGenerator(
    preprocessing_function = efficientnet_preprocess,
    rotation_range         = 40,
    width_shift_range      = 0.25,
    height_shift_range     = 0.25,
    shear_range            = 0.2,
    zoom_range             = 0.35,
    horizontal_flip        = True,
    brightness_range       = [0.5, 1.5],
    fill_mode              = 'reflect',
    validation_split       = 0.2
)

val_aug = ImageDataGenerator(
    preprocessing_function = efficientnet_preprocess,
    validation_split       = 0.2
)

train_gen = train_aug.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training', shuffle=True
)
val_gen = val_aug.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', shuffle=False
)

NUM_CLASSES = train_gen.num_classes
print(f"  Classes   : {NUM_CLASSES}")
print(f"  Train     : {train_gen.samples} images")
print(f"  Validate  : {val_gen.samples} images\n")

# ══════════════════════════════════════════════
#  MODEL — EfficientNetB0
#  More accurate than MobileNetV2 with same size
# ══════════════════════════════════════════════
base = EfficientNetB0(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))
base.trainable = False  # freeze in phase 1

x   = GlobalAveragePooling2D()(base.output)
x   = BatchNormalization()(x)
x   = Dense(512, activation='relu')(x)
x   = Dropout(0.4)(x)
x   = Dense(256, activation='relu')(x)
x   = Dropout(0.3)(x)
out = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base.input, outputs=out)

# ══════════════════════════════════════════════
#  PHASE 1 — Train head only
# ══════════════════════════════════════════════
print("─" * 62)
print("  Phase 1 › Head training  (EfficientNetB0 frozen)")
print("─" * 62)

model.compile(
    optimizer = tf.keras.optimizers.Adam(1e-3),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

cb1 = [
    EarlyStopping(monitor='val_accuracy', patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                     patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint('best_crop_model.h5', monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

h1 = model.fit(train_gen, validation_data=val_gen,
               epochs=EPOCHS_FROZEN, callbacks=cb1)

# ══════════════════════════════════════════════
#  PHASE 2 — Fine-tune top 50 layers
# ══════════════════════════════════════════════
print("\n" + "─" * 62)
print("  Phase 2 › Fine-tuning top 50 layers of EfficientNetB0")
print("─" * 62)

for layer in base.layers[-50:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer = tf.keras.optimizers.Adam(5e-6),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

cb2 = [
    EarlyStopping(monitor='val_accuracy', patience=6,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                     patience=3, min_lr=1e-8, verbose=1),
    ModelCheckpoint('best_crop_model.h5', monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

h2 = model.fit(train_gen, validation_data=val_gen,
               epochs=EPOCHS_FINETUNE, callbacks=cb2)

# ══════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════
print("\n" + "═" * 62)
print("  📊  EVALUATION METRICS")
print("═" * 62)

best = tf.keras.models.load_model('best_crop_model.h5')

val_gen.reset()
y_prob = best.predict(val_gen, verbose=1)
y_pred = np.argmax(y_prob, axis=1)
y_true = val_gen.classes

label_map   = train_gen.class_indices
label_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec  = recall_score(y_true,  y_pred,   average='weighted', zero_division=0)
f1   = f1_score(y_true,      y_pred,   average='weighted', zero_division=0)

print(f"\n  ✅  Overall Accuracy   : {acc  * 100:.2f} %")
print(f"  ✅  Weighted Precision : {prec * 100:.2f} %")
print(f"  ✅  Weighted Recall    : {rec  * 100:.2f} %")
print(f"  ✅  Weighted F1-Score  : {f1   * 100:.2f} %")

print("\n" + "─" * 62)
print("  Per-Class Classification Report")
print("─" * 62)
label_names = [k for k, _ in sorted(train_gen.class_indices.items(), key=lambda x: x[1])]
print(classification_report(y_true, y_pred,
                             target_names=label_names, zero_division=0))

# ══════════════════════════════════════════════
#  CONFUSION MATRIX
# ══════════════════════════════════════════════
cm     = confusion_matrix(y_true, y_pred)
fig_sz = max(12, NUM_CLASSES)
plt.figure(figsize=(fig_sz, fig_sz - 2))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=label_names, yticklabels=label_names,
            linewidths=0.4)
plt.title('Confusion Matrix — EfficientNetB0', fontsize=15, pad=15)
plt.ylabel('True Label',      fontsize=11)
plt.xlabel('Predicted Label', fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=130)
print("\n  📊  confusion_matrix.png saved.")

# ══════════════════════════════════════════════
#  TRAINING CURVES
# ══════════════════════════════════════════════
def _cat(key): return h1.history[key] + h2.history[key]

split = len(h1.history['accuracy'])
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('CropID — EfficientNetB0 Training Curves', fontsize=14)

for ax, (tr_k, val_k), title in zip(
        axes,
        [('accuracy','val_accuracy'), ('loss','val_loss')],
        ['Accuracy', 'Loss']):
    ax.plot(_cat(tr_k),  label='Train',      color='#2ecc71', lw=2)
    ax.plot(_cat(val_k), label='Validation', color='#e74c3c', lw=2)
    ax.axvline(split-1, color='#3498db', ls='--', lw=1.5, label='Fine-tune start')
    ax.set_title(title); ax.legend(); ax.set_xlabel('Epoch'); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=130)
print("  📈  training_curves.png  saved.")

# ══════════════════════════════════════════════
#  SAVE FINAL ASSETS
# ══════════════════════════════════════════════
best.save('crop_model.h5')
with open('labels.txt', 'w') as f:
    for name in label_names:
        f.write(f"{name}\n")

print("\n" + "═" * 62)
print("  ✅  crop_model.h5  saved")
print("  ✅  labels.txt     saved")
print(f"\n  Final Val Accuracy : {acc * 100:.2f} %")
print("═" * 62 + "\n")

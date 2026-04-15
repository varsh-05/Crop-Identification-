import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

print("\nLoading model...")
model = tf.keras.models.load_model('crop_model.h5')

val_gen = ImageDataGenerator(
    rescale=1./255, validation_split=0.2
).flow_from_directory(
    'dataset/', target_size=(224,224), batch_size=32,
    class_mode='categorical', subset='validation', shuffle=False
)

print("Evaluating...")
y_prob  = model.predict(val_gen, verbose=1)
y_pred  = np.argmax(y_prob, axis=1)
y_true  = val_gen.classes
labels  = [k for k,_ in sorted(val_gen.class_indices.items(), key=lambda x: x[1])]

acc  = accuracy_score(y_true, y_pred) * 100
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
rec  = recall_score(y_true, y_pred,    average='weighted', zero_division=0) * 100
f1   = f1_score(y_true, y_pred,        average='weighted', zero_division=0) * 100

print("\n" + "="*45)
print("         MODEL EVALUATION RESULTS")
print("="*45)
print(f"  Accuracy  : {acc:.2f} %")
print(f"  Precision : {prec:.2f} %")
print(f"  Recall    : {rec:.2f} %")
print(f"  F1-Score  : {f1:.2f} %")
print("="*45)
print("\nPer-Class Report:")
print(classification_report(y_true, y_pred,
                             target_names=labels, zero_division=0))

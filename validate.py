"""
CropID — Prediction Validator
==============================
Validates whether the model prediction is correct
by comparing it against the true/known label.

Usage:
    # Test a single image with known label
    python validate.py --image dataset/wheat/img1.jpg --true_label wheat

    # Test all images in a folder
    python validate.py --folder dataset/wheat --true_label wheat

    # Test entire dataset and get full report
    python validate.py --all

    # Test with samples per class
    python validate.py --all --samples 10
"""

import os, sys, argparse, base64, time
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter, ImageEnhance
from io import BytesIO
from groq import Groq
from dotenv import load_dotenv
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

# ══════════════════════════════════════════════
#  LOAD ASSETS
# ══════════════════════════════════════════════
print("\n  Loading model and labels...")
model = tf.keras.models.load_model('crop_model.h5')
with open('labels.txt') as f:
    crop_labels = [l.strip() for l in f if l.strip()]

client       = Groq(api_key=os.getenv("GROQ_API_KEY"))
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DATASET_PATH = "dataset/"

# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════
def encode_image(img: Image.Image) -> str:
    buf = BytesIO()
    img.convert("RGB").resize((512, 512)).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def enhance_image(image: Image.Image) -> Image.Image:
    img = image.convert("RGB")
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Contrast(img).enhance(1.5)
    mean_lum = np.array(img).mean()
    if mean_lum < 70:
        img = ImageEnhance.Brightness(img).enhance(1.7)
    elif mean_lum > 210:
        img = ImageEnhance.Brightness(img).enhance(0.75)
    return img

def local_predict(img: Image.Image):
    def _run(i):
        i   = i.convert("RGB").resize((224, 224))
        arr = tf.keras.preprocessing.image.img_to_array(i) / 255.0
        arr = np.expand_dims(arr, 0)
        return model.predict(arr, verbose=0)[0]
    probs = _run(img)
    conf  = float(np.max(probs))
    if conf < 0.45:
        p2 = _run(enhance_image(img))
        if float(np.max(p2)) > conf:
            probs, conf = p2, float(np.max(p2))
    idx  = int(np.argmax(probs))
    top5 = [(crop_labels[i], round(float(probs[i])*100,1))
            for i in np.argsort(probs)[::-1][:5]]
    return crop_labels[idx], round(conf*100, 1), top5

def groq_predict(img: Image.Image):
    b64       = encode_image(img)
    supported = ", ".join(crop_labels)
    prompt    = f"""Identify the crop/plant in this image.
Choose ONLY from: {supported}
Reply ONLY:
CROP: <name>
CONFIDENCE: <1-100>"""
    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role":"user","content":[
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
            ]}],
            max_tokens=50
        )
        text, pred, conf = resp.choices[0].message.content.strip(), "unknown", 0
        for line in text.split('\n'):
            if line.startswith("CROP:"):
                pred = line.replace("CROP:","").strip()
            elif line.startswith("CONFIDENCE:"):
                try: conf = int(line.replace("CONFIDENCE:","").strip())
                except: conf = 0
        # match to labels
        for c in crop_labels:
            if c.lower() == pred.lower(): return c, conf
        for c in crop_labels:
            if pred.lower() in c.lower() or c.lower() in pred.lower(): return c, conf
        return pred, conf
    except Exception as e:
        return "error", 0

def validate_single(img_path: str, true_label: str, use_groq=True):
    """
    Validate one image against its true label.
    Returns dict with full validation result.
    """
    img = Image.open(img_path).convert("RGB")

    # Local ML prediction
    local_pred, local_conf, top5 = local_predict(img)

    # Groq Vision prediction
    groq_pred, groq_conf = groq_predict(img) if use_groq else ("N/A", 0)

    local_correct = local_pred.lower() == true_label.lower()
    groq_correct  = groq_pred.lower()  == true_label.lower() if use_groq else None

    return {
        "image"        : os.path.basename(img_path),
        "true_label"   : true_label,
        "local_pred"   : local_pred,
        "local_conf"   : local_conf,
        "local_correct": local_correct,
        "groq_pred"    : groq_pred,
        "groq_conf"    : groq_conf,
        "groq_correct" : groq_correct,
        "top5"         : top5,
    }

def print_single_result(r: dict):
    print("\n" + "═"*60)
    print("  VALIDATION RESULT")
    print("═"*60)
    print(f"  Image       : {r['image']}")
    print(f"  True Label  : {r['true_label']}")
    print()
    local_icon = "✅ CORRECT" if r['local_correct'] else "❌ WRONG"
    print(f"  Local ML Model:")
    print(f"    Predicted : {r['local_pred']}")
    print(f"    Confidence: {r['local_conf']}%")
    print(f"    Result    : {local_icon}")
    print()
    print(f"  Top 5 Predictions (Local ML):")
    for i, (name, pct) in enumerate(r['top5'], 1):
        tick = "←✅" if name.lower() == r['true_label'].lower() else ""
        print(f"    {i}. {name:<25} {pct}% {tick}")
    print()
    if r['groq_pred'] != "N/A":
        groq_icon = "✅ CORRECT" if r['groq_correct'] else "❌ WRONG"
        print(f"  Groq Vision AI:")
        print(f"    Predicted : {r['groq_pred']}")
        print(f"    Confidence: {r['groq_conf']}%")
        print(f"    Result    : {groq_icon}")
    print("═"*60)

# ══════════════════════════════════════════════
#  FULL DATASET VALIDATION
# ══════════════════════════════════════════════
def validate_all(samples_per_class=5, use_groq=False):
    """Test model on known images from dataset and compute metrics."""
    classes = sorted([
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
        and not d.startswith('.')
    ])

    print("\n" + "═"*60)
    print("  🌿  CropID — Full Dataset Validation")
    print("═"*60)
    print(f"  Classes       : {len(classes)}")
    print(f"  Samples/class : {samples_per_class}")
    print(f"  Total tests   : ~{len(classes) * samples_per_class}")
    print(f"  Groq Vision   : {'Yes' if use_groq else 'No (local ML only)'}")
    print("═"*60 + "\n")

    y_true_local, y_pred_local = [], []
    y_true_groq,  y_pred_groq  = [], []
    results = []

    for cls_idx, cls_name in enumerate(classes):
        cls_path = os.path.join(DATASET_PATH, cls_name)
        images   = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))
        ][:samples_per_class]

        if not images:
            print(f"  ⚠️  No images: {cls_name}")
            continue

        correct_local = 0
        correct_groq  = 0
        print(f"  [{cls_idx+1:2d}/{len(classes)}] {cls_name} ({len(images)} images)")

        for img_file in images:
            img_path = os.path.join(cls_path, img_file)
            try:
                r = validate_single(img_path, cls_name, use_groq=use_groq)
                results.append(r)

                y_true_local.append(cls_name)
                y_pred_local.append(r['local_pred'])

                local_icon = "✅" if r['local_correct'] else "❌"
                print(f"       {local_icon} Local: {r['local_pred']:<22} ({r['local_conf']}%)", end="")

                if use_groq:
                    y_true_groq.append(cls_name)
                    y_pred_groq.append(r['groq_pred'])
                    groq_icon = "✅" if r['groq_correct'] else "❌"
                    print(f"  |  {groq_icon} Groq: {r['groq_pred']:<22} ({r['groq_conf']}%)", end="")
                    time.sleep(1)

                print()
                if r['local_correct']: correct_local += 1
                if use_groq and r['groq_correct']: correct_groq += 1

            except Exception as e:
                print(f"       ⚠️  Error: {e}")

        local_pct = correct_local / len(images) * 100
        print(f"         Local ML: {correct_local}/{len(images)} ({local_pct:.0f}%)")

    # ── Metrics ──
    print("\n" + "═"*60)
    print("  📊  VALIDATION METRICS — Local ML Model")
    print("═"*60)

    all_cls = sorted(set(y_true_local + y_pred_local))
    t_idx   = [all_cls.index(t) for t in y_true_local]
    p_idx   = [all_cls.index(p) if p in all_cls else 0 for p in y_pred_local]

    acc  = accuracy_score(t_idx, p_idx)  * 100
    prec = precision_score(t_idx, p_idx, average='weighted', zero_division=0) * 100
    rec  = recall_score(t_idx,   p_idx,  average='weighted', zero_division=0) * 100
    f1   = f1_score(t_idx,       p_idx,  average='weighted', zero_division=0) * 100

    correct_total = sum(1 for t,p in zip(y_true_local,y_pred_local) if t==p)

    print(f"\n  ✅  Accuracy       : {acc:.2f} %")
    print(f"  ✅  Precision      : {prec:.2f} %")
    print(f"  ✅  Recall         : {rec:.2f} %")
    print(f"  ✅  F1-Score       : {f1:.2f} %")
    print(f"  🖼️   Images Tested  : {len(y_true_local)}")
    print(f"  ✅  Correct        : {correct_total}")
    print(f"  ❌  Wrong          : {len(y_true_local) - correct_total}")

    if use_groq and y_true_groq:
        print("\n" + "─"*60)
        print("  📊  VALIDATION METRICS — Groq Vision AI")
        print("─"*60)
        g_cls  = sorted(set(y_true_groq + y_pred_groq))
        gt_idx = [g_cls.index(t) for t in y_true_groq]
        gp_idx = [g_cls.index(p) if p in g_cls else 0 for p in y_pred_groq]
        g_acc  = accuracy_score(gt_idx, gp_idx)  * 100
        g_prec = precision_score(gt_idx, gp_idx, average='weighted', zero_division=0) * 100
        g_f1   = f1_score(gt_idx, gp_idx,        average='weighted', zero_division=0) * 100
        g_corr = sum(1 for t,p in zip(y_true_groq,y_pred_groq) if t==p)
        print(f"\n  ✅  Accuracy       : {g_acc:.2f} %")
        print(f"  ✅  Precision      : {g_prec:.2f} %")
        print(f"  ✅  F1-Score       : {g_f1:.2f} %")
        print(f"  ✅  Correct        : {g_corr}/{len(y_true_groq)}")

    # ── Per-class breakdown ──
    print("\n" + "─"*60)
    print("  Per-Class Validation Results (Local ML)")
    print("─"*60)
    class_stats = {}
    for t, p in zip(y_true_local, y_pred_local):
        if t not in class_stats:
            class_stats[t] = {"correct":0,"total":0}
        class_stats[t]["total"]   += 1
        if t == p:
            class_stats[t]["correct"] += 1

    for cls, st in sorted(class_stats.items()):
        pct  = st["correct"] / st["total"] * 100
        icon = "✅" if pct >= 80 else ("⚠️ " if pct >= 50 else "❌ ")
        bar  = "█" * int(pct/10) + "░" * (10-int(pct/10))
        print(f"  {icon} {cls:<28} {bar} {st['correct']}/{st['total']} ({pct:.0f}%)")

    # ── Save confusion matrix ──
    cls_list = sorted(set(y_true_local))
    cm_pred  = [p if p in cls_list else "other" for p in y_pred_local]
    cm       = confusion_matrix(y_true_local, cm_pred, labels=cls_list)
    fig_sz   = max(10, len(cls_list))
    plt.figure(figsize=(fig_sz, fig_sz-2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=cls_list, yticklabels=cls_list)
    plt.title(f'Validation Confusion Matrix  (Accuracy: {acc:.1f}%)', fontsize=13)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('validation_confusion_matrix.png', dpi=120)
    print(f"\n  📊  validation_confusion_matrix.png saved")
    print("═"*60 + "\n")

# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CropID Prediction Validator")
    parser.add_argument("--image",      type=str, help="Path to a single image")
    parser.add_argument("--true_label", type=str, help="True label for the image")
    parser.add_argument("--folder",     type=str, help="Folder of images with same class")
    parser.add_argument("--all",        action="store_true", help="Validate entire dataset")
    parser.add_argument("--samples",    type=int, default=5, help="Samples per class (default 5)")
    parser.add_argument("--groq",       action="store_true", help="Also test Groq Vision")
    args = parser.parse_args()

    if args.image and args.true_label:
        # Single image validation
        print(f"\n  Validating: {args.image}")
        print(f"  True label: {args.true_label}")
        r = validate_single(args.image, args.true_label, use_groq=True)
        print_single_result(r)

    elif args.folder and args.true_label:
        # Folder validation
        folder   = args.folder
        images   = [f for f in os.listdir(folder)
                   if f.lower().endswith(('.jpg','.jpeg','.png'))]
        correct  = 0
        print(f"\n  Validating {len(images)} images from: {folder}")
        print(f"  True label: {args.true_label}\n")
        for img_file in images:
            r = validate_single(os.path.join(folder, img_file),
                               args.true_label, use_groq=False)
            icon = "✅" if r['local_correct'] else "❌"
            print(f"  {icon} {img_file:<30} Pred: {r['local_pred']:<20} ({r['local_conf']}%)")
            if r['local_correct']: correct += 1
        acc = correct / len(images) * 100
        print(f"\n  Result: {correct}/{len(images)} correct ({acc:.1f}%)")

    elif args.all:
        # Full dataset validation
        validate_all(samples_per_class=args.samples, use_groq=args.groq)

    else:
        # Interactive mode — test a single image interactively
        print("\n" + "═"*60)
        print("  🌿  CropID — Interactive Prediction Validator")
        print("═"*60)
        print("\n  Usage examples:")
        print("  python validate.py --image dataset/wheat/img1.jpg --true_label wheat")
        print("  python validate.py --folder dataset/wheat --true_label wheat")
        print("  python validate.py --all --samples 10")
        print("  python validate.py --all --samples 10 --groq")
        print("\n  Running quick demo on 3 images per class...")
        validate_all(samples_per_class=3, use_groq=False)
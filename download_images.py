"""
CropID — Multi-Source Image Downloader
=======================================
Downloads from multiple sources to get 100+ images per class.
Usage: python download_images.py
"""

from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
import os, time, shutil

TARGET = 100  # images per class

crops = {
    "Cherry":               ["Cherry fruit tree", "Cherry crop farm", "Cherry plant leaves"],
    "Coffee-plant":         ["Coffee plant crop", "Coffee bean plant", "Coffee tree leaves"],
    "Cucumber":             ["Cucumber plant crop", "Cucumber vegetable farm", "Cucumber vine"],
    "Fox_nut(Makhana)":     ["Fox nut makhana plant", "Makhana crop field", "Euryale ferox plant"],
    "Lemon":                ["Lemon tree fruit", "Lemon plant crop", "Citrus lemon tree"],
    "Olive-tree":           ["Olive tree plant", "Olive fruit tree", "Olive grove farm"],
    "Pearl_millet(bajra)":  ["Pearl millet bajra crop", "Bajra field farm", "Pearl millet plant"],
    "Tobacco-plant":        ["Tobacco plant leaf", "Tobacco crop field", "Tobacco leaves farm"],
    "almond":               ["Almond tree plant", "Almond crop farm", "Almond nut tree"],
    "banana":               ["Banana plant tree", "Banana crop farm", "Banana bunch plant"],
    "cardamom":             ["Cardamom plant crop", "Cardamom spice plant", "Cardamom farm"],
    "chilli":               ["Chilli pepper plant", "Red chilli crop", "Chilli farm field"],
    "clove":                ["Clove plant tree", "Clove spice crop", "Clove farm"],
    "coconut":              ["Coconut tree palm", "Coconut palm farm", "Coconut plant"],
    "cotton":               ["Cotton plant crop", "Cotton boll field", "Cotton farm"],
    "gram":                 ["Gram chickpea crop", "Chickpea plant field", "Gram pulse farm"],
    "jowar":                ["Jowar sorghum crop", "Sorghum plant field", "Jowar farm"],
    "jute":                 ["Jute plant crop", "Jute fiber plant", "Jute farm field"],
    "maize":                ["Maize corn crop field", "Corn plant farm", "Maize cob plant"],
    "mustard-oil":          ["Mustard plant crop field", "Mustard flower farm", "Yellow mustard crop"],
    "papaya":               ["Papaya tree fruit", "Papaya plant farm", "Papaya crop"],
    "pineapple":            ["Pineapple plant crop", "Pineapple farm field", "Pineapple fruit plant"],
    "rice":                 ["Rice paddy crop field", "Rice plant farm", "Paddy field rice"],
    "soyabean":             ["Soybean plant crop", "Soya bean farm", "Soybean field"],
    "sugarcane":            ["Sugarcane crop field", "Sugarcane plant farm", "Sugar cane"],
    "sunflower":            ["Sunflower plant crop", "Sunflower farm field", "Sunflower bloom"],
    "tea":                  ["Tea plant crop", "Tea leaves farm", "Tea plantation"],
    "tomato":               ["Tomato plant crop", "Tomato farm field", "Tomato vine plant"],
    "vigna-radiati(Mung)":  ["Mung bean plant crop", "Moong dal plant", "Green gram plant"],
    "wheat":                ["Wheat crop field plant", "Wheat farm field", "Wheat grain crop"],
}

def count_images(folder):
    if not os.path.exists(folder): return 0
    return len([f for f in os.listdir(folder)
                if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))])

print("\n" + "="*55)
print("   🌿  CropID Multi-Source Image Downloader")
print("="*55 + "\n")

for folder, keywords in crops.items():
    save_path = f"dataset/{folder}"
    os.makedirs(save_path, exist_ok=True)

    existing = count_images(save_path)
    if existing >= TARGET:
        print(f"✅ {folder}: {existing} images — already enough, skipping")
        continue

    print(f"\n📥 {folder}: has {existing}, need {TARGET}")

    for keyword in keywords:
        current = count_images(save_path)
        if current >= TARGET:
            break

        needed = TARGET - current
        print(f"   🔍 Google: '{keyword}' (need {needed} more)...")
        try:
            GoogleImageCrawler(
                storage={"root_dir": save_path},
                feeder_threads=2,
                parser_threads=2,
                downloader_threads=4
            ).crawl(keyword=keyword, max_num=needed)
        except Exception as e:
            print(f"      ⚠️ Google error: {e}")
        time.sleep(1)

        current = count_images(save_path)
        if current >= TARGET:
            break

        needed = TARGET - current
        print(f"   🔍 Bing:   '{keyword}' (need {needed} more)...")
        try:
            BingImageCrawler(
                storage={"root_dir": save_path},
                feeder_threads=2,
                parser_threads=2,
                downloader_threads=4
            ).crawl(keyword=keyword, max_num=needed)
        except Exception as e:
            print(f"      ⚠️ Bing error: {e}")
        time.sleep(1)

    final = count_images(save_path)
    icon  = "✅" if final >= TARGET else ("⚠️ " if final >= 50 else "❌ ")
    print(f"   {icon} {folder}: {final} total images")

print("\n" + "="*55)
print("   ✅  Download complete!")
print("\n   Image counts:")
total = 0
for folder in sorted(crops.keys()):
    n    = count_images(f"dataset/{folder}")
    icon = "✅" if n >= TARGET else ("⚠️ " if n >= 50 else "❌ ")
    print(f"   {icon}  {folder:<30} {n}")
    total += n
print(f"\n   Total images: {total}")
print("\n   Now run: python train.py")
print("="*55 + "\n")
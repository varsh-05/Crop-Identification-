import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import os, io, base64, json, time, requests
from groq import Groq
from dotenv import load_dotenv

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(
    page_title="CropID — AI Crop Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

CROP_INFO = {
    "wheat": {
        "season": "Rabi (Oct–Mar)", "water": "450–650 mm", "soil": "Loamy / Clay loam",
        "price": "Rs 2,015/quintal", "diseases": "Rust, Smut, Blight",
        "emoji": "🌾", "color": "#d97706",
        "nutrition": {"Calories": "340 kcal", "Protein": "13g", "Carbs": "72g", "Fiber": "10g", "Fat": "2g"},
        "diet_uses": ["Chapati", "Bread", "Pasta", "Porridge", "Biscuits"],
        "health_benefits": "High in fibre, supports digestion, controls blood sugar, rich in B vitamins and iron.",
        "locations": [
            {"state": "Punjab", "lat": 31.1471, "lon": 75.3412},
            {"state": "Haryana", "lat": 29.0588, "lon": 76.0856},
            {"state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462},
            {"state": "Madhya Pradesh", "lat": 22.9734, "lon": 78.6569},
            {"state": "Rajasthan", "lat": 27.0238, "lon": 74.2179},
        ],
    },
    "rice": {
        "season": "Kharif (Jun–Nov)", "water": "1,100–2,000 mm", "soil": "Clay / Silty clay",
        "price": "Rs 2,183/quintal", "diseases": "Blast, Blight, Brown spot",
        "emoji": "🍚", "color": "#10b981",
        "nutrition": {"Calories": "365 kcal", "Protein": "7g", "Carbs": "80g", "Fiber": "1g", "Fat": "1g"},
        "diet_uses": ["Steamed Rice", "Idli", "Dosa", "Khichdi", "Rice flour"],
        "health_benefits": "Easy to digest, gluten-free, good energy source, low in fat.",
        "locations": [
            {"state": "West Bengal", "lat": 22.9868, "lon": 87.8550},
            {"state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462},
            {"state": "Punjab", "lat": 31.1471, "lon": 75.3412},
            {"state": "Andhra Pradesh", "lat": 15.9129, "lon": 79.7400},
            {"state": "Tamil Nadu", "lat": 11.1271, "lon": 78.6569},
        ],
    },
    "maize": {
        "season": "Kharif (Jun–Sep)", "water": "500–800 mm", "soil": "Well-drained loam",
        "price": "Rs 1,962/quintal", "diseases": "Stalk rot, Leaf blight, Smut",
        "emoji": "🌽", "color": "#f97316",
        "nutrition": {"Calories": "365 kcal", "Protein": "9g", "Carbs": "74g", "Fiber": "7g", "Fat": "4g"},
        "diet_uses": ["Corn flour", "Popcorn", "Cornmeal", "Makki roti", "Animal feed"],
        "health_benefits": "Rich in antioxidants, supports eye health, good source of energy and fibre.",
        "locations": [
            {"state": "Karnataka", "lat": 15.3173, "lon": 75.7139},
            {"state": "Andhra Pradesh", "lat": 15.9129, "lon": 79.7400},
            {"state": "Maharashtra", "lat": 19.7515, "lon": 75.7139},
            {"state": "Rajasthan", "lat": 27.0238, "lon": 74.2179},
            {"state": "Madhya Pradesh", "lat": 22.9734, "lon": 78.6569},
        ],
    },
    "mango": {
        "season": "Summer (Mar–Jun)", "water": "750–1,500 mm", "soil": "Loamy / Alluvial",
        "price": "Rs 2,000–8,000/quintal", "diseases": "Powdery mildew, Anthracnose, Mango malformation",
        "emoji": "🥭", "color": "#f59e0b",
        "nutrition": {"Calories": "60 kcal", "Protein": "0.8g", "Carbs": "15g", "Fiber": "1.6g", "Fat": "0.4g"},
        "diet_uses": ["Fresh fruit", "Mango juice", "Aamras", "Pickles (Aam ka achar)", "Mango lassi", "Chutney"],
        "health_benefits": "Rich in Vitamin C and A, boosts immunity, aids digestion, high in antioxidants, supports eye health.",
        "locations": [
            {"state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462},
            {"state": "Andhra Pradesh", "lat": 15.9129, "lon": 79.7400},
            {"state": "Karnataka", "lat": 15.3173, "lon": 75.7139},
            {"state": "Bihar", "lat": 25.0961, "lon": 85.3131},
            {"state": "Gujarat", "lat": 22.2587, "lon": 71.1924},
        ],
    },
    "banana": {
        "season": "Year-round", "water": "1,200–2,200 mm", "soil": "Rich loamy",
        "price": "Rs 1,500–2,500/quintal", "diseases": "Panama wilt, Sigatoka, Bunchy top",
        "emoji": "🍌", "color": "#eab308",
        "nutrition": {"Calories": "89 kcal", "Protein": "1g", "Carbs": "23g", "Fiber": "2.6g", "Fat": "0.3g"},
        "diet_uses": ["Eaten raw", "Smoothies", "Banana chips", "Baby food", "Halwa"],
        "health_benefits": "Rich in potassium, supports heart health, boosts energy, aids digestion.",
        "locations": [
            {"state": "Tamil Nadu", "lat": 11.1271, "lon": 78.6569},
            {"state": "Maharashtra", "lat": 19.7515, "lon": 75.7139},
            {"state": "Gujarat", "lat": 22.2587, "lon": 71.1924},
            {"state": "Andhra Pradesh", "lat": 15.9129, "lon": 79.7400},
            {"state": "Karnataka", "lat": 15.3173, "lon": 75.7139},
        ],
    },
    "tomato": {
        "season": "Oct–Feb", "water": "400–600 mm", "soil": "Sandy loam",
        "price": "Rs 800–2,000/quintal", "diseases": "Early blight, Fusarium wilt, Mosaic virus",
        "emoji": "🍅", "color": "#ef4444",
        "nutrition": {"Calories": "18 kcal", "Protein": "0.9g", "Carbs": "3.9g", "Fiber": "1.2g", "Fat": "0.2g"},
        "diet_uses": ["Salads", "Sauces", "Soups", "Ketchup", "Curries"],
        "health_benefits": "Rich in lycopene, supports heart health, high in Vitamin C, low calorie.",
        "locations": [
            {"state": "Maharashtra", "lat": 19.7515, "lon": 75.7139},
            {"state": "Karnataka", "lat": 15.3173, "lon": 75.7139},
            {"state": "Andhra Pradesh", "lat": 15.9129, "lon": 79.7400},
            {"state": "Tamil Nadu", "lat": 11.1271, "lon": 78.6569},
            {"state": "Odisha", "lat": 20.9517, "lon": 85.0985},
        ],
    },
    "cotton": {
        "season": "Kharif (May–Dec)", "water": "700–1,200 mm", "soil": "Black cotton soil",
        "price": "Rs 6,620/quintal", "diseases": "Bollworm, Fusarium wilt, Leaf curl",
        "emoji": "🌸", "color": "#8b5cf6",
        "nutrition": {"Calories": "506 kcal", "Protein": "23g", "Carbs": "21g", "Fiber": "16g", "Fat": "34g"},
        "diet_uses": ["Cottonseed oil (cooking)", "Cottonseed meal (animal feed)", "Fibre crop"],
        "health_benefits": "Cottonseed oil is used in cooking. Primarily a fibre and textile crop.",
        "locations": [
            {"state": "Gujarat", "lat": 22.2587, "lon": 71.1924},
            {"state": "Maharashtra", "lat": 19.7515, "lon": 75.7139},
            {"state": "Telangana", "lat": 18.1124, "lon": 79.0193},
            {"state": "Andhra Pradesh", "lat": 15.9129, "lon": 79.7400},
            {"state": "Punjab", "lat": 31.1471, "lon": 75.3412},
        ],
    },
    "sugarcane": {
        "season": "Year-round (12–18 months)", "water": "1,500–2,500 mm", "soil": "Deep loam",
        "price": "Rs 315/quintal", "diseases": "Red rot, Smut, Wilt",
        "emoji": "🎋", "color": "#84cc16",
        "nutrition": {"Calories": "269 kcal", "Protein": "0g", "Carbs": "73g", "Fiber": "0g", "Fat": "0g"},
        "diet_uses": ["Sugar", "Jaggery", "Molasses", "Sugarcane juice", "Rum"],
        "health_benefits": "Quick energy boost, natural electrolytes, aids liver function.",
        "locations": [
            {"state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462},
            {"state": "Maharashtra", "lat": 19.7515, "lon": 75.7139},
            {"state": "Karnataka", "lat": 15.3173, "lon": 75.7139},
            {"state": "Tamil Nadu", "lat": 11.1271, "lon": 78.6569},
            {"state": "Andhra Pradesh", "lat": 15.9129, "lon": 79.7400},
        ],
    },
    "tea": {
        "season": "Year-round", "water": "1,500–2,500 mm", "soil": "Acidic loam",
        "price": "Rs 200–500/kg", "diseases": "Blister blight, Red spider mite, Root rot",
        "emoji": "🍵", "color": "#059669",
        "nutrition": {"Calories": "1 kcal", "Protein": "0g", "Carbs": "0.2g", "Fiber": "0g", "Fat": "0g"},
        "diet_uses": ["Black tea", "Green tea", "Chai", "Herbal infusions", "Iced tea"],
        "health_benefits": "Rich in antioxidants, boosts alertness, supports heart health, anti-inflammatory.",
        "locations": [
            {"state": "Assam", "lat": 26.2006, "lon": 92.9376},
            {"state": "West Bengal", "lat": 22.9868, "lon": 87.8550},
            {"state": "Tamil Nadu", "lat": 11.1271, "lon": 78.6569},
            {"state": "Kerala", "lat": 10.8505, "lon": 76.2711},
            {"state": "Himachal Pradesh", "lat": 31.1048, "lon": 77.1734},
        ],
    },
    "coconut": {
        "season": "Year-round", "water": "1,000–2,000 mm", "soil": "Sandy loam / Laterite",
        "price": "Rs 25–40/nut", "diseases": "Bud rot, Root wilt, Eriophyid mite",
        "emoji": "🥥", "color": "#92400e",
        "nutrition": {"Calories": "354 kcal", "Protein": "3g", "Carbs": "15g", "Fiber": "9g", "Fat": "33g"},
        "diet_uses": ["Coconut oil", "Coconut milk", "Chutney", "Curries", "Coconut water"],
        "health_benefits": "Healthy fats, boosts immunity, supports brain function, hydrating.",
        "locations": [
            {"state": "Kerala", "lat": 10.8505, "lon": 76.2711},
            {"state": "Karnataka", "lat": 15.3173, "lon": 75.7139},
            {"state": "Tamil Nadu", "lat": 11.1271, "lon": 78.6569},
            {"state": "Andhra Pradesh", "lat": 15.9129, "lon": 79.7400},
            {"state": "Odisha", "lat": 20.9517, "lon": 85.0985},
        ],
    },
    "chilli": {
        "season": "Kharif & Rabi", "water": "600–1,250 mm", "soil": "Loam / Clay loam",
        "price": "Rs 5,000–15,000/quintal", "diseases": "Anthracnose, Mosaic virus, Damping-off",
        "emoji": "🌶️", "color": "#dc2626",
        "nutrition": {"Calories": "40 kcal", "Protein": "2g", "Carbs": "9g", "Fiber": "1.5g", "Fat": "0.4g"},
        "diet_uses": ["Spice in curries", "Chilli powder", "Pickles", "Sauces", "Chutneys"],
        "health_benefits": "Boosts metabolism, pain relief, rich in Vitamin C, anti-inflammatory.",
        "locations": [
            {"state": "Andhra Pradesh", "lat": 15.9129, "lon": 79.7400},
            {"state": "Telangana", "lat": 18.1124, "lon": 79.0193},
            {"state": "Karnataka", "lat": 15.3173, "lon": 75.7139},
            {"state": "Maharashtra", "lat": 19.7515, "lon": 75.7139},
            {"state": "Odisha", "lat": 20.9517, "lon": 85.0985},
        ],
    },
    "sunflower": {
        "season": "Kharif & Rabi", "water": "500–700 mm", "soil": "Well-drained loam",
        "price": "Rs 6,015/quintal", "diseases": "Downy mildew, Alternaria blight, Rust",
        "emoji": "🌻", "color": "#fbbf24",
        "nutrition": {"Calories": "584 kcal", "Protein": "21g", "Carbs": "20g", "Fiber": "9g", "Fat": "51g"},
        "diet_uses": ["Sunflower oil", "Sunflower seeds", "Snacks", "Salad toppings", "Butter"],
        "health_benefits": "Rich in Vitamin E, lowers cholesterol, anti-inflammatory, heart healthy.",
        "locations": [
            {"state": "Karnataka", "lat": 15.3173, "lon": 75.7139},
            {"state": "Andhra Pradesh", "lat": 15.9129, "lon": 79.7400},
            {"state": "Maharashtra", "lat": 19.7515, "lon": 75.7139},
            {"state": "Bihar", "lat": 25.0961, "lon": 85.3131},
            {"state": "Odisha", "lat": 20.9517, "lon": 85.0985},
        ],
    },
    "mustard": {
        "season": "Rabi (Oct–Mar)", "water": "250–400 mm", "soil": "Loam / Sandy loam",
        "price": "Rs 5,650/quintal", "diseases": "White rust, Alternaria blight, Sclerotinia rot",
        "emoji": "🌼", "color": "#d97706",
        "nutrition": {"Calories": "508 kcal", "Protein": "26g", "Carbs": "28g", "Fiber": "12g", "Fat": "36g"},
        "diet_uses": ["Mustard oil (cooking)", "Pickles", "Mustard sauce", "Tempering spice", "Mustard greens (saag)"],
        "health_benefits": "Omega-3 rich, anti-bacterial, boosts immunity, heart friendly oil.",
        "locations": [
            {"state": "Rajasthan", "lat": 27.0238, "lon": 74.2179},
            {"state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462},
            {"state": "Haryana", "lat": 29.0588, "lon": 76.0856},
            {"state": "Madhya Pradesh", "lat": 22.9734, "lon": 78.6569},
            {"state": "West Bengal", "lat": 22.9868, "lon": 87.8550},
        ],
    },
    "mustard-oil": {
        "season": "Rabi (Oct–Mar)", "water": "250–400 mm", "soil": "Loam / Sandy loam",
        "price": "Rs 5,650/quintal", "diseases": "White rust, Alternaria blight, Sclerotinia rot",
        "emoji": "🌼", "color": "#d97706",
        "nutrition": {"Calories": "884 kcal", "Protein": "0g", "Carbs": "0g", "Fiber": "0g", "Fat": "100g"},
        "diet_uses": ["Mustard oil (cooking)", "Pickles", "Mustard sauce", "Tempering spice"],
        "health_benefits": "Omega-3 rich, anti-bacterial, boosts immunity, heart friendly oil.",
        "locations": [
            {"state": "Rajasthan", "lat": 27.0238, "lon": 74.2179},
            {"state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462},
            {"state": "Haryana", "lat": 29.0588, "lon": 76.0856},
            {"state": "Madhya Pradesh", "lat": 22.9734, "lon": 78.6569},
            {"state": "West Bengal", "lat": 22.9868, "lon": 87.8550},
        ],
    },
    "soyabean": {
        "season": "Kharif (Jun–Oct)", "water": "450–700 mm", "soil": "Well-drained loam",
        "price": "Rs 4,600/quintal", "diseases": "Rust, Mosaic virus, Root rot",
        "emoji": "🫘", "color": "#65a30d",
        "nutrition": {"Calories": "446 kcal", "Protein": "36g", "Carbs": "30g", "Fiber": "9g", "Fat": "20g"},
        "diet_uses": ["Tofu", "Soy milk", "Soybean oil", "Tempeh", "Protein supplements"],
        "health_benefits": "Complete protein source, lowers cholesterol, rich in iron and calcium.",
        "locations": [
            {"state": "Madhya Pradesh", "lat": 22.9734, "lon": 78.6569},
            {"state": "Maharashtra", "lat": 19.7515, "lon": 75.7139},
            {"state": "Rajasthan", "lat": 27.0238, "lon": 74.2179},
            {"state": "Karnataka", "lat": 15.3173, "lon": 75.7139},
            {"state": "Telangana", "lat": 18.1124, "lon": 79.0193},
        ],
    },
    "jowar": {
        "season": "Kharif (Jun–Sep)", "water": "400–600 mm", "soil": "Medium deep black",
        "price": "Rs 3,180/quintal", "diseases": "Downy mildew, Anthracnose, Rust",
        "emoji": "🌿", "color": "#78716c",
        "nutrition": {"Calories": "329 kcal", "Protein": "11g", "Carbs": "72g", "Fiber": "6g", "Fat": "3g"},
        "diet_uses": ["Jowar roti", "Porridge", "Flour", "Animal feed", "Gluten-free baking"],
        "health_benefits": "Gluten-free, controls blood sugar, rich in antioxidants, high fibre.",
        "locations": [
            {"state": "Maharashtra", "lat": 19.7515, "lon": 75.7139},
            {"state": "Karnataka", "lat": 15.3173, "lon": 75.7139},
            {"state": "Andhra Pradesh", "lat": 15.9129, "lon": 79.7400},
            {"state": "Madhya Pradesh", "lat": 22.9734, "lon": 78.6569},
            {"state": "Rajasthan", "lat": 27.0238, "lon": 74.2179},
        ],
    },
    "jute": {
        "season": "Kharif (Mar–Jul)", "water": "1,000–2,000 mm", "soil": "Alluvial loam",
        "price": "Rs 5,050/quintal", "diseases": "Stem rot, Root knot, Anthracnose",
        "emoji": "🧵", "color": "#a16207",
        "nutrition": {"Calories": "35 kcal", "Protein": "3.6g", "Carbs": "6.4g", "Fiber": "2g", "Fat": "0.1g"},
        "diet_uses": ["Jute leaves cooked as greens", "Fibre crop primarily", "Jute leaf soup"],
        "health_benefits": "Jute leaves are rich in iron, calcium, and vitamins, used as a leafy vegetable.",
        "locations": [
            {"state": "West Bengal", "lat": 22.9868, "lon": 87.8550},
            {"state": "Bihar", "lat": 25.0961, "lon": 85.3131},
            {"state": "Assam", "lat": 26.2006, "lon": 92.9376},
            {"state": "Odisha", "lat": 20.9517, "lon": 85.0985},
            {"state": "Meghalaya", "lat": 25.4670, "lon": 91.3662},
        ],
    },
    "coffee": {
        "season": "Year-round", "water": "1,500–2,500 mm", "soil": "Laterite / Forest loam",
        "price": "Rs 200–400/kg", "diseases": "Leaf rust, Berry borer, Root rot",
        "emoji": "☕", "color": "#78350f",
        "nutrition": {"Calories": "2 kcal", "Protein": "0.3g", "Carbs": "0g", "Fiber": "0g", "Fat": "0g"},
        "diet_uses": ["Black coffee", "Espresso", "Cappuccino", "Coffee powder", "Flavouring"],
        "health_benefits": "Boosts alertness, rich in antioxidants, improves physical performance.",
        "locations": [
            {"state": "Karnataka", "lat": 15.3173, "lon": 75.7139},
            {"state": "Kerala", "lat": 10.8505, "lon": 76.2711},
            {"state": "Tamil Nadu", "lat": 11.1271, "lon": 78.6569},
        ],
    },
    "coffee-plant": {
        "season": "Year-round", "water": "1,500–2,500 mm", "soil": "Laterite / Forest loam",
        "price": "Rs 200–400/kg", "diseases": "Leaf rust, Berry borer, Root rot",
        "emoji": "☕", "color": "#78350f",
        "nutrition": {"Calories": "2 kcal", "Protein": "0.3g", "Carbs": "0g", "Fiber": "0g", "Fat": "0g"},
        "diet_uses": ["Black coffee", "Espresso", "Cappuccino", "Coffee powder", "Flavouring"],
        "health_benefits": "Boosts alertness, rich in antioxidants, improves physical performance.",
        "locations": [
            {"state": "Karnataka", "lat": 15.3173, "lon": 75.7139},
            {"state": "Kerala", "lat": 10.8505, "lon": 76.2711},
            {"state": "Tamil Nadu", "lat": 11.1271, "lon": 78.6569},
        ],
    },
    "gram": {
        "season": "Rabi (Oct–Mar)", "water": "300–400 mm", "soil": "Sandy loam / Loam",
        "price": "Rs 5,440/quintal", "diseases": "Wilt, Blight, Root rot",
        "emoji": "🫘", "color": "#713f12",
        "nutrition": {"Calories": "364 kcal", "Protein": "19g", "Carbs": "61g", "Fiber": "17g", "Fat": "6g"},
        "diet_uses": ["Dal", "Besan flour", "Chana masala", "Hummus", "Snacks"],
        "health_benefits": "High protein, controls blood sugar, supports digestion, rich in iron.",
        "locations": [
            {"state": "Madhya Pradesh", "lat": 22.9734, "lon": 78.6569},
            {"state": "Rajasthan", "lat": 27.0238, "lon": 74.2179},
            {"state": "Maharashtra", "lat": 19.7515, "lon": 75.7139},
            {"state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462},
            {"state": "Karnataka", "lat": 15.3173, "lon": 75.7139},
        ],
    },
    "lemon": {
        "season": "Year-round", "water": "750–1,200 mm", "soil": "Sandy loam",
        "price": "Rs 30–60/kg", "diseases": "Canker, Tristeza virus, Phytophthora",
        "emoji": "🍋", "color": "#ca8a04",
        "nutrition": {"Calories": "29 kcal", "Protein": "1g", "Carbs": "9g", "Fiber": "2.8g", "Fat": "0.3g"},
        "diet_uses": ["Lemonade", "Salad dressing", "Pickling", "Cooking", "Tea flavouring"],
        "health_benefits": "Very high in Vitamin C, boosts immunity, aids digestion, alkalises body.",
        "locations": [
            {"state": "Andhra Pradesh", "lat": 15.9129, "lon": 79.7400},
            {"state": "Maharashtra", "lat": 19.7515, "lon": 75.7139},
            {"state": "Tamil Nadu", "lat": 11.1271, "lon": 78.6569},
            {"state": "Gujarat", "lat": 22.2587, "lon": 71.1924},
        ],
    },
    "papaya": {
        "season": "Year-round", "water": "1,500–2,000 mm", "soil": "Sandy loam / Alluvial",
        "price": "Rs 15–40/kg", "diseases": "Ring spot virus, Damping off, Anthracnose",
        "emoji": "🧡", "color": "#ea580c",
        "nutrition": {"Calories": "43 kcal", "Protein": "0.5g", "Carbs": "11g", "Fiber": "1.7g", "Fat": "0.3g"},
        "diet_uses": ["Fresh fruit", "Juices", "Salads", "Jam", "Papaya halwa"],
        "health_benefits": "Rich in papain enzyme, aids digestion, high in Vitamin C and A.",
        "locations": [
            {"state": "Andhra Pradesh", "lat": 15.9129, "lon": 79.7400},
            {"state": "Karnataka", "lat": 15.3173, "lon": 75.7139},
            {"state": "Gujarat", "lat": 22.2587, "lon": 71.1924},
            {"state": "Maharashtra", "lat": 19.7515, "lon": 75.7139},
        ],
    },
    "pineapple": {
        "season": "Year-round", "water": "1,000–1,500 mm", "soil": "Sandy loam / Laterite",
        "price": "Rs 20–50/kg", "diseases": "Heart rot, Root rot, Mealybug wilt",
        "emoji": "🍍", "color": "#d97706",
        "nutrition": {"Calories": "50 kcal", "Protein": "0.5g", "Carbs": "13g", "Fiber": "1.4g", "Fat": "0.1g"},
        "diet_uses": ["Fresh fruit", "Juice", "Jam", "Canned fruit", "Smoothies"],
        "health_benefits": "Contains bromelain enzyme, boosts immunity, anti-inflammatory, aids digestion.",
        "locations": [
            {"state": "Assam", "lat": 26.2006, "lon": 92.9376},
            {"state": "West Bengal", "lat": 22.9868, "lon": 87.8550},
            {"state": "Kerala", "lat": 10.8505, "lon": 76.2711},
            {"state": "Karnataka", "lat": 15.3173, "lon": 75.7139},
        ],
    },
    "apple": {
        "season": "Summer-Autumn (Jul-Oct)", "water": "1,000-1,250 mm", "soil": "Well-drained loam",
        "price": "Rs 50-150/kg", "diseases": "Scab, Fire blight, Powdery mildew",
        "emoji": "🍎", "color": "#ef4444",
        "nutrition": {"Calories": "52 kcal", "Protein": "0.3g", "Carbs": "14g", "Fiber": "2.4g", "Fat": "0.2g"},
        "diet_uses": ["Fresh fruit", "Apple juice", "Apple cider", "Jam", "Dried apple"],
        "health_benefits": "High in fibre, rich in antioxidants, supports heart health, controls blood sugar.",
        "locations": [
            {"state": "Himachal Pradesh", "lat": 31.1048, "lon": 77.1734},
            {"state": "Jammu & Kashmir", "lat": 34.0836, "lon": 74.7973},
            {"state": "Uttarakhand", "lat": 30.0668, "lon": 79.0193},
        ],
    },
    "grapes": {
        "season": "Jan-May", "water": "700-1,200 mm", "soil": "Sandy loam / Loam",
        "price": "Rs 40-150/kg", "diseases": "Downy mildew, Powdery mildew, Anthracnose",
        "emoji": "🍇", "color": "#7c3aed",
        "nutrition": {"Calories": "67 kcal", "Protein": "0.6g", "Carbs": "17g", "Fiber": "0.9g", "Fat": "0.4g"},
        "diet_uses": ["Fresh fruit", "Raisins", "Wine", "Grape juice", "Jam"],
        "health_benefits": "Rich in resveratrol, supports heart health, anti-inflammatory, boosts immunity.",
        "locations": [
            {"state": "Maharashtra", "lat": 19.7515, "lon": 75.7139},
            {"state": "Karnataka", "lat": 15.3173, "lon": 75.7139},
            {"state": "Andhra Pradesh", "lat": 15.9129, "lon": 79.7400},
            {"state": "Tamil Nadu", "lat": 11.1271, "lon": 78.6569},
        ],
    },
    "onion": {
        "season": "Rabi (Oct-Mar)", "water": "350-550 mm", "soil": "Sandy loam / Loam",
        "price": "Rs 800-3,000/quintal", "diseases": "Purple blotch, Stemphylium blight, Basal rot",
        "emoji": "🧅", "color": "#d97706",
        "nutrition": {"Calories": "40 kcal", "Protein": "1.1g", "Carbs": "9g", "Fiber": "1.7g", "Fat": "0.1g"},
        "diet_uses": ["Curries", "Salads", "Pickles", "Soups", "Fried onion"],
        "health_benefits": "Rich in quercetin, anti-inflammatory, supports heart health, boosts immunity.",
        "locations": [
            {"state": "Maharashtra", "lat": 19.7515, "lon": 75.7139},
            {"state": "Karnataka", "lat": 15.3173, "lon": 75.7139},
            {"state": "Madhya Pradesh", "lat": 22.9734, "lon": 78.6569},
            {"state": "Gujarat", "lat": 22.2587, "lon": 71.1924},
            {"state": "Rajasthan", "lat": 27.0238, "lon": 74.2179},
        ],
    },
    "potato": {
        "season": "Rabi (Oct-Mar)", "water": "500-700 mm", "soil": "Sandy loam / Loam",
        "price": "Rs 800-2,000/quintal", "diseases": "Late blight, Early blight, Common scab",
        "emoji": "🥔", "color": "#a16207",
        "nutrition": {"Calories": "77 kcal", "Protein": "2g", "Carbs": "17g", "Fiber": "2.2g", "Fat": "0.1g"},
        "diet_uses": ["Curry", "Chips", "Fries", "Boiled", "Stuffed paratha"],
        "health_benefits": "Good source of Vitamin C and B6, rich in potassium, supports energy metabolism.",
        "locations": [
            {"state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462},
            {"state": "West Bengal", "lat": 22.9868, "lon": 87.8550},
            {"state": "Bihar", "lat": 25.0961, "lon": 85.3131},
            {"state": "Punjab", "lat": 31.1471, "lon": 75.3412},
            {"state": "Gujarat", "lat": 22.2587, "lon": 71.1924},
        ],
    },
}

# ── ALIAS MAP ─────────────────────────────────────────────────────────────────
CROP_ALIASES = {
    "mango tree": "mango", "alphonso": "mango", "dasheri": "mango", "langra": "mango",
    "corn": "maize", "soybean": "soyabean", "soy": "soyabean", "soybeans": "soyabean",
    "chilly": "chilli", "chili": "chilli", "pepper": "chilli",
    "coffee plant": "coffee", "coffee-plant": "coffee",
    "mustard oil": "mustard", "mustard plant": "mustard", "rapeseed": "mustard",
    "sugarcane plant": "sugarcane", "sugar cane": "sugarcane",
    "paddy": "rice", "paddy rice": "rice",
    "jowar sorghum": "jowar", "sorghum": "jowar",
    "groundnut": "gram", "chickpea": "gram", "bengal gram": "gram",
    "lemon tree": "lemon", "citrus": "lemon",
    "banana plant": "banana", "plantain": "banana",
    "coconut tree": "coconut", "coconut palm": "coconut",
    "tomatoes": "tomato", "onions": "onion", "potatoes": "potato",
    "grapes vine": "grapes", "grape": "grapes",
    "wheat plant": "wheat", "rice plant": "rice","corn plant": "maize",
"tea plant": "tea",
"coffee bean": "coffee",
"apple fruit": "apple",
"grape": "grapes",
"potato plant": "potato",
"tomato plant": "tomato",
"cotton plant": "cotton",
"banana tree": "banana",
"mango fruit": "mango",
}


def get_crop_info(crop_name):
    name = crop_name.lower().strip()
    if name in CROP_INFO:
        return CROP_INFO[name], name
    if name in CROP_ALIASES:
        key = CROP_ALIASES[name]
        return CROP_INFO[key], key
    for k in CROP_INFO:
        if k in name or name in k:
            return CROP_INFO[k], k
    for alias, key in CROP_ALIASES.items():
        if alias in name or name in alias:
            return CROP_INFO[key], key
    fallback = {
        "season": "Varies by region", "water": "Moderate", "soil": "Well-drained loam",
        "price": "Market dependent", "diseases": "Consult local expert",
        "emoji": "🌱", "color": "#10b981",
        "nutrition": {"Calories": "120 kcal", "Protein": "5g", "Carbs": "20g", "Fiber": "4g", "Fat": "2g"},
        "diet_uses": [f"Refer to local agricultural guide for {crop_name.title()}"],
        "health_benefits": f"No specific data found for {crop_name.title()}. Consult a nutritionist.",
        "locations": [{"state": "India", "lat": 20.5937, "lon": 78.9629}],
    }
    return fallback, crop_name


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
    html,body,.stApp{background:#f8fafc!important;font-family:'DM Sans',sans-serif;color:#1e293b;}
    .stApp>header{background:transparent!important;}
    section[data-testid="stSidebar"]{background:#f0fdf4!important;border-right:2px solid #bbf7d0;}
    .hero{background:linear-gradient(135deg,#dcfce7,#ede9fe);border:1px solid #bbf7d0;
          border-radius:20px;padding:2.5rem 3rem;margin-bottom:1.5rem;text-align:center;}
    .hero h1{font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;
             background:linear-gradient(135deg,#16a34a,#6366f1);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;}
    .hero p{color:#64748b;font-size:1.05rem;margin:.4rem 0 0;}
    .info-card{background:#fff;border:1px solid #e2e8f0;border-radius:16px;padding:1.4rem;
               margin:.5rem 0;box-shadow:0 1px 6px #0000000a;transition:transform .2s,box-shadow .2s;}
    .info-card:hover{transform:translateY(-2px);box-shadow:0 6px 20px #16a34a18;}
    .result-card{background:linear-gradient(135deg,#dcfce7,#ede9fe);border:2px solid #16a34a;
                 border-radius:20px;padding:2rem;text-align:center;margin:1rem 0;}
    .result-card h2{font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;color:#16a34a;margin:0;}
    .stat-pill{display:inline-block;background:#dcfce7;border:1px solid #86efac;border-radius:999px;
               padding:.3rem .85rem;font-size:.78rem;color:#15803d;margin:.2rem;font-weight:500;}
    .conf-bar-wrap{background:#e2e8f0;border-radius:999px;height:10px;overflow:hidden;margin:.35rem 0;}
    .conf-bar{height:100%;border-radius:999px;background:linear-gradient(90deg,#16a34a,#6366f1);}
    .top3-item{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:.75rem 1rem;
               margin:.35rem 0;display:flex;align-items:center;gap:.75rem;box-shadow:0 1px 4px #0000000a;}
    .top3-rank{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:#16a34a;width:2rem;}
    .top3-name{flex:1;font-weight:500;color:#1e293b;}
    .top3-pct{color:#6366f1;font-weight:700;font-family:'Syne',sans-serif;}
    .section-title{font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#1e293b;
                   border-left:4px solid #16a34a;padding-left:.75rem;margin:1.5rem 0 1rem;}
    .chat-user{background:#ede9fe;border:1px solid #c4b5fd;border-radius:16px 16px 4px 16px;
               padding:.75rem 1rem;margin:.4rem 0;color:#1e293b;}
    .chat-bot{background:#dcfce7;border:1px solid #86efac;border-radius:16px 16px 16px 4px;
              padding:.75rem 1rem;margin:.4rem 0;color:#1e293b;}
    .nut-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:.6rem;margin:.8rem 0;}
    .nut-box{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;padding:.8rem .4rem;text-align:center;}
    .nut-val{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#15803d;}
    .nut-lbl{font-size:.7rem;color:#64748b;margin-top:.2rem;}
    .use-pill{display:inline-block;background:#ede9fe;border:1px solid #c4b5fd;border-radius:999px;
              padding:.25rem .8rem;font-size:.8rem;color:#4f46e5;margin:.2rem;font-weight:500;}
    .benefit-box{background:#f0fdf4;border-left:4px solid #16a34a;border-radius:10px;
                 padding:.9rem 1rem;margin-top:.8rem;}
    #MainMenu,footer,.stDeployButton{display:none!important;}
    .block-container{padding-top:1.5rem!important;}
    .stButton>button{background:linear-gradient(135deg,#16a34a,#6366f1)!important;color:white!important;
                     border:none!important;border-radius:12px!important;font-family:'DM Sans',sans-serif!important;
                     font-weight:500!important;padding:.5rem 1.5rem!important;}
    .stButton>button:hover{opacity:.85!important;}
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not TF_AVAILABLE:
        return None, []
    try:
        model = tf.keras.models.load_model("crop_model.h5")
        with open("labels.txt") as f:
            labels = [l.strip() for l in f.readlines()]
        return model, labels
    except Exception:
        return None, []


@st.cache_resource
def get_groq():
    key = os.getenv("GROQ_API_KEY", "")
    return Groq(api_key=key) if key else None


def enhance_image(img):
    return ImageEnhance.Contrast(ImageEnhance.Sharpness(img).enhance(1.5)).enhance(1.2)


def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    return np.expand_dims(np.array(img) / 255.0, 0)


def predict_local(img, model, labels):
    preds = model.predict(preprocess(img), verbose=0)[0]
    top3_idx = preds.argsort()[-3:][::-1]
    return [(labels[i], float(preds[i]) * 100) for i in top3_idx]


def predict_groq(img, client):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        resp = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": (
                    'Identify the crop or plant in the image. '
                    'Reply ONLY with valid JSON: '
                    '{"crop":"mango","confidence":92,"alternatives":[{"crop":"papaya","confidence":5},{"crop":"banana","confidence":3}],"note":"one line observation"}. '
                    'Use common English crop names (e.g. mango, wheat, rice, tomato). '
                    'No markdown, no extra text.'
                )},
            ]}],
            max_tokens=250,
        )
        raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        top3 = [(data["crop"], data["confidence"])]
        for alt in data.get("alternatives", [])[:2]:
            top3.append((alt["crop"], alt["confidence"]))
        return top3, data.get("note", "")
    except Exception as e:
        return None, str(e)

def voice_button(text, key="v"):
    clean = text.replace("'", "").replace('"', "").replace("\n", " ")[:400]

    st.components.v1.html(f"""
    <button onclick="speak_{key}()" style="
        background:linear-gradient(135deg,#16a34a,#6366f1);
        color:white;
        border:none;
        border-radius:12px;
        padding:.45rem 1.2rem;
        cursor:pointer;
        font-size:.88rem;
        margin-top:.4rem;
        font-family:sans-serif;">
        🔊 Read Aloud
    </button>

    <script>
    function speak_{key}() {{

        window.speechSynthesis.cancel();

        const text = "{clean}";
        const utterance = new SpeechSynthesisUtterance(text);

        // English female voice settings
        utterance.lang = "en-GB";
        utterance.rate = 0.95;
        utterance.pitch = 1.2;
        utterance.volume = 1;

        // Load available voices
        let voices = window.speechSynthesis.getVoices();

        // Try selecting a female English voice
        let femaleVoice = voices.find(v =>
            v.name.includes("Female") ||
            v.name.includes("Samantha") ||
            v.name.includes("Victoria") ||
            v.name.includes("Google UK English Female") ||
            v.name.includes("Microsoft Zira")
        );

        // Fallback to any English voice
        if (!femaleVoice) {{
            femaleVoice = voices.find(v =>
                v.lang.includes("en")
            );
        }}

        if (femaleVoice) {{
            utterance.voice = femaleVoice;
        }}

        window.speechSynthesis.speak(utterance);
    }}

    // Ensure voices load properly
    window.speechSynthesis.onvoiceschanged = () => {{
        window.speechSynthesis.getVoices();
    }};
    </script>
    """, height=60)


def show_top3(top3):
    medals = ["🥇", "🥈", "🥉"]
    st.markdown('<div class="section-title">Top Predictions</div>', unsafe_allow_html=True)
    for i, (crop, conf) in enumerate(top3):
        pct = min(100, round(conf, 1))
        st.markdown(f"""
        <div class="top3-item">
            <span class="top3-rank">{medals[i]}</span>
            <span class="top3-name">{crop.title()}</span>
            <div style="flex:2"><div class="conf-bar-wrap">
                <div class="conf-bar" style="width:{pct}%"></div>
            </div></div>
            <span class="top3-pct">{pct}%</span>
        </div>""", unsafe_allow_html=True)


def show_crop_card(crop_name, info):
    st.markdown(f"""
    <div class="info-card" style="border-left:4px solid {info['color']}">
        <h3 style="font-family:'Syne',sans-serif;margin:0 0 1rem;font-size:1.4rem;">
            {info['emoji']} {crop_name.title()} — Crop Profile
        </h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.8rem;">
            <div><span style="color:#94a3b8;font-size:.78rem;">🌱 GROWING SEASON</span><br/>
                 <strong>{info['season']}</strong></div>
            <div><span style="color:#94a3b8;font-size:.78rem;">💧 WATER REQUIREMENT</span><br/>
                 <strong>{info['water']}</strong></div>
            <div><span style="color:#94a3b8;font-size:.78rem;">🌍 BEST SOIL TYPE</span><br/>
                 <strong>{info['soil']}</strong></div>
            <div><span style="color:#94a3b8;font-size:.78rem;">💰 MARKET PRICE (MSP)</span><br/>
                 <strong style="color:{info['color']}">{info['price']}</strong></div>
        </div>
        <div style="margin-top:.8rem;">
            <span style="color:#94a3b8;font-size:.78rem;">🐛 COMMON DISEASES</span><br/>
            <span style="color:#ef4444;">{info['diseases']}</span>
        </div>
    </div>""", unsafe_allow_html=True)


def show_diet_tab(crop_name, info):
    nut = info.get("nutrition", {})
    uses = info.get("diet_uses", [])
    benefits = info.get("health_benefits", "")

    st.markdown(f"""
    <div style="text-align:center;padding:1rem 0 .5rem">
        <span style="font-size:3rem">{info['emoji']}</span>
        <h2 style="font-family:'Syne',sans-serif;color:#15803d;margin:.3rem 0">
            {crop_name.title()} — Nutrition & Diet Report
        </h2>
        <p style="color:#64748b">Per 100g serving</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Nutritional Values (per 100g)</div>', unsafe_allow_html=True)
    nut_html = "".join(f"""
        <div class="nut-box">
            <div class="nut-val">{v}</div>
            <div class="nut-lbl">{k}</div>
        </div>""" for k, v in nut.items())
    st.markdown(f'<div class="nut-grid">{nut_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🍽️ Common Dietary Uses</div>', unsafe_allow_html=True)
    use_pills = "".join(f'<span class="use-pill">🍴 {u}</span>' for u in uses)
    st.markdown(f'<div style="line-height:2.6">{use_pills}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">💚 Health Benefits</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="benefit-box">
        <p style="margin:0;font-size:.95rem;line-height:1.7;color:#1e293b">{benefits}</p>
    </div>""", unsafe_allow_html=True)

    voice_text = f"{crop_name} nutritional information. {benefits} Common uses include {', '.join(uses)}."
    voice_button(voice_text, key=f"diet_{crop_name.replace(' ','_')}")


def show_india_map(crop_name, info):
    st.markdown('<div class="section-title">🗺️ Growing Regions in India</div>', unsafe_allow_html=True)
    try:
        import folium
        from streamlit_folium import st_folium
        m = folium.Map(location=[22.0, 80.0], zoom_start=4, tiles="CartoDB positron")
        for loc in info["locations"]:
            folium.CircleMarker(
                location=[loc["lat"], loc["lon"]], radius=30,
                color=info["color"], fill=True, fill_color=info["color"],
                fill_opacity=0.12, weight=1,
            ).add_to(m)
            folium.CircleMarker(
                location=[loc["lat"], loc["lon"]], radius=12,
                color=info["color"], fill=True, fill_color=info["color"],
                fill_opacity=0.75, weight=2,
                popup=folium.Popup(
                    f"<div style='font-family:sans-serif;padding:4px'>"
                    f"<b>{info['emoji']} {loc['state']}</b><br/>"
                    f"<span style='color:#666'>Major {crop_name.title()} region</span></div>",
                    max_width=200),
                tooltip=f"{info['emoji']} {loc['state']}",
            ).add_to(m)
            folium.Marker(
                location=[loc["lat"], loc["lon"]],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:15px;margin:-10px 0 0 -8px">{info["emoji"]}</div>',
                    icon_size=(28, 28), icon_anchor=(14, 14)),
            ).add_to(m)
        st_folium(m, width=None, height=460, returned_objects=[], use_container_width=True)
        pills = "".join(f'<span class="stat-pill">📍 {l["state"]}</span>' for l in info["locations"])
        st.markdown(f'<div style="margin-top:.6rem;line-height:2.4">{pills}</div>', unsafe_allow_html=True)
    except ImportError:
        pills = "".join(f'<span class="stat-pill">📍 {l["state"]}</span>' for l in info["locations"])
        st.markdown(f"""
        <div class="info-card" style="text-align:center;border:2px dashed #bbf7d0">
            <p style="color:#64748b;font-size:.85rem;margin:0 0 .8rem">
                Install folium for interactive map:<br/>
                <code>pip install folium streamlit-folium</code>
            </p>
            <div style="line-height:2.5">{pills}</div>
        </div>""", unsafe_allow_html=True)


def run_identification(img_proc, engine, groq_client, model, labels):
    """Run AI identification. Returns (top3, note, canonical_crop_name, info) or (None,msg,None,None)."""
    top3, note = None, ""
    if engine in ["Groq Vision (Primary)", "Both"] and groq_client:
        top3, note = predict_groq(img_proc, groq_client)
    if (top3 is None or engine == "Local Model") and model:
        top3 = predict_local(img_proc, model, labels)
    if top3:
        crop_name_raw = top3[0][0]
        info, canonical = get_crop_info(crop_name_raw)
        # Rebuild top3 with canonical names so Diet tab matches correctly
        return top3, note, canonical, info
    return None, note, None, None


def get_image_input():
    mode = st.radio("", ["📁 Upload File", "🔗 Image URL", "📷 Webcam"],
                    horizontal=True, label_visibility="collapsed")
    img, img_orig = None, None
    if mode == "📁 Upload File":
        f = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        if f:
            img = Image.open(f)
            img_orig = img.copy()
            st.image(img, caption="Uploaded Image", use_container_width=True)
    elif mode == "🔗 Image URL":
        url = st.text_input("Paste image URL", placeholder="https://example.com/crop.jpg")
        if url:
            try:
                r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                img = Image.open(io.BytesIO(r.content))
                img_orig = img.copy()
                st.image(img, caption="From URL", use_container_width=True)
            except Exception as e:
                st.error(f"Cannot load image: {e}")
    elif mode == "📷 Webcam":
        st.info("📷 Allow camera when prompted")
        cam = st.camera_input("Take a photo")
        if cam:
            img = Image.open(cam)
            img_orig = img.copy()
    return img, img_orig


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🌿 CropID")
        st.markdown("---")
        st.markdown("### 📤 Upload Mode")
        mode = st.radio("", ["Single Image", "Dual-Image Batch"], label_visibility="collapsed")
        st.markdown("---")
        st.markdown("### 🤖 AI Engine")
        engine = st.selectbox("", ["Groq Vision (Primary)"], label_visibility="collapsed")
        st.markdown("---")
        st.markdown("""
        <div style="font-size:.82rem;color:#374151;line-height:2">
        ✅ Upload / URL / Webcam<br/>
        ✅ Top-3 confidence scores<br/>
        ✅ Crop profile & diseases<br/>
        ✅ Nutrition & Diet report<br/>
        ✅ Interactive India map<br/>
        ✅ Voice readout<br/>
        ✅ AI expert chatbot<br/><br/>
        🔑 <a href="https://console.groq.com" target="_blank" style="color:#16a34a">Groq API key</a><br/>
        🗺️ Map: <code style="font-size:.7rem">pip install folium streamlit-folium</code>
        </div>""", unsafe_allow_html=True)
    return mode, engine


def init_session():
    defaults = {
        "last_crop": None,
        "last_info": None,
        # Each entry: (canonical_crop_name, info_dict, confidence_float)
        "identified_crops": [],
        "chat_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _add_crop_to_session(crop_name, info, conf):
    """Add an identified crop to session state without duplicates."""
    existing_names = [c[0] for c in st.session_state.identified_crops]
    if crop_name not in existing_names:
        st.session_state.identified_crops.append((crop_name, info, conf))
    st.session_state.last_crop = crop_name
    st.session_state.last_info = info


def main():
    inject_css()
    init_session()
    mode, engine = render_sidebar()

    st.markdown("""
    <div class="hero">
        <h1>🌿 CropID</h1>
        <p>AI Crop Identification · Diet & Nutrition Report · Interactive India Map · Voice Output</p>
    </div>""", unsafe_allow_html=True)

    model, labels = load_model()
    groq_client = get_groq()

    if not groq_client:
        st.warning("⚠️ Groq API key missing. Add GROQ_API_KEY to .env file.")
    if not TF_AVAILABLE or model is None:
        st.info("ℹ️ Using Groq Vision only — works perfectly!")

    tab1, tab2, tab3 = st.tabs(["🔍 Identify Crop", "🥗 Diet & Nutrition", "💬 Ask AI Expert"])

    # ════ TAB 1 — IDENTIFY ═══════════════════════════════════════════════════
    with tab1:
        if mode == "Single Image":
            col_up, col_res = st.columns([1, 1], gap="large")
            with col_up:
                st.markdown('<div class="section-title">Select Input Method</div>', unsafe_allow_html=True)
                img, img_orig = get_image_input()
                if img:
                    enhance = st.checkbox("✨ Auto-enhance image", value=True)
                    img_proc = enhance_image(img) if enhance else img
                    identify = st.button("🔍 Identify Crop", use_container_width=True)
                else:
                    identify = False
                    img_proc = None

            with col_res:
                if img and identify:
                    with st.spinner("Analysing your crop image..."):
                        time.sleep(0.2)
                        top3, note, crop_name, info = run_identification(
                            img_proc, engine, groq_client, model, labels
                        )

                    if top3:
                        conf = top3[0][1]
                        # ── KEY FIX: always save to session, regardless of input mode ──
                        _add_crop_to_session(crop_name, info, conf)

                        st.markdown(f"""
                        <div class="result-card">
                            <div style="font-size:3rem">{info['emoji']}</div>
                            <h2>{crop_name.title()}</h2>
                            <p style="color:#64748b;margin:.2rem 0 .5rem">{note or 'Identified by AI'}</p>
                            <span class="stat-pill">✅ {conf:.1f}% confidence</span>
                        </div>""", unsafe_allow_html=True)

                        voice_button(
                            f"The crop identified is {crop_name}. Confidence {conf:.0f} percent. "
                            f"Growing season is {info['season']}. Water requirement is {info['water']}.",
                            key="id"
                        )
                        show_top3(top3)
                        show_crop_card(crop_name, info)
                        show_india_map(crop_name, info)

                        if img_orig:
                            st.markdown('<div class="section-title">Image Comparison</div>', unsafe_allow_html=True)
                            c1, c2 = st.columns(2)
                            with c1:
                                st.image(img_orig, caption="Original", use_container_width=True)
                            with c2:
                                st.image(img_proc, caption="Enhanced", use_container_width=True)

                        st.success("✅ Done! Switch to the **🥗 Diet & Nutrition** tab to see the full report.")
                    else:
                        st.error("❌ Could not identify. Try a clearer image or check Groq API key.")

        else:
            # ── BATCH MODE ────────────────────────────────────────────────────
            st.markdown('<div class="section-title">Batch Upload</div>', unsafe_allow_html=True)
            uploads = st.file_uploader(
                "Upload multiple images", type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True, label_visibility="collapsed"
            )
            if uploads and st.button("🔍 Identify All", use_container_width=True):
                # Clear old batch results
                st.session_state.identified_crops = []
                results = []
                prog = st.progress(0, text="Identifying...")

                for i, f in enumerate(uploads):
                    img_raw = Image.open(f)
                    img_proc = enhance_image(img_raw)
                    top3, note, crop_name, info = run_identification(
                        img_proc, engine, groq_client, model, labels
                    )
                    if top3:
                        conf = top3[0][1]
                        results.append((f.name, crop_name, conf, Image.open(f)))
                        _add_crop_to_session(crop_name, info, conf)
                    prog.progress((i + 1) / len(uploads), text=f"{i + 1}/{len(uploads)} done")

                if results:
                    cols = st.columns(min(len(results), 3))
                    for i, (fname, crop, conf, thumb) in enumerate(results):
                        info_c, _ = get_crop_info(crop)
                        with cols[i % 3]:
                            st.image(thumb, use_container_width=True)
                            st.markdown(f"""
                            <div class="info-card" style="text-align:center">
                                <div style="font-size:1.5rem">{info_c['emoji']}</div>
                                <strong style="font-family:'Syne',sans-serif">{crop.title()}</strong>
                                <p style="color:#64748b;font-size:.85rem;margin:.2rem 0">{conf:.1f}%</p>
                                <p style="color:#94a3b8;font-size:.72rem">{fname}</p>
                            </div>""", unsafe_allow_html=True)

                    import pandas as pd
                    st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
                    st.dataframe(
                        pd.DataFrame(
                            [(n, c.title(), f"{p:.1f}%") for n, c, p, _ in results],
                            columns=["File", "Crop", "Confidence"]
                        ), use_container_width=True
                    )
                    st.success(
                        f"✅ {len(results)} crops identified! "
                        "Switch to **🥗 Diet & Nutrition** to view each crop's report."
                    )

    # ════ TAB 2 — DIET & NUTRITION ════════════════════════════════════════════
    with tab2:
        identified = st.session_state.identified_crops

        # Fallback: if somehow identified_crops is empty but last_crop is set
        if not identified and st.session_state.last_crop:
            identified = [(st.session_state.last_crop, st.session_state.last_info, 100.0)]

        if identified:
            if len(identified) == 1:
                crop_name, info, conf = identified[0]
                show_diet_tab(crop_name, info)
            else:
                st.markdown(
                    '<div class="section-title">Select a Crop to View its Diet Report</div>',
                    unsafe_allow_html=True
                )
                radio_labels = [
                    f"{info['emoji']}  {crop.title()}  ({conf:.0f}% confidence)"
                    for crop, info, conf in identified
                ]
                chosen_label = st.radio("Identified crops:", radio_labels, label_visibility="collapsed")
                chosen_idx = radio_labels.index(chosen_label)
                crop_name, info, conf = identified[chosen_idx]
                st.markdown("---")
                show_diet_tab(crop_name, info)
        else:
            st.markdown("""
            <div class="info-card" style="text-align:center;padding:3rem;border:2px dashed #bbf7d0">
                <div style="font-size:3rem">🥗</div>
                <h3 style="font-family:'Syne',sans-serif;color:#15803d;margin:.5rem 0">
                    No crop identified yet
                </h3>
                <p style="color:#64748b">
                    Go to the <b>🔍 Identify Crop</b> tab, upload an image and click Identify.<br/>
                    The full nutrition and diet report will appear here automatically!
                </p>
            </div>""", unsafe_allow_html=True)

    # ════ TAB 3 — AI CHAT ════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">Ask the Crop AI Expert</div>', unsafe_allow_html=True)
        st.markdown("Ask anything — diseases, harvest time, soil, market prices, diet, nutrition…")

        for role, msg in st.session_state.chat_history:
            cls = "chat-user" if role == "user" else "chat-bot"
            icon = "🧑" if role == "user" else "🤖"
            st.markdown(f'<div class="{cls}">{icon}&nbsp; {msg}</div>', unsafe_allow_html=True)

        user_q = st.chat_input("e.g. What are the health benefits of eating jowar?")
        if user_q:
            if groq_client:
                st.session_state.chat_history.append(("user", user_q))
                with st.spinner("Thinking..."):
                    resp = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": (
                                "You are an expert Indian agricultural and nutritional advisor. "
                                "Give practical, concise answers about Indian crops, farming, diseases, "
                                "MSP prices, soil, weather, nutrition and diet plans. "
                                "Keep responses under 150 words."
                            )},
                            *[{"role": r, "content": m} for r, m in st.session_state.chat_history],
                        ],
                        max_tokens=400,
                    )
                answer = resp.choices[0].message.content
                st.session_state.chat_history.append(("assistant", answer))
                voice_button(answer[:300], key=f"chat{len(st.session_state.chat_history)}")
                st.rerun()
            else:
                st.error("❌ Groq API key not set.")

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()


if __name__ == "__main__":
    main()
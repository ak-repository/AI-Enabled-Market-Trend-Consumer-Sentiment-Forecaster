import pandas as pd
import re
import torch
from transformers import pipeline

# 1. Load CSV File
df = pd.read_csv("generated_data_from_analysis/C_Reduceed_dataset.csv")


# 2. Clean Product Name Functions
def clean_product_name(text):
    text = str(text).lower()
    text = re.sub(r"\d+", " ", text)                  # Remove Numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)          # Remove Special Characters
    text = re.sub(r"\s+", " ", text).strip()          # Remove Extra Spaces
    return text

# product column ko hi clean kar diya
df["cleaned_product"] = df["product"].apply(clean_product_name)


# 3. Split Flipkart / Non-Flipkart
df_flipkart = df[df["source"].str.lower() == "flipkart"].copy()
df_non_flipkart = df[df["source"].str.lower() != "flipkart"].copy()


# 4. Zero-shot Classifier (GPU if available)
device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)


# 5. Category Labels
labels = [
    "Electricals_Power_Backup",
    "Home_Appliances",
    "Kitchen_Appliances",
    "Furniture",
    "Home_Storage_Organization",
    "Computers_Tablets",
    "Mobile_Accessories",
    "Wearables",
    "TV_Audio_Entertainment",
    "Networking_Devices",
    "Toys_Kids",
    "Gardening_Outdoor",
    "Kitchen_Dining",
    "Mens_Clothing",
    "Footwear",
    "Beauty_Personal_Care",
    "Security_Surveillance",
    "Office_Printer_Supplies",
    "Software",
    "Fashion_Accessories",
    "Home_Furnishings",
    "Sports_Fitness",
    "Grocery",
    "Tools",
    "Health_Care"
]


# 6. Deduplicate Flipkart Products
unique_products = (
    df_flipkart[["cleaned_product"]]
    .dropna()
    .drop_duplicates()
    .to_dict("records")
)


# 7. Rule-based Keyword Override
def keyword_override(name):

    if "mixer" in name or "grinder" in name or "juicer" in name or "kettle" in name or "cooker" in name or "microwave" in name or "otg" in name:
        return "Kitchen_Appliances"

    if "washing machine" in name or "refrigerator" in name or "fridge" in name or "cooler" in name or "fan" in name or "ac" in name:
        return "Home_Appliances"

    if "charger" in name or "cable" in name or "usb" in name or "type c" in name or "cover" in name or "case" in name or "earphone" in name:
        return "Mobile_Accessories"

    if "laptop" in name or "keyboard" in name or "mouse" in name or "monitor" in name or "intel" in name or "ryzen" in name:
        return "Computers_Tablets"

    if "tv" in name or "speaker" in name or "soundbar" in name or "bluetooth" in name or "home theatre" in name:
        return "TV_Audio_Entertainment"

    if "bed" in name or "sofa" in name or "chair" in name or "table" in name or "wardrobe" in name:
        return "Furniture"

    if "bedsheet" in name or "curtain" in name or "blanket" in name or "cushion" in name or "mat" in name:
        return "Home_Furnishings"

    if "shirt" in name or "jeans" in name or "t shirt" in name or "trouser" in name:
        return "Mens_Clothing"

    if "shoe" in name or "sandal" in name or "slipper" in name:
        return "Footwear"

    if "cream" in name or "shampoo" in name or "soap" in name or "facewash" in name or "hair oil" in name:
        return "Beauty_Personal_Care"

    if "printer" in name or "cartridge" in name or "toner" in name or "scanner" in name:
        return "Office_Printer_Supplies"

    if "cctv" in name or "camera" in name or "security" in name:
        return "Security_Surveillance"

    if "rice" in name or "atta" in name or "dal" in name or "oil" in name or "spice" in name:
        return "Grocery"

    if "cycle" in name or "gym" in name or "fitness" in name or "dumbbell" in name:
        return "Sports_Fitness"

    if "tool" in name or "cutter" in name or "drill" in name or "soldering" in name:
        return "Tools"

    return None


# 8. Predict Category (ONLY FLIPKART)
pred_map = {}
conf_map = {}   # ---- confidence store karne ke liye ----

batch_size = 16
for i in range(0, len(unique_products), batch_size):

    batch = unique_products[i:i + batch_size]
    texts = [item["cleaned_product"] for item in batch]

    results = classifier(texts, labels)

    if isinstance(results, dict):
        results = [results]

    for item, res in zip(batch, results):

        rule_cat = keyword_override(item["cleaned_product"])

        if rule_cat:
            final_cat = rule_cat
            confidence = 1.0
        else:
            final_cat = res["labels"][0]
            confidence = round(float(res["scores"][0]), 3)

        pred_map[item["cleaned_product"]] = final_cat
        conf_map[item["cleaned_product"]] = confidence


# 9. Fill Prediction in SAME category column
df_flipkart["category"] = df_flipkart["cleaned_product"].map(pred_map)
df_flipkart["category_confidence"] = df_flipkart["cleaned_product"].map(conf_map)


# 10. Combine Flipkart + Non-Flipkart
df_final = pd.concat([df_flipkart, df_non_flipkart], ignore_index=True)


# 11. Save Output
df_final.to_csv("generated_data_from_analysis/D_After_flipkart_category_predictions.csv", index=False)

print("Category prediction completed")

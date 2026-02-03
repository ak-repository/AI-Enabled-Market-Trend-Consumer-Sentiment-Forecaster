import requests
import pandas as pd
from datetime import datetime
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
from dotenv import load_dotenv

from notifications.notification import send_mail, send_slack_notifications
from external_api.sentiment_rapid_spike import rapid_sentiment_spike


tqdm.pandas()

# LOAD ENV
load_dotenv()

RAPIDAPI_KEY = os.getenv("RapidAPI_Key")
if not RAPIDAPI_KEY:
    raise ValueError("RapidAPI_Key not found in .env file")

# API CONFIG
HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"
}

SEARCH_URL = "https://real-time-amazon-data.p.rapidapi.com/search"
REVIEW_URL = "https://real-time-amazon-data.p.rapidapi.com/product-reviews"

COUNTRY = "US"
MAX_PRODUCTS_PER_KEYWORD = 5

# CATEGORY KEYWORDS
CATEGORY_KEYWORDS = {
    "Electricals_Power_Backup": ["inverter", "ups", "power backup"],
    "Home_Appliances": ["air conditioner", "refrigerator"],
    "Kitchen_Appliances": ["mixer", "microwave"],
    "Computers_Tablets": ["laptop", "tablet"],
    "Mobile_Accessories": ["charger", "power bank"],
    "Wearables": ["smartwatch"],
    "TV_Audio_Entertainment": ["smart tv", "speaker"]
}

# LOAD SENTIMENT MODEL
print("🤖 Loading sentiment model...")

MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

LABELS = ["negative", "neutral", "positive"]
NUMERIC_MAP = {"negative": -1, "neutral": 0, "positive": 1}


def get_sentiment(text):
    if not text or not text.strip():
        return pd.Series(["neutral", 0.0, 0])

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    label = LABELS[probs.argmax().item()]
    score = probs.max().item()
    numeric = NUMERIC_MAP[label]

    return pd.Series([label, score, numeric])


# AMAZON API FUNCTIONS
def search_products(query):
    params = {
        "query": query,
        "country": COUNTRY,
        "sort_by": "RELEVANCE"
    }
    response = requests.get(SEARCH_URL, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json().get("data", {}).get("products", [])


def fetch_reviews(asin):
    params = {
        "asin": asin,
        "country": COUNTRY,
        "sort_by": "TOP_REVIEWS"
    }
    response = requests.get(REVIEW_URL, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json().get("data", {}).get("reviews", [])


# MAIN PIPELINE
def amazon_api():
    try:
        all_rows = []

        for category, keywords in CATEGORY_KEYWORDS.items():
            print(f"🔍 Searching Amazon for: {category}")

            for keyword in keywords:
                products = search_products(keyword)

                for product in products[:MAX_PRODUCTS_PER_KEYWORD]:
                    asin = product.get("asin")
                    if not asin:
                        continue

                    reviews = fetch_reviews(asin)

                    for r in reviews:
                        all_rows.append({
                            "source": "Amazon",
                            "category_label": category,
                            "search_keyword": keyword,
                            "asin": asin,
                            "product_title": product.get("title", ""),
                            "brand": product.get("brand", ""),
                            "rating": r.get("rating"),
                            "review_title": r.get("review_title", ""),
                            "review_text": r.get("review_text", ""),
                            "review_date": r.get("review_date"),
                            "verified_purchase": r.get("verified_purchase"),
                            "created_date": datetime.utcnow()
                        })

                time.sleep(1)

        if not all_rows:
            print("⚠ No Amazon data fetched")
            return

        # CREATE DATAFRAME
        df = pd.DataFrame(all_rows)
        df.drop_duplicates(subset=["asin", "review_text"], inplace=True)

        df["combined_text"] = (
            df["review_title"].fillna("") + " " +
            df["review_text"].fillna("")
        ).str.slice(0, 500)

        # SENTIMENT ANALYSIS
        print("📊 Running sentiment analysis...")
        df[["sentiment_label", "sentiment_score", "sentiment_numeric"]] = (
            df["combined_text"].progress_apply(get_sentiment)
        )

        df.drop(columns=["combined_text"], inplace=True)

        # SAVE FINAL FILE
        os.makedirs("Final data", exist_ok=True)
        file_path = "Final data/amazon_review_trend_data.xlsx"
        df.to_excel(file_path, index=False)
        print("✅ Saved:", file_path)

        # SENTIMENT SPIKE
        result_df = rapid_sentiment_spike(file_path)

        # NOTIFICATIONS
        if result_df.empty:
            subject = "Amazon Review Alert"
            text = (
                "Amazon data extracted successfully.\n"
                "No major weekly sentiment spikes detected."
            )

            mail_sent = send_mail(text, subject)

        else:
            subject = "Amazon Review Sentiment Spike Detected"
            text = (
                "Amazon review data extracted successfully.\n"
                "Please find attached sentiment spike report."
            )

            mail_sent = send_mail(
                subject=subject,
                text=text,
                df=result_df
            )

        if not mail_sent:
            send_slack_notifications(
                text=f"{subject}\n{text}"
            )

    except Exception as e:
        print("❌ Amazon data extraction failed:", e)

        subject = "Amazon Data Extraction Failed"
        text = f"Pipeline failed due to error:\n{e}"

        mail_sent = send_mail(subject, text)

        if not mail_sent:
            send_slack_notifications(
                text=f"{subject}\n{text}"
            )


if __name__ == "__main__":
    print(amazon_api())

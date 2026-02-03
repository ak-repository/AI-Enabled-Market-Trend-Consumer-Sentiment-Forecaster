import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import notifications.notification as notification
from .sentiment_news_spike import news_sentiment_spike

tqdm.pandas()


# 1. LOAD ENVIRONMENT VARIABLES
load_dotenv()

API_KEY = os.getenv("News_API_KEY")
if not API_KEY:
    raise ValueError("NEWS_API_KEY not found in .env file")


# 2. API CONFIG
BASE_URL = "https://newsapi.org/v2/everything"
FROM_DATE = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
LANGUAGE = "en"
PAGE_SIZE = 50



# 3. CATEGORY KEYWORDS
CATEGORY_KEYWORDS = {
    "Electricals_Power_Backup": ["inverter", "ups", "power backup", "generator"],
    "Home_Appliances": ["air conditioner", "refrigerator", "washing machine"],
    "Kitchen_Appliances": ["mixer", "grinder", "microwave", "oven"],
    "Furniture": ["sofa", "bed", "table", "chair"],
    "Computers_Tablets": ["laptop", "tablet", "desktop"],
    "Mobile_Accessories": ["charger", "earphones", "power bank"],
    "Wearables": ["smartwatch", "fitness band"],
    "TV_Audio_Entertainment": ["smart tv", "speaker", "soundbar"],
    "Networking_Devices": ["router", "wifi modem"],
    "Beauty_Personal_Care": ["skincare", "beauty products"],
    "Software": ["software", "saas"]
}



# 4. LOAD SENTIMENT MODEL (RoBERTa – SAME AS REDDIT)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

LABELS = ["negative", "neutral", "positive"]
NUMERIC_MAP = {"negative": -1, "neutral": 0, "positive": 1}



# 5. FETCH NEWS
def fetch_news(query: str, category: str) -> list:
    params = {
        "q": query,
        "from": FROM_DATE,
        "language": LANGUAGE,
        "sortBy": "popularity",
        "pageSize": PAGE_SIZE,
        "apiKey": API_KEY
    }

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()

    data = response.json()
    articles = []

    for a in data.get("articles", []):
        articles.append({
            "source": a.get("source", {}).get("name"),
            "author": a.get("author"),
            "title": a.get("title"),
            "description": a.get("description"),
            "content": a.get("content"),
            "url": a.get("url"),
            "published_at": a.get("publishedAt"),
            "category": category,
            "query_used": query,
            "collected_at": datetime.utcnow()
        })

    return articles


# 6. SENTIMENT FUNCTION
def get_sentiment(text: str):
    if pd.isna(text) or not text.strip():
        return pd.Series(["neutral", 0.0, 0])

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    label = LABELS[probs.argmax().item()]
    score = probs.max().item()
    numeric = NUMERIC_MAP[label]

    return pd.Series([label, score, numeric])



# 7. MAIN PIPELINE
def get_news_data():
    try:
        all_articles = []

        for category, keywords in tqdm(CATEGORY_KEYWORDS.items(), desc="Categories"):
            for keyword in keywords:
                try:
                    all_articles.extend(fetch_news(keyword, category))
                except Exception as e:
                    print(f"Error fetching '{keyword}': {e}")

        if not all_articles:
            print("⚠ No news data fetched")
            return

        news_df = pd.DataFrame(all_articles)
        news_df.drop_duplicates(subset="url", inplace=True)

        # Combine text fields
        news_df["combined_text"] = (
            news_df["title"].fillna("") + ". " +
            news_df["description"].fillna("") + ". " +
            news_df["content"].fillna("")
        ).str.slice(0, 500)

        news_df[
            ["sentiment_label", "sentiment_score", "sentiment_numeric"]
        ] = news_df["combined_text"].progress_apply(get_sentiment)

        news_df.drop(columns=["combined_text"], inplace=True)

        # Save output
        os.makedirs("Final data", exist_ok=True)
        OUTPUT_FILE = "Final data/news_data_with_sentiment.csv"
        news_df.to_csv(OUTPUT_FILE, index=False)

        print(f"✅ Saved {len(news_df)} articles")

        
        # 8. SENTIMENT SPIKE ANALYSIS
        result_df = news_sentiment_spike(news_df)

        if result_df.empty:
            subject = "News Data Alert"
            text = (
                "News data extracted successfully.\n"
                "No major weekly sentiment spikes detected."
            )
            mail_sent = notification.send_mail(subject, text)
        else:
            subject = "Weekly News Sentiment Spike Report"
            text = (
                "News data extracted successfully.\n"
                "Please find the attached weekly sentiment spike report."
            )
            mail_sent = notification.send_mail(subject, text, df=result_df)

        if not mail_sent:
            notification.send_slack_notifications(f"{subject}\n{text}")

    except Exception as e:
        print("❌ Pipeline failed:", e)

        subject = "News Data Extraction Failed"
        text = f"Reason:\n{e}"

        mail_sent = notification.send_mail(subject, text)
        if not mail_sent:
            notification.send_slack_notifications(f"{subject}\n{text}")



# # 8. RUN
# if __name__ == "__main__":
#     get_news_data()







# import requests
# import pandas as pd
# from datetime import datetime, timedelta
# from tqdm import tqdm
# import os
# from dotenv import load_dotenv
# import notifications.notification as notification
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import notifications.notification as notification
# #from sentiment_news_spike import news_sentiment_spike
# from .sentiment_news_spike import news_sentiment_spike




# # 1. LOAD ENVIRONMENT VARIABLES
# load_dotenv()

# API_KEY = os.getenv("News_API_KEY")
# if not API_KEY:
#     raise ValueError("NEWS_API_KEY not found in .env file")


# #  2. API CONFIG
# BASE_URL = "https://newsapi.org/v2/everything"

# # i am collecting data form the last week so that i will have new data every week 
# FROM_DATE = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

# LANGUAGE = "en"
# PAGE_SIZE = 50


# #  3. CATEGORY KEYWORDS
# CATEGORY_KEYWORDS = {
#     "Electricals_Power_Backup": ["inverter", "ups", "power backup", "generator"],
#     "Home_Appliances": ["air conditioner", "refrigerator", "washing machine", "air cooler"],
#     "Kitchen_Appliances": ["mixer", "grinder", "microwave", "oven", "juicer"],
#     "Furniture": ["sofa", "bed", "table", "chair"],
#     "Home_Storage_Organization": ["storage box", "wardrobe", "organizer"],
#     "Computers_Tablets": ["laptop", "tablet", "desktop"],
#     "Mobile_Accessories": ["charger", "earphones", "power bank"],
#     "Wearables": ["smartwatch", "fitness band"],
#     "TV_Audio_Entertainment": ["smart tv", "speaker", "soundbar"],
#     "Networking_Devices": ["router", "wifi modem"],
#     "Toys_Kids": ["kids toys", "children games"],
#     "Gardening_Outdoor": ["gardening", "lawn tools"],
#     "Kitchen_Dining": ["cookware", "utensils"],
#     "Mens_Clothing": ["mens clothing", "mens fashion"],
#     "Footwear": ["shoes", "sneakers"],
#     "Beauty_Personal_Care": ["skincare", "beauty products"],
#     "Security_Surveillance": ["cctv", "security camera"],
#     "Office_Printer_Supplies": ["printer", "scanner"],
#     "Software": ["software", "saas"],
#     "Fashion_Accessories": ["handbag", "watch", "wallet"]
# }

# # # Load model for sentiment
# # MODEL_NAME = "ProsusAI/finbert"

# # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# # model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# # model.eval()

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model.to(device)



# # 4. FETCH NEWS FUNCTION
# def fetch_news(query: str, category: str) -> list:
#     params = {
#         "q": query,
#         "from": FROM_DATE,
#         "language": LANGUAGE,
#         "sortBy": "popularity",
#         "pageSize": PAGE_SIZE,
#         "apiKey": API_KEY
#     }

#     response = requests.get(BASE_URL, params=params)
#     response.raise_for_status()

#     data = response.json()
#     articles = []

#     for a in data.get("articles", []):
#         articles.append({
#             "source": a.get("source", {}).get("name"),
#             "author": a.get("author"),
#             "title": a.get("title"),
#             "description": a.get("description"),
#             "content": a.get("content"),
#             "url": a.get("url"),
#             "image_url": a.get("urlToImage"),
#             "published_at": a.get("publishedAt"),
#             "category": category,
#             "query_used": query,
#             "collected_at": datetime.utcnow()
#         })

#     return articles


# # SENTIMENT PREDICTION FUNCTION
# def get_sentiment(text):
    
#     MODEL_NAME = "ProsusAI/finbert"

#     # LOAD MODEL & TOKENIZER
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
#     model.eval()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
    
#     if pd.isna(text) or text.strip() == "":
#         return "Neutral"

#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=512
#     )

#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.nn.functional.softmax(outputs.logits, dim=1)
#         sentiment_idx = torch.argmax(probs).item()

#     label_map = {
#         0: "Negative",
#         1: "Neutral",
#         2: "Positive"
#     }

#     return label_map[sentiment_idx]


# # MAIN PIPELINE

# def get_news_data():
#     try:
#         all_articles = []

#         for category, keywords in tqdm(CATEGORY_KEYWORDS.items()):
#             for keyword in keywords:
#                 try:
#                     articles = fetch_news(keyword, category)
#                     all_articles.extend(articles)
#                 except Exception as e:
#                     print(f"Error fetching {keyword}: {e}")

#         if not all_articles:
#             print("⚠ No data fetched.")
#             return

        
#         # SAVE TO CSV
#         news_df = pd.DataFrame(all_articles)

#         # Remove duplicates based on URL
#         news_df.drop_duplicates(subset="url", inplace=True)

#         news_df.to_csv("Final data/news_data_with_sentiment.csv", index=False)

#         print(f"Saved {len(news_df)} articles to news_data_with_sentiment.csv")


#         # CONFIG       
#         OUTPUT_FILE = "Final data/news_data_with_sentiment.csv"
        

#         # LOAD DATA

#         # COMBINE TEXT FIELDS
#         news_df["combined_text"] = (
#             news_df["title"].fillna("") + ". " +
#             news_df["description"].fillna("") + ". " +
#             news_df["content"].fillna("")
#         )


#         # APPLY SENTIMENT MODEL
#         tqdm.pandas()
#         news_df["sentiment_label"] = news_df["combined_text"].progress_apply(get_sentiment)

        
#         # SAVE OUTPUT
#         news_df.drop(columns=["combined_text"], inplace=True)
        
#         if os.path.exists(OUTPUT_FILE):
#             # Append without header
#             news_df.to_csv(OUTPUT_FILE, mode="a", index=False, header=False)
#         else:
#             # Create file with header
#             news_df.to_csv(OUTPUT_FILE, index=False)

#         news_df="../OUTPUT_FILE"

#         result_df = news_sentiment_spike(news_df)

#     #     if result_df.empty:
#     #         notification.send_mail("News Data Alert", "News Data Extracted Successfully and No major weekly News sentiment spikes or trend shifts detected." )
#     #     else:
#     #         notification.send_mail("News Data Alert", "News Data Extracted Successfully and please find the attached report of sentiment spike and trend shift of this week", result_df )
#     #         print("✅ Sentiment analysis completed and saved successfully.")

        
#     #     # news_df.to_csv("news_data_categorized.csv", index=False)

#     #     # print(f"Saved {len(news_df)} articles to news_data_categorized.csv")
#     # except Exception as e:
#     #     notification.send_mail("News Data Alert", f"Failed to Extract News Data and the reason is: {e}")

          
#         if result_df.empty:
#             subject = "News Data Alert"
#             text = (
#                 "News Data Extracted Successfully.\n"
#                 "No major weekly News sentiment spikes or trend shifts detected."
#             )

#             mail_sent = notification.send_mail(subject, text)

#         else:
#             subject = "Extracted data from News API"
#             text = (
#                 "Successfully saved data from News API.\n"
#                 "Please find the attached report of sentiment spike and trend shift of this week."
#             )

#             mail_sent = notification.send_mail(
#                 subject=subject,
#                 text=text,
#                 df=result_df
#             )

#         # 🔁 EMAIL FAIL → SLACK
#         if not mail_sent:
#             notification.send_slack_notifications(
#                 text=f"{subject}\n{text}"
#             )

#     except Exception as e:
#         print("Failed to save news data:", e)

#         subject = "News Data Extraction Failed"
#         text = f"Failed to extract data from News api due to reason:\n{e}"

#         mail_sent = notification.send_mail(subject, text)

#         if not mail_sent:
#             notification.send_slack_notifications(
#                 text=f"{subject}\n{text}"
#             )


           
# Import Libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# 1. Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# 2. Cleaning Functions

def clean_lowercase(text):
    if isinstance(text, str):
        return text.lower()
    return text

def clean_punctuation(text):
    if isinstance(text, str):
        return re.sub(r"[^\w\s]", "", text)
    return text

def clean_stopwords(text):
    if isinstance(text, str):
        return " ".join([w for w in text.split() if w not in stop_words])
    return text

def clean_whitespaces(text):
    if isinstance(text, str):
        return re.sub(r"\s+", " ", text).strip()
    return text

def clean_text(text):
    text = clean_lowercase(text)
    text = clean_punctuation(text)
    text = clean_stopwords(text)
    text = clean_whitespaces(text)
    return text

# Load Files
df_flip = pd.read_csv("../Data/flipkart_product.csv", encoding="latin1", engine="python")
df_amazon = pd.read_excel("../Data/Amazon_DataSheet.xlsx")

def clean_text_flip(t):
    t = str(t)

    # 1. Remove non-ASCII garbled characters
    t = re.sub(r"[^\x00-\x7F]+", "", t)

    # 2. Remove all special characters except letters/numbers/space
    t = re.sub(r"[^a-zA-Z0-9\s]", "", t)

    # 3. Collapse multiple spaces into one
    t = re.sub(r"\s+", " ", t)

    return t.strip()

# Apply Flipkart-specific cleaning
df_flip = df_flip.applymap(lambda x: clean_text_flip(x) if isinstance(x, str) else x)


# Standardize Flipkart Columns

df_flip = df_flip.rename(columns={
    "ProductName": "product",
    "Review": "review_title",
    "Summary": "review_text",
    "Rate": "rating"
})

df_flip["source"] = "flipkart"
df_flip["review_date"] = ""        # Flipkart has no date
df_flip["sentiment_label"] = ""    # Flipkart has no sentiment
df_flip["category"] = ""           # Flipkart has no category


# Standardize Amazon Columns

df_amazon = df_amazon.rename(columns={
    "Product Name": "product",
    "Comment": "review_text",
    "Star Rating": "rating",
    "Date of Review": "review_date",
    "Category": "category",
    "Sentiment": "sentiment_label"  # Important update
})

df_amazon["source"] = "amazon"
df_amazon["review_title"] = ""     # Amazon title not used / not available


# Keep only Required Columns

required = [
    "source",
    "product",
    "review_text",
    "review_title",
    "rating",
    "category",
    "review_date",
    "sentiment_label"
]

df_flip = df_flip[required]
df_amazon = df_amazon[required]


# Combine both

df = pd.concat([df_flip, df_amazon], ignore_index=True)

# Clean review_text
df["cleaned_text"] = df["review_text"].astype(str).apply(clean_text)

# Clean Product name
df["product"] = df["product"].astype(str).apply(clean_text)

# Add Sentiment_score Placeholder
df["sentiment_score"] = ""

# Save Output
df.to_csv("generated_data_from_analysis/B_Merge_cleaned_data.csv", index=False)

print("Final dataset saved as Combined_cleaned_data.csv (Amazon sentiment included)")

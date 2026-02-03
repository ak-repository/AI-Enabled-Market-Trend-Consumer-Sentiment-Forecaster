# 1. Import required libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# 2. Download and load stopwords
nltk.download("stopwords")
stop_words=set(stopwords.words("english"))

# 3. Load CSV file
df = pd.read_csv("../Data/flipkart_product.csv", encoding="latin1", engine="python")

# 4. Remove duplicates rows
df=df.drop_duplicates()

# 5. Convert everything in lowercase
for col in df.select_dtypes(include=["object"]).columns:
    df[col]=df[col].astype(str).str.lower()

# 6. Remove punctuation
def remove_punctuation(text):
    if isinstance(text,str):
        return re.sub(r"[^\w\s]"," ",text)
    return text


for col in df.select_dtypes(include=["object"]).columns:
    df[col]=df[col].apply(remove_punctuation)

# 7.Remove stopwords
def remove_stopwords(text):
    if isinstance(text,str):
        return " ".join([word for word in text.split() if word not in stop_words])
    return text

for col in df.select_dtypes(include=["object"]).columns:
    df[col]=df[col].apply(remove_stopwords)


# 8. Remove extra spaces
for col in df.select_dtypes(include=["object"]).columns:
    df[col]=df[col].str.replace(r"\s+"," ",regex=True).str.strip()


# 9. Save the cleaned CSV
df.to_csv("generated_data_from_analysis/A_flipkart_product_data_after_cleaning.csv",index=False)
print("Data cleaning completed successfully")

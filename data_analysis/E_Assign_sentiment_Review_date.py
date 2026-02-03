import pandas as pd
import numpy as np

# 1. Load Data
df=pd.read_csv("generated_data_from_analysis/D_After_flipkart_category_predictions.csv")

# 2. Clean Rating Column (VERY IMPORTANT)
#    Converts "4.0", "5", "", NaN → numeric safely

df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# 3. Rating-Based sentiment for flipkart
def rating_sentiment(row):
    if str(row['source']).lower() == 'flipkart':
        rating = row['rating']

        if pd.isna(rating):
            return "unknown"
        elif rating in [1, 2]:
            return "Negative"
        elif rating == 3:
            return "Neutral"
        elif rating in [4, 5]:
            return "Positive"
        else:
            return "unknown"
    else:
        return row['sentiment_label']   # Keep previous Value

df['sentiment_label'] = df.apply(rating_sentiment, axis=1)

# 4. Add Random review date (2021-01-01 to 2025-01-01)
start_date=pd.to_datetime("2021-01-01")
end_date=pd.to_datetime("2025-01-01")

mask = df['review_date'].isna() | (df['review_date'].astype(str).str.strip() == "")

random_dates=pd.to_datetime(
    np.random.randint(
        start_date.value // 10**9,
        end_date.value // 10**9,
        size=mask.sum()
    ),
    unit='s'
)

df.loc[mask,'review_date']=random_dates.strftime("%Y-%m-%d")

# 5. Save File
df.to_csv("generated_data_from_analysis/E_After_Assign_sentiment_Review_date.csv",index=False)
print("Rating-based sentiment analysis completed")



import pandas as pd

df=pd.read_csv("generated_data_from_analysis/B_Merge_cleaned_data.csv")

# Keep only 30% of the data
KEEP_RATIO = 0.3

df_reduce=(
    df.groupby(['product','rating'],group_keys=False) #grouping product and rating
    .apply(lambda x:x.sample(n=max(1,int(len(x)*KEEP_RATIO)),random_state=42))
    .reset_index(drop=True) #Clean Index
)

print("Original rows:",len(df))
print("Reduced rows:",len(df_reduce))

df_reduce.to_csv("generated_data_from_analysis/C_Reduceed_dataset.csv",index=False,encoding="utf-8")
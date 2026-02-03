import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# CONFIG
INPUT_FILE = "generated_data_from_analysis/E_After_Assign_sentiment_Review_date.csv"
OUTPUT_FILE = "generated_data_from_analysis/F_category_wise_lda_output.csv"

NUM_TOPICS_PER_CATEGORY = 5
MIN_DOCS_PER_CATEGORY = 20
MAX_DF = 0.8
MIN_DF = 5
TOP_WORDS = 10
RANDOM_STATE = 42


# Custom Stopwords
CUSTOM_STOPWORDS = set([
    "good", "bad", "excellent", "poor", "amazing", "nice", "worst", "best",
    "love", "hate", "perfect", "terrible", "awesome", "waste",
    "money", "worth", "value", "price",
    "product", "products", "quality", "buy", "purchase",
    "using", "use", "used", "really", "very", "highly",
    "recommend", "recommended", "work", "works", "working", "flipkart"
])


# Text Cleaning for LDA
def clean_for_lda(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in CUSTOM_STOPWORDS and len(t) > 2]
    return " ".join(tokens)


# Load Data
df = pd.read_csv(INPUT_FILE)

df = df.dropna(subset=["cleaned_text", "category"])
df = df[df["cleaned_text"].str.strip() != ""]

df["lda_text"] = df["cleaned_text"].apply(clean_for_lda)

print("Total records:", len(df))


# Helper: Extract Top Topic Words
def get_topic_words(model, feature_names, top_n):
    topic_map = {}
    for idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-top_n:][::-1]
        words = [feature_names[i] for i in top_indices]
        topic_map[idx] = ", ".join(words)
    return topic_map


# Store Results
final_results = []


# Category-wise LDA
for category, df_cat in df.groupby("category"):

    if len(df_cat) < MIN_DOCS_PER_CATEGORY:
        print(f"Skipping '{category}' (only {len(df_cat)} reviews)")
        continue

    print(f"\nRunning LDA for category: {category} ({len(df_cat)} reviews)")


    # vectorizer = CountVectorizer(
    #     stop_words="english",
    #     max_df=MAX_DF,
    #     min_df=MIN_DF,
    #     ngram_range=(1, 2)
    # )


    dynamic_min_df = 2 if len(df_cat) < 100 else MIN_DF

    vectorizer = CountVectorizer(
        stop_words="english",
        max_df=MAX_DF,
        min_df=dynamic_min_df,
        ngram_range=(1, 2)
    )

    # ✅ MINIMUM CHANGE FIX (crash protection)
    try:
        doc_term_matrix = vectorizer.fit_transform(df_cat["lda_text"])
    except ValueError:
        print(f"⚠️ Skipping '{category}' (no terms after pruning)")
        continue

    if doc_term_matrix.shape[1] < NUM_TOPICS_PER_CATEGORY:
        print(f"Skipping '{category}' (not enough unique terms)")
        continue

    lda = LatentDirichletAllocation(
        n_components=NUM_TOPICS_PER_CATEGORY,
        random_state=RANDOM_STATE,
        learning_method="batch",
        max_iter=20
    )

    lda.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topic_word_map = get_topic_words(lda, feature_names, TOP_WORDS)

    print(f"\nTopics for category: {category}")
    for topic_id, keywords in topic_word_map.items():
        print(f"  Topic {topic_id}: {keywords}")

    topic_dist = lda.transform(doc_term_matrix)

    df_cat = df_cat.copy()
    df_cat["lda_topic"] = topic_dist.argmax(axis=1)
    df_cat["lda_topic_confidence"] = topic_dist.max(axis=1)
    df_cat["lda_topic_keywords"] = df_cat["lda_topic"].map(topic_word_map)

    final_results.append(df_cat)


# Combine & Save
if final_results:
    final_df = pd.concat(final_results, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved results to: {OUTPUT_FILE}")
else:
    print("No category had enough data for LDA.")

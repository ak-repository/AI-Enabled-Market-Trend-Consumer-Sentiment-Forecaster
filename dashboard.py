import streamlit as st
import pandas as pd
import plotly.express as px

# RAG + AI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai
from google.genai import types
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# for scheduler
import schedule
import threading
import time
from notifications import notification
from external_api import News_api
from external_api import Reddit_api
from external_api import Rapid_api

def run_scheduler():
    
    # call function every 10 seconds 
    #schedule.every(10).seconds.do(notifications.testing_function)
    #schedule.every(10).minutes.do(notifications.testing_function)
    #schedule.every().day.at("16:35").do(News_api.get_news_data)
    #schedule.every().day.at("18:26").do(Redit_api.get_reddit_data)

    # schedule.every(10).sunday.at("02:00").do(News_api.get_news_data)
    # schedule.every(10).sunday.at("03:00").do(Reddit_api.reddit_api)
    # schedule.every(10).sunday.at("03:00").do(Rapid_api.amazon_api)


    schedule.every().tuesday.at("09:37").do(News_api.get_news_data)
    schedule.every().tuesday.at("10:00").do(Reddit_api.reddit_api)
    # schedule.every().sunday.at("20:50").do(Rapid_api.get_amazon_review_data)

    
    
    # schedule.every(10).monday.at("02:00").do(notifications.testing_function)
    
    while True:
        schedule.run_pending()
        time.sleep(5)
    
threading.Thread(target=run_scheduler, daemon=True).start()

#__________________________________________________________________________


if __name__ == "__main__":

    # Page configuration
    st.set_page_config(
        page_title="AI-Enabled Market Trend & Consumer Sentiment Forecaster",
        layout="wide"
    )

    # Dashboard Title
    st.title("AI-Enabled Market Trend & Consumer Sentiment Dashboard")

    # Description
    st.markdown(
        "Consumer Sentiment, topic trends, and social insights from reviews, news, and Reddit data."
    )

    @st.cache_data
    def load_data():
        reviews=pd.read_csv("Final data/category_wise_lda_output_with_topic_labels.csv")

        reddit=pd.read_excel("Final data/reddit_category_trend_data.xlsx")

        news=pd.read_csv("Final data/news_data_with_sentiment.csv")

        if "review_date" in reviews.columns:
            reviews["review_date"]=pd.to_datetime(
                reviews["review_date"], errors="coerce"  # Convert invalid date into NaT
            )


        if "created_date" in reddit.columns:
            reddit["created_date"]=pd.to_datetime(
                reddit["created_date"], errors="coerce"  # Convert invalid date into NaT
            )


        if "published_at" in news.columns:
            news["published_at"]=pd.to_datetime(
                news["published_at"], errors="coerce"  # Convert invalid date into NaT
            )
        
        return reviews, reddit, news


    reviews_df, reddit_df, news_df = load_data()


    # load vector database

    @st.cache_resource
    def load_vector_db():
        # load embedding model

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # load faiss index
        vector_db = FAISS.load_local(
            "consumer_sentiment_faiss",
            embeddings,
            allow_dangerous_deserialization= True
        )
        return vector_db


    vector_db = load_vector_db()

    # load gemini
    @st.cache_resource
    def load_gemini_client():
        return genai.Client(api_key=os.getenv("Gemini_API_KEY"))

    gemini_client = load_gemini_client()

    # load Groq
    @st.cache_resource
    def load_groq_ai():
        return Groq(api_key=os.getenv("GROQ_API_KEY"),)

    groq_client = load_groq_ai()

    def generate_ai_response(prompt):
        """
        Primary  : Gemini
        Fallback : Groq
        """

        # Gemini
        try:
            gemini_response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2
                ),
            )

            # Validate Gemini response
            if gemini_response and gemini_response.text:
                print("Gemini response used")
                return gemini_response.text

            # Gemini returned empty text
            raise Exception("Gemini returned empty response")

        except Exception as gemini_error:
            print("Gemini failed → Switching to Groq")
            print("Reason:", gemini_error)


        # Groq
        try:
            groq_response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user",
                    "content": prompt}
                ],
                temperature=0.2
            )

            # Validate Groq response
            if groq_response and groq_response.choices:
                print("Groq response used")
                return groq_response.choices[0].message.content

            raise Exception("Groq returned empty response")

        except Exception as groq_error:
            print("Groq failed")
            print("Reason:", groq_error)

            # Final safe message for user
            return "AI service is temporarily unavailable. Please try again later."




    main_col, right_sidebar =st.columns([3,1])


    with main_col:
    # sidebar filters

        st.sidebar.header("Filters")

        source_filter = st.sidebar.multiselect(
            "Select Source",
            options=reviews_df["source"].unique(),
            default=reviews_df["source"].unique()
        )


        category_filter = st.sidebar.multiselect(
            "Select Category",
            options=reviews_df["category"].unique(),
            default=reviews_df["category"].unique()
        )

        filtered_reviews = reviews_df[
            (reviews_df["source"].isin(source_filter))&
            (reviews_df["category"].isin(category_filter))
        ]


        # KPI Metrics

        st.subheader("Key Metrics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Reviews", len(filtered_reviews))
        col2.metric("Positive %",round((filtered_reviews["sentiment_label"]=="Positive").mean()*100,1))
        col3.metric("Negative %",round((filtered_reviews["sentiment_label"]=="Negative").mean()*100,1))
        col4.metric("Neutral %",round((filtered_reviews["sentiment_label"]=="Neutral").mean()*100,1))

        # Sentiment distribution

        col1, col2 = st.columns(2)

        with col1:
            sentiment_dist = filtered_reviews["sentiment_label"].value_counts().reset_index()
            sentiment_dist.columns=["Sentiment", "Count"]

            fig = px.pie(
                sentiment_dist,
                names="Sentiment",
                values="Count",
                title="Overall Sentiment Distribution",
                hole=0.4
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            category_sentiment = (
                filtered_reviews.groupby(["category", "sentiment_label"]).size().reset_index(name="count")
            )

            fig = px.bar(
                category_sentiment,
                x="category",
                y="count",
                color="sentiment_label",
                title="Category-wise Sentiment Comparision",
                barmode="group"
            )

            st.plotly_chart(fig,use_container_width=True)


        # Sentiment trend over time

        st.subheader("Sentiment Trend Over Time")

        sentiment_trand= (
            filtered_reviews.groupby([pd.Grouper(key="review_date", freq="W"), "sentiment_label"])
            .size()
            .reset_index(name="count")
        )

        fig_trend = px.line(
            sentiment_trand,
            x="review_date",
            y="count",
            color = "sentiment_label",
            title="Weekly Sentiment Trend"
        )

        st.plotly_chart(fig_trend, use_container_width=True)

        # category trend over time

        st.subheader("Category Trend Over Time (Product Demand)")

        category_trend=(
            filtered_reviews.groupby([pd.Grouper(key="review_date", freq="M"), "category"])
            .size()
            .reset_index(name="count")
        )

        fig_category_trend=px.line(
            category_trend,
            x="review_date",
            y="count",
            color="category",
            title="Monthly Category Demand Trend"
        )

        st.plotly_chart(fig_category_trend,use_container_width=True)

        # category vs sentiment
        # Category-wise Sentiment Distribution

        st.subheader("Category-wise Sentiment Distribution")

        cat_sent=(
            filtered_reviews
            .groupby(["category","sentiment_label"])
            .size()
            .reset_index(name="count")
        )

        fig_cat=px.bar(
            cat_sent,
            x="category",
            y="count",
            color="sentiment_label",
            barmode="group"
        )

        st.plotly_chart(fig_cat, use_container_width=True)


        # Topic distribution

        st.subheader("Topic Insights")

        topic_dist=(
            filtered_reviews["topic_label"]
            .value_counts()
            .reset_index()
        )

        topic_dist.columns=["Topic", "Count"]

        fig_topic = px.bar(
            topic_dist,
            x="Topic",
            y="Count",
            title="Topic Distribution"
        )

        st.plotly_chart(fig_topic, use_container_width=True)


        # Reddit Sentiment

        st.subheader("Reddit Sentiment Overview")

        reddit_sent = (
            reddit_df
            .groupby("sentiment_label")
            .size()
            .reset_index(name="count")
        )

        fig_sentiment = px.pie(
            reddit_sent,
            names="sentiment_label",
            values="count",
            title="Reddit Sentiment Distribution"
        )

        st.plotly_chart(fig_sentiment, use_container_width=True)



        # Reddit category trend

        st.subheader("Reddit Category Popularity")

        reddit_trend=(
            reddit_df
            .groupby("category_label")
            .size()
            .reset_index(name="Mentions")
            .sort_values("Mentions", ascending=False)
        )

        fig_reddit = px.bar(
            reddit_trend,
            x="category_label",
            y="Mentions",
            title="Trending categories on Reddit"
        )

        st.plotly_chart(fig_reddit,use_container_width=True)


        # news sentiment

        st.subheader("News Sentiment Overview")

        news_sent=(
            news_df
            .groupby("sentiment_label")
            .size()
            .reset_index(name="count")
        )

        fig_sentiment = px.pie(
            news_sent,
            names="sentiment_label",
            values="count",
            title="News Sentiment distribution"
        )

        st.plotly_chart(fig_sentiment,use_container_width=True)

        # News category trend

        st.subheader("News Category Popularity")

        news_trend=(
            news_df
            .groupby("category")
            .size()
            .reset_index(name="Articles")
            .sort_values("Articles",ascending=False)
        )

        fig_news = px.bar(
            news_trend,
            x="category",
            y="Articles",
            title="Trainding categories on News"
        )

        st.plotly_chart(fig_news,use_container_width=True)



        # Cross Platform Sentiment Share

        review_sentiment = (
            reviews_df.groupby("sentiment_label")
            .size()
            .reset_index(name="Review")
            .rename(columns={"sentiment_label": "sentiment"})
        )

        reddit_sentiment = (
            reddit_df.groupby("sentiment_label")
            .size()
            .reset_index(name="Reddit")
            .rename(columns={"sentiment_label": "sentiment"})
        )

        news_sentiment = (
            news_df.groupby("sentiment_label")
            .size()
            .reset_index(name="News")
            .rename(columns={"sentiment_label": "sentiment"})
        )

        merged = review_sentiment.merge(reddit_sentiment, on="sentiment", how="outer")
        merged = merged.merge(news_sentiment, on="sentiment", how="outer")
        merged = merged.fillna(0)


        st.subheader("Sentiment Count Across Platforms")

        fig = px.bar(
            merged,
            x="sentiment",                      
            y=["Review", "Reddit", "News"], 
            title="Sentiment Distribution (Number of Posts)",
            text_auto=True
        )

        fig.update_layout(
            barmode="group",           
            xaxis_title="Sentiment",
            yaxis_title="Number of Posts",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)



        # Cross Platform Category Share

        review_counts = (
            reviews_df.groupby("category")
            .size()
            .reset_index(name="Review")
        )

        reddit_counts = (
            reddit_df.groupby("category_label")
            .size()
            .reset_index(name="Reddit")
            .rename(columns={"category_label": "category"})
        )

        news_counts = (
            news_df.groupby("category")
            .size()
            .reset_index(name="News")
        )

        # Merge all
        merged = review_counts.merge(reddit_counts, on="category", how="outer")
        merged = merged.merge(news_counts, on="category", how="outer")
        merged = merged.fillna(0)

        # Convert to percentage
        merged["total"] = merged["Review"] + merged["Reddit"] + merged["News"]

        merged["Review %"] = (merged["Review"] / merged["total"]) * 100
        merged["Reddit %"] = (merged["Reddit"] / merged["total"]) * 100
        merged["News %"]   = (merged["News"] / merged["total"]) * 100

        st.subheader("Cross-Source Category Comparison")

        fig = px.bar(
            merged,
            y="category",
            x=["Review %", "Reddit %", "News %"],
            orientation="h",
            title="Category Presence Across Platforms (Percentage Distribution)",
            text_auto=".1f"
        )

        fig.update_layout(
            barmode="stack",
            height=650,
            xaxis_title="Value (%)",
            yaxis_title="Category",
            legend_title="Platform",
            margin=dict(l=40, r=20, t=60, b=40)
        )

        fig.update_traces(
            hovertemplate="%{x:.1f}%<extra></extra>"
        )

        st.plotly_chart(fig, use_container_width=True)



        # Trending on all plateform

        st.subheader("Cross-Source Category Camparision")

        # Review category count

        review_cat=(
            reviews_df
            .groupby("category")
            .size()
            .reset_index(name="Review Mentions")
        )

        # Reddit category count

        reddit_cat=(
            reddit_df
            .groupby("category_label")
            .size()
            .reset_index(name="Reddit Mentions")
            .rename(columns={"category_label":"category"})
        )

        # News category data

        news_cat=(
            news_df
            .groupby("category")
            .size()
            .reset_index(name="News Mentions")
        )

        # Merge all

        category_compare = review_cat\
            .merge(reddit_cat, on="category", how="outer")\
            .merge(news_cat, on="category", how="outer")\
            .fillna(0)


        fig_compare = px.bar(
            category_compare,
            x="category",
            y=["Review Mentions", "Reddit Mentions", "News Mentions"],
            title="Category Presence Across Reviews, Reddit and News",
            barmode="group"
        )

        st.plotly_chart(fig_compare, use_container_width=True)
        st.dataframe(category_compare)


    with right_sidebar:
        st.markdown("### 🤖  AI Insight Panel")    
        st.caption("Ask Question using reviews, news, & Reddit data")
        
        user_query= st.text_area(
            "Your Question",
            height=140
        )
        
        ask_btn = st.button("Get Insight", use_container_width=True)
        
        if ask_btn and user_query:
            with st.spinner("Analyzing Market Intelligence..."):
            
                results = vector_db.similarity_search(user_query, k=10)
                retrived_docs = [r.page_content for r in results]
                
                
                prompt=f"""

                You are a market intelligence analyst
                
                using only the information from the provided context
                
                give response based on the question
                
                do not use bullet points, headings, or sections
                do not add external knowledge
                
                Context:
                {retrived_docs}
                
                Question:
                {user_query}
                
                Answer:
            """   
                
                final_answer = generate_ai_response(prompt) 
                # response = client.models.generate_content(
                # model="gemini-2.5-flash",
                # contents=prompt,
                # config=types.GenerateContentConfig(
                #         thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
                #         temperature=0.2
                #             ),
                # )
            
            st.success("Insight Generated")
            st.write(final_answer)

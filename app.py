import streamlit as st
import requests
import joblib
from bs4 import BeautifulSoup
import re

st.set_page_config(layout="wide")
st.title("📰 Fake News Detector")

# ——————————————————————————————————————————————
# Load model & vectorizer once
model = joblib.load("src/model.pkl")
vectorizer = joblib.load("src/vectorizer.pkl")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(text.split())

# 1) GNews API headlines
def fetch_headlines_gnews(topic):
    api_key = "0452f06fa6883fac204fb9b13a552861"
    url = "https://gnews.io/api/v4/top-headlines"
    params = {"token": api_key, "lang": "en", "topic": topic, "max": 10}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return []
    
    articles = []
    for article in r.json().get("articles", []):
        if article.get("title") and article.get("url"):
            articles.append({
                "title": article["title"],
                "url": article["url"],
                "image": article.get("image", None)  # Capture the image URL if available
            })
    return articles

# 2) Manual text prediction
# 3) URL extraction
def extract_article_text(url):
    try:
        r = requests.get(url, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        paras = soup.find_all("p")
        return " ".join([p.get_text() for p in paras])
    except:
        return None

# ——————————————————————————————————————————————
# TOP ROW: three input columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1️⃣ Live Headlines")
    topic = st.selectbox("Topic", ["world", "nation", "business", "technology", "entertainment"])
    if st.button("Fetch Headlines", key="fetch"):
        st.session_state["headlines"] = fetch_headlines_gnews(topic)
        st.session_state["option_selected"] = "headlines"
    else:
        st.session_state["headlines"] = fetch_headlines_gnews('technology')  # Default to technology headlines
        st.session_state["option_selected"] = "headlines"

with col2:
    st.subheader("2️⃣ Manual Text")
    st.session_state["user_text"] = st.text_area("Enter text", key="manual")
    if st.button("Predict Text", key="predict_manual"):
        txt = st.session_state["user_text"]
        if txt:
            vect = vectorizer.transform([preprocess_text(txt)])
            st.session_state["manual_pred"] = model.predict(vect)[0]
            st.session_state["option_selected"] = "manual"

with col3:
    st.subheader("3️⃣ From URL")
    st.session_state["url"] = st.text_input("Article URL", key="url_input")
    if st.button("Fetch & Predict", key="predict_url"):
        art = extract_article_text(st.session_state["url"])
        if art and len(art) > 50:
            vect = vectorizer.transform([preprocess_text(art)])
            st.session_state["url_pred"] = model.predict(vect)[0]
            st.session_state["url_text"] = art
            st.session_state["option_selected"] = "url"
        else:
            st.session_state["url_pred"] = None
            st.session_state["url_text"] = None
            st.session_state["option_selected"] = "url"

# ——————————————————————————————————————————————
# BOTTOM ROW: Display results based on the selected option
if "option_selected" in st.session_state:
    option = st.session_state["option_selected"]

    if option == "headlines":
        st.subheader("Headlines Predictions")
        if "headlines" in st.session_state:
            for article in st.session_state["headlines"]:
                title = article["title"]
                url = article["url"]
                image_url = article.get("image", None)

                # fetch full article text
                article_text = extract_article_text(url) or ""
                paragraphs = article_text.split("\n")
                preview_side = "\n".join(paragraphs[:1])
                preview_below = "\n".join(paragraphs[1:3])

                # Predict
                vect = vectorizer.transform([preprocess_text(article_text)])
                pred = model.predict(vect)[0]
                prediction_label = '🟢 Real' if pred == 1 else '🔴 Fake'

                with st.container():
                    # Headline and prediction in same row
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"<h3 style='font-size:22px; margin-bottom: 0;'>{title}</h3>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<p style='font-size:16px; font-weight:bold; text-align:right;'>Prediction: {prediction_label}</p>", unsafe_allow_html=True)

                    center_col = st.columns([1, 3, 1])[1]
                    with center_col:
                        if image_url:
                            st.image(image_url, use_container_width=True)

                    # Preview text
                    st.markdown(f"<p style='font-size:17px'>{preview_side}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:16px'>{preview_below}… <a href='{url}'>Read more</a></p>", unsafe_allow_html=True)

                    st.markdown("---")

                
    elif option == "manual":
        st.subheader("Manual Text Result")
        if "manual_pred" in st.session_state:
            label = "🟢 Real" if st.session_state["manual_pred"] == 1 else "🔴 Fake"
            st.success(label)
        else:
            st.write("Enter text and click Predict Text")
    
    elif option == "url":
        st.subheader("URL Article Result")
        if "url_pred" in st.session_state:
            if st.session_state["url_pred"] is not None:
                label = "🟢 Real" if st.session_state["url_pred"] == 1 else "🔴 Fake"
                st.success(label)

                st.markdown("**Extracted Article Preview:**")
                preview = "\n".join(st.session_state["url_text"].split("\n")[:3])

                # Two-column layout for image and preview
                col1, col2 = st.columns([1, 2])  # Image smaller than text

                with col1:
                    if "url_image" in st.session_state and st.session_state["url_image"]:
                        st.image(st.session_state["url_image"], use_container_width=True)

                with col2:
                    st.markdown(f"<p style='font-size:16px'>{preview}…</p>", unsafe_allow_html=True)

            else:
                st.error("Could not extract or predict from URL.")
        else:
            st.write("Enter URL and click Fetch & Predict")

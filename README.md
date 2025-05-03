# Fake news Detection

A model which detects the news is fake or not trained on fake and true news datasets using logistic regression and deployed via Streamlit.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run notebook and save the model: `using jupiter Notebook`
3. Load model: `python src/model.py`
4. Run the app: `streamlit run app.py`

## Features
- Text cleaning and preprocessing
- TF-IDF vectorization
- Logistic Regression classifier
- Detecting live news by news api.
- scraping article from url by using beautifulsoap and detecting fake or real.
- Streamlit UI for predictions

## Sample Output
Input: "Congress MP's 'no one saw surgical strikes' remark gets BJP's 'go to Pak' reply"
Output: Real

## Dataset
dataset: [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=True.csv](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=True.csv)
"""

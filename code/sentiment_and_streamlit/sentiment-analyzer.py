import streamlit as st
import pandas as pd
import numpy as np
import requests
from plotly.offline import iplot
import plotly.graph_objs as go
import plotly.express as px
from pandas.io.json import json_normalize
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score
wordnet_lemmatizer = WordNetLemmatizer()

loaded_model = pickle.load(open('models.p', 'rb'))
vectorizer = pickle.load(open('vector.pickel', 'rb'))

#test_feature = vectorizer.transform(['Meat Week Day 3: Tummy hurts every night'])
#model.predict(test_feature)


fig = go.Figure()
st.write("""
# Twitter Sentiment Analysis✌

""")

st.set_option('deprecation.showfileUploaderEncoding', False)
st.sidebar.header('User Input(s)')
st.sidebar.subheader('Single Tweet Analysis')
single_review = st.sidebar.text_input('Enter single review below:')
st.sidebar.subheader('Multiple Tweet Analysis')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
count_positive = 0
count_negative = 0
count_neutral = 0
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    for i in range(input_df.shape[0]):
        #url = 'https://aisentimentsanalyzer.herokuapp.com/classify/?text='+str(input_df.iloc[i])
        #r = requests.get(url)
        result = r.json()["text_sentiment"]
        if result=='positive':
            count_positive+=1
        elif result=='negative':
            count_negative+=1
        else:
            count_neutral+=1 

    x = ["Positive", "Negative", "Neutral"]
    y = [count_positive, count_negative, count_neutral]

    if count_positive>count_negative:
        st.write("""# 😃""")
    elif count_negative>count_positive:
        st.write("""# 😔""")
    else:
        st.write("""# 😶""")
        
    layout = go.Layout(
        title = 'Multiple Reviews Analysis',
        xaxis = dict(title = 'Category'),
        yaxis = dict(title = 'Number of reviews'),)
    
    fig.update_layout(dict1 = layout, overwrite = True)
    fig.add_trace(go.Bar(name = 'Multi Reviews', x = x, y = y))
    st.plotly_chart(fig, use_container_width=True)

elif single_review:
    #url = 'https://aisentimentsanalyzer.herokuapp.com/classify/?text='+single_review
    #r = requests.get(url)
    test_feature = vectorizer.transform([single_review])
    
    result = loaded_model['model'].predict(test_feature)
    if result=='positive':
        st.write("""# Positive Tweet 😃""")
    elif result=='negative':
        st.write("""# Negative Tweet 😔""")
    else:
        st.write("""# Neutral Tweet 😶""")

else:
    st.write("# ⬅ Enter user input from the sidebar to see the nature of the review.")



import streamlit as st
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
#from wordcloud import WordCloud
import plotly.graph_objects as go

st.set_page_config(
    page_title="Home",
    page_icon="",
)

st.header("Spam Detective: Unveiling the Power Ensemble of Classifiers for SMS & Email Filtering")

st.subheader("Abstract:")
st.markdown("""
            Focusing on crafting an efficient email/SMS spam classifier, this project explores diverse vectorization techniques and ensemble modeling. Textual data is transformed into vectors using Bag of Words, TF-IDF, and Word2Vec methods. TF-IDF emerges as the most effective, offering superior accuracy and precision. Multinomial Naive Bayes, Random Forest, and ExtraTrees stand out among various classification algorithms.

Employing Voting and Stacking ensemble techniques enhances predictive accuracy, with the Voting Classifier achieving an outstanding 98% accuracy and 100% precision in identifying spam. Deployment via Streamlit ensures cloud-based accessibility, showcasing the project's emphasis on comparative analysis. Optimizing vectorization and ensemble methods, this project culminates in the deployment of a highly accurate spam classification system, underscoring the significance of meticulous technique selection.
            """)

st.subheader("Data:")
st.write("https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")

df = pd.read_csv('spam.csv')
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True, axis=1)
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.drop_duplicates(inplace = True)
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

st.subheader("Descriptive Statistics: ")
st.write('1) Clearly we can see the data is imbalance by the below chart , here 13% is spam data and rest is ham.')
fig1 = px.pie(df['target'].value_counts(),values=df['target'].value_counts(),names=['ham','spam'])
st.plotly_chart(fig1)

df['num_character'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

st.write('2) The below plots are showing the counts of Spam and Ham messages where 0 denotes Ham and 1 denotes'
         'Spam, and we can see the count of character_word_sentences are less for spam messages.')
fig2 = px.histogram(df, x='num_character', color='target',
                   barmode='overlay')
# Update layout and display the plot
fig2.update_layout(bargap=0.1)
st.plotly_chart(fig2)

fig3 = px.histogram(df, x='num_words', color='target',
                   barmode='overlay')
# Update layout and display the plot
fig3.update_layout(bargap=0.1)
st.plotly_chart(fig3)

fig4 = px.histogram(df, x='num_sentences', color='target',
                   barmode='overlay')
# Update layout and display the plot
fig4.update_layout(bargap=0.1)
st.plotly_chart(fig4)

correlation_matrix = df.corr()
annotations = []
for i, row in enumerate(correlation_matrix.values):
    for j, value in enumerate(row):
        annotations.append(
            dict(
                text=str(round(value, 2)),
                x=correlation_matrix.columns[j],
                y=correlation_matrix.index[i],
                xref='x',
                yref='y',
                showarrow=False
            )
        )
st.write('3) Here we can see the high correlation between the features that imply there is '
         'multicolinerity is present among the features , hence I only use num character feature '
         'as it has the highest correlation with the target column.')
# Create heatmap using Plotly
fig5 = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.index,
    colorscale='plotly3',
    colorbar=dict(title='Correlation'),
    text=correlation_matrix.round(2).values
))
fig5.update_layout(title='Correlation Heatmap', width=600, height=400,annotations=annotations)
st.plotly_chart(fig5)

image_path1 = 'C:/Users/korak/PycharmProjects/spam_detection/sample_plot1.png'
image_path2 = 'C:/Users/korak/PycharmProjects/spam_detection/sample_plot2.png'
image_path3 = 'C:/Users/korak/PycharmProjects/spam_detection/download1.png'
image_path4 = 'C:/Users/korak/PycharmProjects/spam_detection/download2.png'

st.write('4) Most Common Spam And Ham Words')
try:
    with open(image_path1, 'rb') as f:
        st.image(f.read(), caption='Spam Words', use_column_width=True)
except FileNotFoundError:
    st.error('Image not found. Please provide a valid file path.')

try:
    with open(image_path2, 'rb') as f:
        st.image(f.read(), caption='Ham Words', use_column_width=True)
except FileNotFoundError:
    st.error('Image not found. Please provide a valid file path.')

st.write('5) Top 30 Spam And Ham Words')
try:
    with open(image_path3, 'rb') as f:
        st.image(f.read(), caption='Spam Words', use_column_width=True)
except FileNotFoundError:
    st.error('Image not found. Please provide a valid file path.')

try:
    with open(image_path4, 'rb') as f:
        st.image(f.read(), caption='Ham Words', use_column_width=True)
except FileNotFoundError:
    st.error('Image not found. Please provide a valid file path.')


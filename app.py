import json

import streamlit as st
import numpy as np
import pandas as pd


# Parameters
data_dir = f'./processed'
weight_dir = f'./weight'
info_path = f'./processed/summary_book.csv'
num = 10
lb = 0

# Load R matrix from file
R = np.load(f'{data_dir}/R.npy', allow_pickle=True)
# Load prediction
prediction = np.load(f'{weight_dir}/predicted.npy', allow_pickle=True)
# Load dictionary from JSON file
with open(f'{data_dir}/user_id_map.json', 'r') as file:
    user2id = json.load(file)
with open(f'{data_dir}/book_id_map.json', 'r') as file:
    book2id = json.load(file)


# Define the input and output functions for Gradio
def recommend_books(user_id):
    # Recommend
    user_idx = user2id[str(user_id)]
    predict = prediction[:, user_idx]  # get prediction for user
    predict_dict = {book: np.round(predict[idx], 2) for book, idx in book2id.items()}
    # Load information about book
    book_df = pd.read_csv(info_path)
    book_df = book_df[book_df["Num-Rating"] > lb]
    book_df['predict'] = book_df["ISBN"].map(predict_dict)
    df = book_df.nlargest(num, ["predict", "Mean-Rating"]).reset_index(drop=True)
    df["context"] = df.apply(
        lambda book: f"{book['Book-Title']} ({book['Year-Of-Publication']}) - by {book['Book-Author']}", axis=1
    )

    return df['context'].values

st.title('Book Recommender System')

# Display dialogue box that contains content
user_id = st.selectbox(
    'Enter your ID:',
    user2id.keys()
)

# Setting a button
if st.button('Recommend'):
    recommendations = recommend_books(user_id)
    st.write('**_Your ID:_**', user_id)
    st.write('**_Your top 10 recommendations:_**')
    for num, i in enumerate(recommendations):
        st.write(num + 1, ':', i)

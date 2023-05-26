# Book Recommender System

This is a book recommender system that uses collaborative filtering algorithm to provide personalized book recommendations based on user preferences.

## Overview

The book recommender system uses collaborative filtering to analyze user behavior and patterns to recommend books that are likely to be of interest to the user. It leverages user ratings and preferences to find similar users and books, and then suggests books that the target user has not yet rated or interacted with.
I highly recommend you visit [User-Based Approach notebook](https://github.com/QuyAnh2005/recommender-systems/blob/main/Collaborative%20Filtering/User-Based%20Approach.ipynb) to understand algorithm.

## Features

- Personalized book recommendations based on user preferences
- Collaborative filtering algorithm for accurate recommendations
- Efficient computation and scalable for small datasets
- Simple and user-friendly interface
- Easy integration with existing book-related applications or websites

## Data
Dataset is available at [here](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). After downloading, the `dataset` folder should have the below structure.
```
dataset
    - books.csv
    - ratings.csv
    - users.csv
    
app.py
preprocessing.py
recommend.py
requirements.txt
train.py
utils_c.py
```

## Installation

1. Clone the repository:
```shell
git clone https://github.com/your-username/book-recommender-system.git
```

2. Install the required dependencies:
```shell
pip install -r requirements.txt
```

## Usage

1. Preprocessing data
```shell
python preprocessing.py
```
or 
```shell
python preprocessing.py --rating_path ./dataset/ratings.csv --book_path ./dataset/books.csv --out_dir ./processed --limit 1000
```
More detail at [preproocessing.py](preproocessing.py)

2. Training model
```shell
python train.py
```
More detail at [train.py](train.py)

3. Recommendation
```shell
python recommend.py
```
More detail at [recommend.py](recommend.py)
## Demo
Run 
```shell
streamlit run app.py
```
or available at [my hugging face](https://huggingface.co/spaces/quyanh/Book-Recommender-System).
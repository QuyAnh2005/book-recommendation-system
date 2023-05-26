import json
import numpy as np
import pandas as pd

from jsonargparse import ArgumentParser


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, default="./processed")
    parser.add_argument("--weight_dir", type=str, required=True, default="./weight")
    parser.add_argument("--info_path", type=str, required=True, default="./processed/summary_book.csv")
    parser.add_argument("--user_id", required=True, default="276729")
    parser.add_argument("--num", type=int, required=True, default=10)
    parser.add_argument("--lb", type=int, required=True, default=0)

    return vars(parser.parse_args())

def main(
    data_dir,
    weight_dir,
    info_path,
    user_id,
    num,
    lb,
    **kwargs
):
    # Load R matrix from file
    R = np.load(f'{data_dir}/R.npy', allow_pickle=True)
    # Load prediction
    prediction = np.load(f'{weight_dir}/predicted.npy', allow_pickle=True)
    # Load dictionary from JSON file
    with open(f'{data_dir}/user_id_map.json', 'r') as file:
        user2id = json.load(file)
    with open(f'{data_dir}/book_id_map.json', 'r') as file:
        book2id = json.load(file)

    # Recommend
    user_idx = user2id[str(user_id)]
    predict = prediction[:, user_idx]   # get prediction for user
    predict_dict = {book: np.round(predict[idx], 2) for book, idx in book2id.items()}
    # Load information about book
    book_df = pd.read_csv(info_path)
    book_df = book_df[book_df["Num-Rating"] > lb]
    book_df['predict'] = book_df["ISBN"].map(predict_dict)
    recommendations = book_df.nlargest(num, ["predict", "Mean-Rating"]).reset_index(drop=True)
    print(recommendations)


if __name__ == "__main__":
    main(**parse_args())

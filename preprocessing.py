import json
import yaml
import pandas as pd
import numpy as np

from pathlib import Path
from jsonargparse import ArgumentParser


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--rating_path", type=str, required=True, default="./dataset/ratings.csv")
    parser.add_argument("--book_path", type=str, required=True, default="./dataset/books.csv")
    parser.add_argument("--out_dir", type=str, required=True, default="./processed")
    parser.add_argument("--limit", required=True, type=int, default=1000)

    return vars(parser.parse_args())


def main(
    rating_path,
    book_path,
    out_dir,
    limit,
    **kwargs
):
    data = pd.read_csv(rating_path, delimiter=';', nrows=limit, encoding='ISO-8859-1')

    # Make Y
    Y = data.pivot(index='ISBN', columns='User-ID', values='Book-Rating')
    Y = Y.fillna(0)
    Y = Y.values

    # Make R
    R = np.where(Y != 0, 1, 0)

    # Save Y and R as dense matrices
    out_dir_path = Path(out_dir)
    if out_dir_path.exists():
        assert out_dir_path.is_dir()
    else:
        out_dir_path.mkdir(parents=True)
    np.save(f'{out_dir_path}/Y.npy', Y)
    np.save(f'{out_dir_path}/R.npy', R)

    # Create mappings for book and user IDs
    book_lst = data['ISBN'].unique()
    user_lst = data['User-ID'].unique()
    book_id_map = {book_id: i for i, book_id in enumerate(book_lst)}
    user_id_map = {user_id: i for i, user_id in enumerate(user_lst)}
    # Convert keys to compatible types
    book_id_map = {str(key): value for key, value in book_id_map.items()}
    user_id_map = {str(key): value for key, value in user_id_map.items()}

    # Save book_id_map to file
    with open(f'{out_dir_path}/book_id_map.json', 'w') as f:
        json.dump(book_id_map, f)

    # Save user_id_map to file
    with open(f'{out_dir_path}/user_id_map.json', 'w') as f:
        json.dump(user_id_map, f)

    # Get summary
    function = {
        "Book-Rating": "mean",
        "User-ID": "count"
    }

    book_df = pd.read_csv(book_path, delimiter=';', encoding='ISO-8859-1', on_bad_lines='skip')
    summary_rating = data.groupby("ISBN").agg(function, axis=0)
    summary_rating = summary_rating.rename(columns={"Book-Rating": "Mean-Rating", "User-ID": "Num-Rating"})
    df = book_df.merge(summary_rating, how="left", left_on="ISBN", right_on="ISBN")
    df.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"], inplace=True)
    df.to_csv(f"{out_dir_path}/summary_book.csv", index=False)


if __name__ == "__main__":
    main(**parse_args())

import pickle
import polars as pl
from lib import add_col

DATA_DIR = './tracon01/'

# ---

# コミット情報を読み取り
print("コミット情報を読み取っています...")
commits = pl.read_csv(DATA_DIR + 'commits_sampled_10.csv')

# ---

with open('cache/load_commit.commits.pkl', 'wb') as f:
    pickle.dump(commits, f)
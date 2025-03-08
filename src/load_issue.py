import pickle
import polars as pl
from lib import add_col

DATA_DIR = './tracon01/'

# ---

print("issue情報を読み取っています...")
issues = pl.read_csv(DATA_DIR + 'issues.csv')

# ---

with open('cache/load_issue.issues.pkl', 'wb') as f:
    pickle.dump(issues, f)
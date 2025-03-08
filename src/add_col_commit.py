import pickle
import polars as pl
from lib import add_col

DATA_DIR = './tracon01/'

with open('cache/add_col_repo.train.pkl', 'rb') as f:
    train = pickle.load(f)
with open('cache/add_col_repo.test.pkl', 'rb') as f:
    test = pickle.load(f)

with open('cache/reshape_commit_data.repo_commit_cnt_df.pkl', 'rb') as f:
    repo_commit_cnt_df = pickle.load(f)
with open('cache/reshape_commit_data.repo_latest_commit_date_df.pkl', 'rb') as f:
    repo_latest_commit_date_df = pickle.load(f)
with open('cache/reshape_commit_data.repo_commit_members_df.pkl', 'rb') as f:
    repo_commit_members_df = pickle.load(f)
with open('cache/reshape_commit_data.repo_commit_message_sum_df.pkl', 'rb') as f:
    repo_commit_message_sum_df = pickle.load(f)
with open('cache/reshape_commit_data.repo_recent_commit_cnt_df.pkl', 'rb') as f:
    repo_recent_commit_cnt_df = pickle.load(f)

print(train)

# ---

print("コミット情報を結合します...")

train = train.join(repo_commit_cnt_df, on='repo_url', how='left')
test = test.join(repo_commit_cnt_df, on='repo_url', how='left')

train = train.join(repo_latest_commit_date_df, on='repo_url', how='left')
test = test.join(repo_latest_commit_date_df, on='repo_url', how='left')

train = train.join(repo_commit_members_df, on='repo_url', how='left')
test = test.join(repo_commit_members_df, on='repo_url', how='left')

train = train.join(repo_commit_message_sum_df, on='repo_url', how='left')
test = test.join(repo_commit_message_sum_df, on='repo_url', how='left')

train = train.join(repo_recent_commit_cnt_df, on='repo_url', how='left')
test = test.join(repo_recent_commit_cnt_df, on='repo_url', how='left')

print("コミット情報を読み取れました")

# ---

with open('cache/add_col_commit.train.pkl', 'wb') as f:
    pickle.dump(train, f)
with open('cache/add_col_commit.test.pkl', 'wb') as f:
    pickle.dump(test, f)
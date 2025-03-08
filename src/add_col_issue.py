import pickle
import polars as pl
from lib import add_col

DATA_DIR = './tracon01/'

with open('cache/add_col_commit.train.pkl', 'rb') as f:
    train = pickle.load(f)
with open('cache/add_col_commit.test.pkl', 'rb') as f:
    test = pickle.load(f)

with open('cache/reshape_issue_data.issue_count_df.pkl', 'rb') as f:
    issue_count_df = pickle.load(f)
with open('cache/reshape_issue_data.issue_open_count_df.pkl', 'rb') as f:
    issue_open_count_df = pickle.load(f)
with open('cache/reshape_issue_data.repo_latest_closed_issue_df.pkl', 'rb') as f:
    repo_latest_closed_issue_df = pickle.load(f)
with open('cache/reshape_issue_data.issue_message_len_df.pkl', 'rb') as f:
    issue_message_len_df = pickle.load(f)

print(train)

# ---

print("issue情報を結合しています...")

train = train.join(issue_count_df, on='repo_id', how='left')
test = test.join(issue_count_df, on='repo_id', how='left')

train = train.join(issue_open_count_df, on='repo_id', how='left')
test = test.join(issue_open_count_df, on='repo_id', how='left')

train = train.join(repo_latest_closed_issue_df, on='repo_id', how='left')
test = test.join(repo_latest_closed_issue_df, on='repo_id', how='left')

train = add_col(train, "issue_open_ratio", train["n_open_issues"] / train["n_issues"])
test = add_col(test, "issue_open_ratio", test["n_open_issues"] / test["n_issues"])

train = train.join(issue_message_len_df, on='repo_id', how='left')
test = test.join(issue_message_len_df, on='repo_id', how='left')

train = add_col(train, "ave_issue_body_len",
                       train["issue_message_len"] / train["n_issues"]).fill_null(0).fill_nan(0)
test_merged = add_col(test, "ave_issue_body_len",
                      test["issue_message_len"] / test["n_issues"]).fill_null(0).fill_nan(0)

print("issue情報の取り込みが完了しました")

# ---

with open('cache/add_col_issue.train.pkl', 'wb') as f:
    pickle.dump(train, f)
with open('cache/add_col_issue.test.pkl', 'wb') as f:
    pickle.dump(test, f)
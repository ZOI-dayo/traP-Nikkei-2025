import pickle
import polars as pl
from lib import add_col

DATA_DIR = './tracon01/'

with open('cache/add_col_issue.train.pkl', 'rb') as f:
    train = pickle.load(f)
with open('cache/add_col_issue.test.pkl', 'rb') as f:
    test = pickle.load(f)

# ---

print("PR情報を読み取っています...")
issues = pl.read_csv(DATA_DIR + 'pulls.csv')

print("PR情報を加工しています...")

pull_count_map = {}
pull_open_count_map = {}
for row in issues.iter_rows(named=True):
    pull_count_map[row["repo_id"]] = pull_count_map.get(row["repo_id"], 0) + 1
    if row["state"] == 'open':
        pull_open_count_map[row["repo_id"]] = pull_open_count_map.get(row["repo_id"], 0) + 1
pull_count_df = pl.DataFrame({"repo_id": pull_count_map.keys(), "n_pulls": pull_count_map.values()})
pull_open_count_df = pl.DataFrame(
    {"repo_id": pull_open_count_map.keys(), "n_open_pulls": pull_open_count_map.values()}
)

print("PR情報を結合しています...")

train = train.join(pull_count_df, on='repo_id', how='left').fill_null(0)
test = test.join(pull_count_df, on='repo_id', how='left').fill_null(0)

train = train.join(pull_open_count_df, on='repo_id', how='left').fill_null(0)
test = test.join(pull_open_count_df, on='repo_id', how='left').fill_null(0)

print(train["n_pulls"])
print(train["n_open_pulls"])

train = add_col(train, "pull_open_ratio", train["n_open_pulls"] / (train["n_pulls"] + 0.01))
test = add_col(test, "pull_open_ratio", test["n_open_pulls"] / (test["n_pulls"] + 0.01))

print(train["pull_open_ratio"])
print("PR情報の取り込みが完了しました")

# ---

with open('cache/all_load_pull.train.pkl', 'wb') as f:
    pickle.dump(train, f)
with open('cache/all_load_pull.test.pkl', 'wb') as f:
    pickle.dump(test, f)
import pickle
import polars as pl

DATA_DIR = './tracon01/'

with open('cache/load_issue.issues.pkl', 'rb') as f:
    issues = pickle.load(f)

# ---

from datetime import datetime as dt

print("issue情報を加工しています...")

issue_count_map = {}
issue_open_count_map = {}
repo_latest_closed_issue_map = {}
issue_message_len_map = {}
for row in issues.iter_rows(named=True):
    issue_count_map[row["repo_id"]] = issue_count_map.get(row["repo_id"], 0) + 1
    if row["state"] == 'open':
        issue_open_count_map[row["repo_id"]] = issue_open_count_map.get(row["repo_id"], 0) + 1
    else:
        if row["closed_at"] is not None:
            date = dt.fromisoformat(row["closed_at"]).timestamp()
            repo_latest_closed_issue_map[row["repo_id"]] = max(repo_latest_closed_issue_map.get(row["repo_id"], 0),
                                                               date)
    if row["body"] is not None:
        issue_message_len_map[row["repo_id"]] = issue_message_len_map.get(row["repo_id"], 0) + len(row["body"])
issue_count_df = pl.DataFrame({"repo_id": issue_count_map.keys(), "n_issues": issue_count_map.values()})
issue_open_count_df = pl.DataFrame(
    {"repo_id": issue_open_count_map.keys(), "n_open_issues": issue_open_count_map.values()})
repo_latest_closed_issue_df = pl.DataFrame(
    {"repo_id": repo_latest_closed_issue_map.keys(), "latest_closed_issue": repo_latest_closed_issue_map.values()}
)
issue_message_len_df = pl.DataFrame(
    {"repo_id": issue_message_len_map.keys(), "issue_message_len": issue_message_len_map.values()}
)

# ---

with open('cache/reshape_issue_data.issue_count_df.pkl', 'wb') as f:
    pickle.dump(issue_count_df, f)
with open('cache/reshape_issue_data.issue_open_count_df.pkl', 'wb') as f:
    pickle.dump(issue_open_count_df, f)
with open('cache/reshape_issue_data.repo_latest_closed_issue_df.pkl', 'wb') as f:
    pickle.dump(repo_latest_closed_issue_df, f)
with open('cache/reshape_issue_data.issue_message_len_df.pkl', 'wb') as f:
    pickle.dump(issue_message_len_df, f)

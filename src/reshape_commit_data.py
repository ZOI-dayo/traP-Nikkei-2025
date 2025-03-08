import pickle
import polars as pl

DATA_DIR = './tracon01/'

with open('cache/load_commit.commits.pkl', 'rb') as f:
    commits = pickle.load(f)

# ---

print("コミット情報を集計します...")

from datetime import datetime as dt

repo_commit_cnt = {}
repo_latest_commit_date = {}
repo_commit_members = {}
repo_commit_message_sum = {}
repo_recent_commit_cnt = {}
repo_max_repocnt = {}

for row in commits.iter_rows(named=True):
    # print(f"row.repo_names = {row.repo_names}")
    # 日付として Author Date を利用
    if row["author_date"] is None:
        continue
    date = dt.strptime(row["author_date"], '%Y-%m-%d %H:%M:%S').timestamp()
    repos = eval(row["repo_names"])
    for repo_name in repos:
        # print(f"repo_name = {repo_name}")
        repo_commit_cnt[repo_name] = repo_commit_cnt.get(repo_name, 0) + 1
        repo_latest_commit_date[repo_name] = max(repo_latest_commit_date.get(repo_name, 0), date)
        if not repo_name in repo_commit_members:
            repo_commit_members[repo_name] = {}
        repo_commit_members[repo_name][row["author_name"]] = \
            repo_commit_members[repo_name].get(row["author_name"], 0) + 1
        if row["message"] is not None:
            repo_commit_message_sum[repo_name] = repo_commit_message_sum.get(repo_name, 0) + len(row["message"])
        if date > 1640995200 - 31 * 24 * 60 * 60:  # 2022/1/1 = 1640995200
            repo_recent_commit_cnt[repo_name] = repo_recent_commit_cnt.get(repo_name, 0) + 1
        repo_max_repocnt[repo_name] = max(repo_max_repocnt.get(repo_name, 1), len(repos))

repo_commit_cnt_df = pl.DataFrame({"repo_url": repo_commit_cnt.keys(), "n_commits": repo_commit_cnt.values()})
repo_latest_commit_date_df = pl.DataFrame(
    {"repo_url": repo_latest_commit_date.keys(), "last_commit_date": repo_latest_commit_date.values()})
repo_commit_members_df = pl.DataFrame(
    {
        "repo_url": repo_commit_members.keys(),
        "n_commit_members": [len(e) for e in repo_commit_members.values()],
        "first_author_ratio": [
            max(e) / sum(e) if len(e) != 0 else 0 for e in
            [sorted(list(e.values())) for e in repo_commit_members.values()]
        ],
        "first_or_second_author_ratio": [
            (
                e[0] + e[1] if len(e) >= 2
                else e[0] if len(e) == 1
                else 0
            ) / sum(e) for e in
            [list(reversed(sorted(list(e.values())))) for e in repo_commit_members.values()]
        ],
    }
).fill_nan(-1)
repo_commit_message_sum_df = pl.DataFrame(
    {"repo_url": repo_commit_message_sum.keys(), "len_commit_messages": repo_commit_message_sum.values()}
)
repo_recent_commit_cnt_df = pl.DataFrame(
    {"repo_url": repo_recent_commit_cnt.keys(), "n_recent_commits": repo_recent_commit_cnt.values()}
)
repo_max_repocnt_df = pl.DataFrame(
    {"repo_url": repo_recent_commit_cnt.keys(), "n_max_repocnt": repo_recent_commit_cnt.values()}
).fill_null(0)

# ---

with open('cache/reshape_commit_data.repo_commit_cnt_df.pkl', 'wb') as f:
    pickle.dump(repo_commit_cnt_df, f)
with open('cache/reshape_commit_data.repo_latest_commit_date_df.pkl', 'wb') as f:
    pickle.dump(repo_latest_commit_date_df, f)
with open('cache/reshape_commit_data.repo_commit_members_df.pkl', 'wb') as f:
    pickle.dump(repo_commit_members_df, f)
with open('cache/reshape_commit_data.repo_commit_message_sum_df.pkl', 'wb') as f:
    pickle.dump(repo_commit_message_sum_df, f)
with open('cache/reshape_commit_data.repo_recent_commit_cnt_df.pkl', 'wb') as f:
    pickle.dump(repo_recent_commit_cnt_df, f)
with open('cache/reshape_commit_data.repo_max_repocnt_df.pkl', 'wb') as f:
    pickle.dump(repo_max_repocnt_df, f)

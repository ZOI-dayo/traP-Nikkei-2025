from lib import add_col

import pickle
import polars as pl
from lib import add_col

DATA_DIR = './tracon01/'

with open('cache/all_load_pull.train.pkl', 'rb') as f:
    train = pickle.load(f)
with open('cache/all_load_pull.test.pkl', 'rb') as f:
    test = pickle.load(f)
    
# ---

from datetime import datetime as dt

# 要素数を取得する関数
def list_len(s: str):
    return s.count(',')


# 各データの "n_stars" に "stars" の要素数をいれる
train = add_col(train, "n_stars", train["stars"].map_elements(list_len))
test = add_col(test, "n_stars", test["stars"].map_elements(list_len))
# train["n_stars"] = train["stars"].apply(list_len)
# test["n_stars"] = test["stars"].apply(list_len)

train = add_col(train, "n_recent_stars", train["stars"].map_elements(
    lambda x: len(list(
        filter(
            # 2022/1/1 = 1640995200
            lambda y: dt.fromisoformat(y["created_at"]).timestamp() > 1640995200 - 31 * 24 * 60 * 60,
            eval(x)
        )
    ))
))
test = add_col(test, "n_recent_stars", test["stars"].map_elements(list_len))

# 各データの "n_files" に "files" の要素数をいれる
train = add_col(train, "n_files", train["files"].map_elements(list_len))
test = add_col(test, "n_files", test["files"].map_elements(list_len))
# train["n_files"] = train["files"].apply(list_len)
# test["n_files"] = test["files"].apply(list_len)

# n_starsの分布を表示
# plt.hist(train['n_stars'], bins=50, alpha=0.5, label='train', log=True)
# plt.hist(test['n_stars'], bins=50, alpha=0.5, label='test', log=True)
# plt.show()

# n_filesの分布を表示
# plt.hist(train['n_files'], bins=50, alpha=0.5, label='train', log=True)
# plt.hist(test['n_files'], bins=50, alpha=0.5, label='test', log=True)
# plt.legend()
# plt.show()

# "star_file_ratio" に n_files / n_stars を代入 (スターが多いほど小さく、ファイルが多いほど大きくなる)
train = add_col(train, "star_file_ratio", train["n_files"] / train["n_stars"])
test = add_col(test, "star_file_ratio", test["n_files"] / test["n_stars"])
# train["star_file_ratio"] = train["n_files"] / train["n_stars"]
# test["star_file_ratio"] = test["n_files"] / test["n_stars"]

# "file_par_commit" に n_files / n_commits を代入 (commitあたりの平均ファイル数)
train = add_col(train, "file_par_commit", train["n_files"] / train["n_commits"])
test = add_col(test, "file_par_commit", test["n_files"] / test["n_commits"])
# train["file_par_commit"] = train["n_files"] / train["n_commits"]
# test["file_par_commit"] = test["n_files"] / test["n_commits"]

train = add_col(train, "star_par_commit", train["n_stars"] / train["n_commits"])
test = add_col(test, "star_par_commit", test["n_stars"] / test["n_commits"])

import math

# n_issues はlogを取ったほうが扱いやすい値なので操作
# train = add_col(train, "n_issues_log", train["n_issues"].map_elements(math.log))
# test = add_col(test, "n_issues_log", test["n_issues"].map_elements(math.log))
# train["n_issues_log"] = train["n_issues"].apply(math.log)
# test["n_issues_log"] = test["n_issues"].apply(math.log)

# ---

with open('cache/add_col.train.pkl', 'wb') as f:
    pickle.dump(train, f)
with open('cache/add_col.test.pkl', 'wb') as f:
    pickle.dump(test, f)
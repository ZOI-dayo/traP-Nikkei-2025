import pickle
import polars as pl
from lib import add_col

DATA_DIR = './tracon01/'

with open('cache/load_repo.train.pkl', 'rb') as f:
    train = pickle.load(f)
with open('cache/load_repo.test.pkl', 'rb') as f:
    test = pickle.load(f)

print(train)

# ---

def get_readme_size(s):
    l = eval(s)
    for e in l:
        if e["name"] == "README.md":
            return e["size"]
    return 0


def get_file_size(s):
    l = eval(s)
    res = 0
    for e in l:
        res += e["size"]
    return res

train = add_col(train, "readme_size", train["files"].map_elements(get_readme_size))
test = add_col(test, "readme_size", test["files"].map_elements(get_readme_size))

train = add_col(train, "file_size", train["files"].map_elements(get_file_size))
test = add_col(test, "file_size", test["files"].map_elements(get_file_size))

readme_size_cnt = {}

for row in train.iter_rows(named=True):
    readme_size_cnt[row["readme_size"]] = readme_size_cnt.get(row["readme_size"], 0) + 1
for row in test.iter_rows(named=True):
    readme_size_cnt[row["readme_size"]] = readme_size_cnt.get(row["readme_size"], 0) + 1


def get_readme_count(readme_size):
    return readme_size_cnt[readme_size]

train = add_col(train, "readme_size_cnt", train["readme_size"].map_elements(get_readme_count))
test = add_col(test, "readme_size_cnt", test["readme_size"].map_elements(get_readme_count))

train = add_col(train, "repo_url", train["owner"] + "/" + train["repo"])
test = add_col(test, "repo_url", test["owner"] + "/" + test["repo"])
# ---

with open('cache/add_col_repo.train.pkl', 'wb') as f:
    pickle.dump(train, f)
with open('cache/add_col_repo.test.pkl', 'wb') as f:
    pickle.dump(test, f)
import pickle
import polars as pl

DATA_DIR = './tracon01/'

with open('cache/load_train.train.pkl', 'rb') as f:
    train = pickle.load(f)
with open('cache/load_train.test.pkl', 'rb') as f:
    test = pickle.load(f)

repo = pl.read_csv(DATA_DIR + 'repo.csv')

train = train.join(repo, on='repo_id', how='left')
test = test.join(repo, on='repo_id', how='left')

with open('cache/load_repo.train.pkl', 'wb') as f:
    pickle.dump(train, f)
with open('cache/load_repo.test.pkl', 'wb') as f:
    pickle.dump(test, f)

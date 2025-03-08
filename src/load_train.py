import pickle
import polars as pl

DATA_DIR = './tracon01/'

train = pl.read_csv(DATA_DIR + 'train.csv')
test = pl.read_csv(DATA_DIR + 'test.csv')

with open('cache/load_train.train.pkl', 'wb') as f:
    pickle.dump(train, f)
with open('cache/load_train.test.pkl', 'wb') as f:
    pickle.dump(test, f)

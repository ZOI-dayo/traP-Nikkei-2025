import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# データ読み込み
DATA_DIR = './tracon01/'
train = pd.read_csv(DATA_DIR + 'train.csv')
test = pd.read_csv(DATA_DIR + 'test.csv')

print(train)
print(test)

# レポジトリ情報を読み取り
repo = pd.read_csv(DATA_DIR + 'repo.csv')

# repo_idをもとにレポジトリ情報を結合
train_merged = train.merge(repo, on='repo_id', how='left')
test_merged = test.merge(repo, on='repo_id', how='left')

# repo urlを追加
train_merged["repo_url"] = train_merged["owner"] + "/" + train_merged["repo"]
test_merged["repo_url"] = test_merged["owner"] + "/" + test_merged["repo"]

# コミット情報を読み取り
print("コミット情報を読み取っています...")
commits = pd.read_csv(DATA_DIR + 'commits_sampled_10.csv')

print("コミット情報を集計します...")

repo_commit_cnt = {}

for row in commits.itertuples():
    # print(f"row.repo_names = {row.repo_names}")
    for repo_name in eval(row.repo_names):
        # print(f"repo_name = {repo_name}")
        repo_commit_cnt[repo_name] = repo_commit_cnt.get(repo_name, 0) + 1

repo_commit_cnt_df = pd.DataFrame({"repo_url" : repo_commit_cnt.keys(), "commit_cnt": repo_commit_cnt.values()})
# print(repo_commit_cnt_df)

print("コミット情報を結合します...")

print(repo_commit_cnt_df)
print(train_merged["repo_url"])

train_merged = train_merged.merge(repo_commit_cnt_df, on='repo_url', how='left')
test_merged = test_merged.merge(repo_commit_cnt_df, on='repo_url', how='left')

print("コミット情報を読み取れました")

print(train_merged["commit_cnt"])


# 要素数を取得する関数
def list_len(s: str):
    return s.count(',')


# 各データの "n_stars" に "stars" の要素数をいれる
train_merged["n_stars"] = train_merged["stars"].apply(list_len)
test_merged["n_stars"] = test_merged["stars"].apply(list_len)

# 各データの "n_files" に "files" の要素数をいれる
train_merged["n_files"] = train_merged["files"].apply(list_len)
test_merged["n_files"] = test_merged["files"].apply(list_len)

# n_starsの分布を表示
plt.hist(train_merged['n_stars'], bins=50, alpha=0.5, label='train', log=True)
plt.hist(test_merged['n_stars'], bins=50, alpha=0.5, label='test', log=True)
plt.show()

# n_filesの分布を表示
plt.hist(train_merged['n_files'], bins=50, alpha=0.5, label='train', log=True)
plt.hist(test_merged['n_files'], bins=50, alpha=0.5, label='test', log=True)
plt.legend()
plt.show()

# "star_file_ratio" に n_files / n_stars を代入 (スターが多いほど小さく、ファイルが多いほど大きくなる)
train_merged["star_file_ratio"] = train_merged["n_files"] / train_merged["n_stars"]
test_merged["star_file_ratio"] = test_merged["n_files"] / test_merged["n_stars"]

import seaborn as sns

def show_dist(df, key):
    sns.displot(df[key], kde=False, rug=False, log_scale=10).set(title=f"{key} log : all")
    sns.displot(df[df["active"] == True][key], kde=False, rug=False, log_scale=10).set(title=f"{key} log : true")
    sns.displot(df[df["active"] == False][key], kde=False, rug=False, log_scale=10).set(title=f"{key} log : false")


# fileの個数分布を表示
show_dist(train_merged, "n_files")
# starの個数分布を表示
show_dist(train_merged, "n_stars")

show_dist(train_merged, "star_file_ratio")

show_dist(train_merged, "commit_cnt")

# KFoldでデータを分割
kf = KFold(n_splits=4, shuffle=True, random_state=34)

# 学習対象の行
use_cols = ["n_stars", "n_files", "star_file_ratio", "commit_cnt"]
target_col = "active"

for train_index, valid_index in kf.split(train_merged):
    train_data = train_merged.iloc[train_index]
    valid_data = train_merged.iloc[valid_index]
    print("学習データの数:", len(train_data), "検証データの数:", len(valid_data))

import lightgbm as lgb


# 各分割をもらって、lightgbm のモデルを訓練して訓練した後のモデルを返す関数
def train_fold(train_X: pd.DataFrame, train_y: pd.Series, valid_X: pd.DataFrame, valid_y: pd.Series) -> lgb.Booster:
    # データセットを作成
    lgb_train = lgb.Dataset(train_X, train_y)
    lgb_valid = lgb.Dataset(valid_X, valid_y, reference=lgb_train)

    params = {
        # 二値分類として解く
        'objective': 'binary',
        # 評価指標として binary_logloss と binary_error を使う
        'metric': ['binary_logloss', 'binary_error'],
    }

    # 学習. binary_logloss が 100ステップ以上改善しないなら打ち切るように設定する
    model = lgb.train(params, lgb_train, valid_sets=[lgb_valid],
                      callbacks=[lgb.early_stopping(100, first_metric_only=True)])

    return model


# 各分割で学習した結果をいれた
models = []

for train_index, valid_index in kf.split(train_merged):
    train_data = train_merged.iloc[train_index]
    valid_data = train_merged.iloc[valid_index]

    train_X = train_data[use_cols]
    train_y = train_data[target_col]

    valid_X = valid_data[use_cols]
    valid_y = valid_data[target_col]

    model = train_fold(train_X, train_y, valid_X, valid_y)

    models.append(model)

# `oof_pred` に今回訓練したモデルたちによる予測結果を格納する

import numpy as np

oof_pred = np.zeros(len(train_merged))

for i, (train_index, valid_index) in enumerate(kf.split(train_merged)):
    # バリデーションデータを取り出す
    valid_data = train_merged.iloc[valid_index]
    valid_X = valid_data[use_cols]

    # 予測結果を出力
    oof_pred[valid_index] = models[i].predict(valid_X)

from sklearn.metrics import roc_auc_score, roc_curve

score = roc_auc_score(train_merged["active"], oof_pred)

print("score: ", score)

# ROC 曲線のプロット
fpr, tpr, thresholds = roc_curve(train_merged["active"], oof_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {score}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # ランダムな分類器の基準線
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns


def plot_importance(models: list):
    feature_importance = pd.DataFrame()

    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df['feature_importance'] = model.feature_importance()
        _df['column'] = model.feature_name()
        _df['fold'] = i + 1
        feature_importance = pd.concat([feature_importance, _df], axis=0, ignore_index=True)

    order = feature_importance.groupby('column').sum()[['feature_importance']].sort_values('feature_importance',
                                                                                           ascending=False).index

    plt.figure(figsize=(6, 4))
    sns.boxenplot(data=feature_importance, x='feature_importance', y='column', order=order)
    plt.title('feature importance')
    plt.grid()
    plt.show()


# 各データをどれだけ重視したか見る
plot_importance(models)

# 予測値の分布を表示
plt.hist(oof_pred, bins=20)
plt.show()

from sklearn.metrics import confusion_matrix

# 提出

sample_sub = pd.read_csv(DATA_DIR + 'sample_submission.csv')

pred = np.zeros(sample_sub.shape[0])

for model in models:
    pred += model.predict(test_merged[use_cols])

pred /= len(models)

sample_sub["pred"] = pred

plt.hist(pred, bins=30)
plt.show()

sample_sub.to_csv("submission.csv", index=False)

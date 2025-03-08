from lib import add_col

import pickle
import polars as pl
from lib import add_col

DATA_DIR = './tracon01/'

with open('cache/add_col.train.pkl', 'rb') as f:
    train = pickle.load(f)
with open('cache/add_col.test.pkl', 'rb') as f:
    test = pickle.load(f)

# ---

import seaborn as sns
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np


def show_dist(df, key, log):
    sns.displot(df[key], kde=False, rug=False, bins=500, log_scale=10 if log else None).set(title=f"{key} log : all")
    sns.displot(df.filter(pl.col("active") == True)[key], kde=False, rug=False, bins=500,
                log_scale=10 if log else None).set(
        title=f"{key} log : true")
    sns.displot(df.filter(pl.col("active") == False)[key], kde=False, rug=False, bins=500,
                log_scale=10 if log else None).set(
        title=f"{key} log : false")


# fileの個数分布を表示
# show_dist(train, "n_files")
# starの個数分布を表示
# show_dist(train, "n_stars")

# show_dist(train, "star_file_ratio")

# show_dist(train, "n_commits")

# show_dist(train, "file_par_commit")

show_dist(train, "author_timezonedelta_ave", False)

# KFoldでデータを分割
kf = KFold(n_splits=4, shuffle=True, random_state=34)

# 学習対象の行
use_cols = ["n_stars", "n_files", "star_file_ratio", "n_commits", "file_par_commit", "last_commit_date",
            "n_commit_members", "n_issues", "n_pulls", "readme_size", "readme_size_cnt", "latest_closed_issue",
            "file_size", "issue_open_ratio", "pull_open_ratio", "len_commit_messages", "star_par_commit",
            "n_recent_commits", "issue_message_len", "first_author_ratio", "first_or_second_author_ratio",
            "author_timezonedelta_ave", "owner_cnt"]
target_col = "active"

for train_index, valid_index in kf.split(train):
    train_data = train[train_index]
    valid_data = train[valid_index]
    print("学習データの数:", len(train_data), "検証データの数:", len(valid_data))

import lightgbm as lgb


# 各分割をもらって、lightgbm のモデルを訓練して訓練した後のモデルを返す関数
def train_fold(train_X: pd.DataFrame, train_y: pd.Series, valid_X: pd.DataFrame, valid_y: pd.Series,
               trial) -> lgb.Booster:
    # データセットを作成
    lgb_train = lgb.Dataset(train_X, train_y)
    lgb_valid = lgb.Dataset(valid_X, valid_y, reference=lgb_train)

    # https://zenn.dev/robes/articles/d53ff6d665650f
    # {'learning_rate': 0.036710796431638056, 'num_leaves': 74, 'min_data_in_leaf': 37, 'min_sum_hessian_in_leaf': 21, 'bagging_fraction': 0.1505235263266912, 'bagging_freq': 6, 'feature_fraction': 0.9944044553565908}
    # {'learning_rate': 0.035829155667069366, 'num_leaves': 99, 'min_data_in_leaf': 27, 'min_sum_hessian_in_leaf': 47, 'bagging_fraction': 0.395652389722927, 'bagging_freq': 5, 'feature_fraction': 0.9581394359092842}
    params = {
        # 二値分類として解く
        'objective': 'binary',
        # 評価指標として auc と accuracy を使う
        'metric': ['binary_logloss', 'binary_error'],
        # 'learning_rate':
        # # trial.suggest_uniform('learning_rate', 0.01, 0.05),
        #     0.036,
        # 'n_estimators': 1000,
        # 'importance_type': 'gain',
        # 'num_leaves':
        # # trial.suggest_int('num_leaves', 10, 100),
        #     90,
        # 'min_data_in_leaf':
        # # trial.suggest_int('min_data_in_leaf', 5, 50),
        #     30,
        # 'min_sum_hessian_in_leaf':
        # # trial.suggest_int('min_sum_hessian_in_leaf', 5, 50),
        #     35,
        # 'lambda_l1': 0,
        # 'lambda_l2': 0,
        # 'bagging_fraction':
        # # trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
        #     0.3,
        # 'bagging_freq':
        # # trial.suggest_int('bagging_freq', 0, 10),
        #     5,
        # 'feature_fraction':
        # # trial.suggest_uniform('feature_fraction', 0.1, 1.0),
        #     0.98,
        # 'random_seed': 42
        # # --- --- ---
        "learning_rate": 0.01,
        'feature_fraction': 0.8,
        'bagging_freq': 1,
        'bagging_fraction': 0.8,
        'num_leaves': 31,
        'random_state': 0,
        'num_iterations': 2000,
    }

    # 学習. auc が 100ステップ以上改善しないなら打ち切るように設定する
    model = lgb.train(params, lgb_train, valid_sets=[lgb_valid])

    return model


def learn(trial):
    # 各分割で学習した結果をいれた
    models = []

    for train_index, valid_index in kf.split(train):
        train_data = train[train_index]
        valid_data = train[valid_index]

        train_X = train_data[use_cols].to_pandas()
        train_y = train_data[target_col].to_pandas()

        valid_X = valid_data[use_cols].to_pandas()
        valid_y = valid_data[target_col].to_pandas()

        model = train_fold(train_X, train_y, valid_X, valid_y, trial)

        models.append(model)

    # `oof_pred` に今回訓練したモデルたちによる予測結果を格納する

    oof_pred = np.zeros(len(train))

    for i, (train_index, valid_index) in enumerate(kf.split(train)):
        # バリデーションデータを取り出す
        valid_data = train[valid_index]
        valid_X = valid_data[use_cols]

        # 予測結果を出力
        oof_pred[valid_index] = models[i].predict(valid_X)

    from sklearn.metrics import roc_auc_score, roc_curve

    score = roc_auc_score(train["active"], oof_pred)

    print("score: ", score)
    # return score

    # ROC 曲線のプロット
    # fpr, tpr, thresholds = roc_curve(train["active"], oof_pred)
    #
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {score}')
    # plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # ランダムな分類器の基準線
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve")
    # plt.legend()
    # plt.grid()
    # plt.show()
    return score, models, oof_pred


def objective(trial):
    score, models, oof_pred = learn(trial)
    return score


# import optuna
#
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=300)
#
# print('Number of finished trials:', len(study.trials))
# print('Best trial:', study.best_trial.params)
#
# exit(0)

score, models, oof_pred = learn(None)

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
    pred += model.predict(test[use_cols])

pred /= len(models)

sample_sub["pred"] = pred

plt.hist(pred, bins=30)
plt.show()

sample_sub.to_csv("submission.csv", index=False)

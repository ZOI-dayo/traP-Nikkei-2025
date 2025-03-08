run: submission.csv

cache/load_train.train.pkl: ./src/load_train.py
	python ./src/load_train.py
cache/load_repo.train.pkl: ./src/load_repo.py cache/load_train.train.pkl
	python ./src/load_repo.py
cache/add_col_repo.train.pkl: ./src/add_col_repo.py cache/load_repo.train.pkl
	python ./src/add_col_repo.py
cache/load_commit.commits.pkl: ./src/load_commit.py
	python ./src/load_commit.py
cache/reshape_commit_data.repo_commit_cnt_df.pkl: ./src/reshape_commit_data.py cache/load_commit.commits.pkl
	python ./src/reshape_commit_data.py
cache/add_col_commit.train.pkl: ./src/add_col_commit.py cache/add_col_repo.train.pkl cache/reshape_commit_data.repo_commit_cnt_df.pkl
	python ./src/add_col_commit.py
cache/load_issue.issues.pkl: ./src/load_issue.py
	python ./src/load_issue.py
cache/reshape_issue_data.issue_count_df.pkl: ./src/reshape_issue_data.py cache/load_issue.issues.pkl
	python ./src/reshape_issue_data.py
cache/add_col_issue.train.pkl: ./src/add_col_issue.py cache/add_col_commit.train.pkl cache/reshape_issue_data.issue_count_df.pkl
	python ./src/add_col_issue.py
cache/all_load_pull.train.pkl: ./src/all_load_pull.py cache/add_col_issue.train.pkl
	python ./src/all_load_pull.py
cache/add_col.train.pkl: ./src/add_col.py cache/all_load_pull.train.pkl
	python ./src/add_col.py
submission.csv: ./src/learn.py cache/add_col.train.pkl
	python ./src/learn.py
def add_col(df, name, add):
    return df.with_columns(add.alias(name))
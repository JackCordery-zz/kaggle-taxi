import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    return pd.read_csv(path)




def prepare_data(df, label_name, train_test_ratio, random_seed):
    df.dropna(inplace=True)
    df = df.drop(["key", "pickup_datetime"], axis=1)
    X = df.drop(label_name, axis=1)
    y = df[label_name]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=train_test_ratio, random_state=random_seed)
    return {"X_train": X_train, "X_val": X_val, "y_train": y_train, "y_val": y_val}


def main():
    print("Nothing to see here: {}".format(__file__))
    return


if __name__ == '__main__':
    main()

from config import PATHS, VARIABLES
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    return pd.read_csv(path)


def prepare_data(df, label_name, train_test_ratio, random_seed):
    columns = df.columns
    df = df.drop(["key", "pickup_datetime"], axis=1)
    X = df.drop(label_name, axis=1)
    y = df[label_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_ratio, random_state=random_seed)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def main():
    sample_path = PATHS["sample"]
    label_name = VARIABLES["label_name"]
    random_seed = VARIABLES["random_seed"]
    train_test_ratio = VARIABLES["train_test_ratio"]
    df = pd.read_csv(sample_path)
    print(df.head())
    print(df.describe())
    print("The dataframe has the following shape: {}".format(df.shape))
    data = prepare_data(df, label_name, train_test_ratio, random_seed)
    print(data["X_train"].head())
    print(data["y_train"].head())
    return


if __name__ == '__main__':
    main()

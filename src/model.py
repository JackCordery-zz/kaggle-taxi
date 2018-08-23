from math import sqrt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def get_model_stats(y_test, y_pred, coef_names, regressor):
    mse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    coef = dict(zip(coef_names, regressor.coef_))
    return {"rmse": mse, "r2": r2, "coef": coef}


def model_linear(data):
    X_train = data["X_train"]
    y_train = data["y_train"]
    regressor = linear_model.LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def predict(data, regressor):
    X_test = data["X_val"]
    y_pred = regressor.predict(X_test)
    return y_pred


def evaluate(data, regressor):
    y_test = data["y_val"]
    coef_names = data["X_train"].columns
    y_pred = predict(data, regressor)

    stats = get_model_stats(y_test, y_pred, coef_names, regressor)
    return stats


def main():
    print("Nothing to see here: {}".format(__file__))
    return


if __name__ == '__main__':
    main()

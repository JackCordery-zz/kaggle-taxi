from math import sqrt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def model_linear(data):
    X_train = data["X_train"]
    y_train = data["y_train"]
    regressor = linear_model.LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def predict(data, regressor):
    X_test = data["X_test"]
    y_pred = regressor.predict(X_test)
    return y_pred


def evaluate(data, regressor):
    y_test = data["y_test"]
    y_pred = predict(data, regressor)

    mse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {"rmse": mse, "r2": r2, "coef": regressor.coef_}


def main():
    print("Nothing to see here: {}".format(__file__))
    return


if __name__ == '__main__':
    main()

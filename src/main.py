from config import PATHS, VARIABLES
from etl import prepare_data, load_data
from model import model_linear, evaluate


def main():
    train_set_name = VARIABLES["train_set"]
    train_path = PATHS[train_set_name]
    label_name = VARIABLES["label_name"]
    random_seed = VARIABLES["random_seed"]
    train_test_ratio = VARIABLES["train_test_ratio"]

    df = load_data(train_path)
    data = prepare_data(df, label_name, train_test_ratio, random_seed)

    model = model_linear(data)
    evaluate_scores = evaluate(data, model)

    print(evaluate_scores)

    return


if __name__ == '__main__':
    main()

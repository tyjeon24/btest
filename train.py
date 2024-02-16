import bentoml
import xgboost

if __name__ == "__main__":
    # read in data
    dtrain = xgboost.DMatrix(data=[[0], [1], [2], [3], [4]], label=[0, 0, 1, 1, 1])

    # specify parameters via dictionary
    param = {
        "booster": "dart",
        "max_depth": 2,
        "eta": 1,
        "objective": "binary:logistic",
    }
    num_round = 2
    bst = xgboost.train(param, dtrain, num_round)

    bentoml.xgboost.save_model("agaricus", bst)

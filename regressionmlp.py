# -*- coding: utf-8 -*-

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from functions import loadCSV, load_position_data_batch, load_dataframe, standardize, destandardize, strip_nonnumeric_evaluations, cast_as
import FenTransformer

SCRIPTLOCATION = "/home/ml/sgchess"
position_data = load_position_data_batch("dataprocessed.p", SCRIPTLOCATION + "/dataprocessed")
evals = load_dataframe("evals.p", SCRIPTLOCATION + "/dataprocessed")
orig = evals.copy()

print("data loaded")
X_train_full, X_test, y_train_full, y_test = train_test_split(
    position_data, 
    standardize(evals),
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, 
    y_train_full, 
    random_state=4
)
print("data split")


mlp_reg = MLPRegressor(hidden_layer_sizes = [128, 64, 32], random_state=42)

#pipeline = make_pipeline(FenTransformer(), mlp_reg)

mlp_reg.fit(X_train, y_train)
print("model trained")
y_pred = mlp_reg.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(f'rmse: {rmse}')
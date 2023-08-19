# -*- coding: utf-8 -*-

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from functions import loadCSV, preprocess_position_data, load_dataframe, standardize, destandardize, strip_nonnumeric_evaluations, cast_as
import FenTransformer

position_data = load_dataframe("datatrimmed.p")

# Evaluation data is going to have strings containing # to indicate checkmating
#  sequences, and all kinds of other potential junk data.
# The chess-python module has functionality that can calculate checkmates,
# There's no need to teach the neural network these positions
# Therefore, these are going to be removed from the dataset in this step

position_data = strip_nonnumeric_evaluations(position_data)

# Cast these values which should always be numeric now into the appropriate type
#data.loc[:,(EVAL_COL_NAME)] = data.loc[:,(EVAL_COL_NAME)].astype('int')
cast_as(position_data)

evals = position_data['Evaluation']
orig = evals.copy()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    preprocess_position_data(position_data), 
    standardize(evals),
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, 
    y_train_full, 
    random_state=4
)



mlp_reg = MLPRegressor(hidden_layer_sizes = [5, 5], random_state=42)

#pipeline = make_pipeline(FenTransformer(), mlp_reg)
mlp_reg.fit(X_train, y_train)
y_pred = destandardize(mlp_reg.predict(X_valid), orig)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
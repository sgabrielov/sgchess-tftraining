# -*- coding: utf-8 -*-

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from functions import loadCSV, strip_checkmating_positions
import FenTransformer

position_data = loadCSV()
position_data = strip_checkmating_positions(position_data)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    position_data['FEN'], 
    position_data['Evaluation'],
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, 
    y_train_full, 
    random_state=4
)

mlp_reg = MLPRegressor(hidden_layer_sizes = [64, 16], random_state=42)

pipeline = make_pipeline(FenTransformer(), mlp_reg)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
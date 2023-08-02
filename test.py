# -*- coding: utf-8 -*-
import functions.py
from sklearn.neural_network import MLPRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

def main():

    #
    # Load California housing data set
    #
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    #
    # Create training/ test data split
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    #
    # Instantiate MLPRegressor
    #
    nn = MLPRegressor(
        activation='relu',
        hidden_layer_sizes=(10, 100),
        alpha=0.001,
        random_state=20,
        early_stopping=False
    )
    #
    # Train the model
    #
    nn.fit(X_train, y_train)
    
    
if __name__ == "__main__":
    main()
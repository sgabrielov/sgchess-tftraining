# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator, TransformerMixin
from functions import convert_fen_to_bitboard

class FenTransformer(BaseEstimator, TransformerMixin):
    
    # initialize the transformer with essential information about the dataframe
    def __init__(self, fenlabel='FEN'):
        # fenlabel is the ID of the fen string in the dataframe
        self.fenlabel=fenlabel
        
    # nothing to fit - this transformer will always convert a FEN string to 
    # bitwise notation the same way for every board
    def fit(self, X, y=None):
        return self
        
    
        
    # transform the FEN column in X by applying the convert_fen_to_bitboard
    # function to every element
    def transform(self, X):
        
        X.loc[:,('FEN')] = X.loc[:,('FEN')].transform(lambda fen: convert_fen_to_bitboard(fen))
        return X
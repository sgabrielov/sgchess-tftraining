from kaggle.api.kaggle_api_extended import KaggleApi
import json, os, sys, zipfile, chess, pandas, scipy, numpy, pickle
from tqdm import tqdm


# This file contains definitions for all useful library functons.
# Eventually, these will be split into separate files according to which
# modules they depend on 


SCRIPTLOCATION = "/home/ml/sgchess"
# SCRIPTLOCATION = "~"

# deps: 
# from kaggle.api.kaggle_api_extended import KaggleApi
# import zipfile

# --> kaggle.py

def download_kaggle_data(url: str, file: str, path: str, competition=False):
    """Authenticate Kaggle connection using ~/.kaggle/kaggle.json
    Download kaggle data located at url/file to path
    
        Sample URL:  ronakbadhe/chess-evaluations
        Sample File: chessData.csv
        Sample Path: /home/ml/sgchess
        
    """
    api = KaggleApi()
    api.authenticate()
    
    if competition:
        api.competition_download_file(url, file, path)
    else:
        api.dataset_download_file(url, file, path)
        
           
        #api.dataset_download(url, file, path) # deprecated by Kaggle
        
    
# deps:
# import json

# --> functions.py

def load_json(filename: str, path: str) -> object:
    """Load config settings from the given json file
    Returns the decoded python object
    """
    with open(path+filename, "r") as infile:
        data = infile.read()
    return json.loads(data)

# deps: 
# import chess

# --> chess_functions.py

def convert_fen_to_bitboard(fen: pandas.Series) -> pandas.core.series.Series:
    
    
    """Converts a fen string to a bitboard mapping
        
        Parameters
        ----------
        fen : str
            The FEN string
            
        Returns
        -------
        list
            A list of bool
    """
    
    # The bitboard mapping is going to use 1 hot encoding - where each bit
    # corresponds to a specific square, piece, and color
    
    board = chess.Board(fen)
    outlist = []
    
    
    # encode white pieces
    # in python-chess chess.WHITE = True and chess.BLACK = False
    # chess.Pawn = 1, King = 6, etc
    for i in range(1,7):
        outlist.extend(board.pieces(i, chess.WHITE).tolist())
    
    # encode castling rights for white
    
    outlist.append(board.has_castling_rights(chess.WHITE))
    outlist.append(board.has_queenside_castling_rights(chess.WHITE))
    
    # encode black pieces
    for i in range(1,7):
        outlist.extend(board.pieces(i, chess.BLACK).tolist())
    
    # encode castling rights for black
    
    outlist.append(board.has_castling_rights(chess.BLACK))
    outlist.append(board.has_queenside_castling_rights(chess.BLACK))

    return pandas.Series(outlist)

# deps
# import pandas
# --> functions.py


def loadCSV(csv='chessData.csv.zip') -> pandas.DataFrame():
    """Loads the CSV file in the script directory into a pandas DataFrame
    
    """
    return pandas.read_csv(SCRIPTLOCATION + '/' + csv)

def strip_nonnumeric_evaluations(data: pandas.DataFrame()) -> pandas.DataFrame():
    """Deletes all position data that have non numeric elements in evaluation
    and converts to int
    """
    return data[data['Evaluation'].str.contains('[^0-9][+-]', regex=True) == False]
    
def preprocess_position_data_old(data: pandas.DataFrame()) -> pandas.DataFrame():
    # Couldn't get this method working correctly
    
    
    """Handles initial data processing before handing off to sklearn"""
       
    # Evaluation data is going to have strings containing # to indicate checkmating
    #  sequences, and all kinds of other potential junk data.
    # The chess-python module has functionality that can calculate checkmates,
    # There's no need to teach the neural network these positions
    # Therefore, these are going to be removed from the dataset in this step
    
    data = strip_nonnumeric_evaluations(data)
    
    # Cast these values which should always be numeric now into the appropriate type
    data.loc[:,('Evaluation')] = data.loc[:,('Evaluation')].astype('int')
    
    ###########
    # WARNING #
    ###########
    
    # Multiplying by 1 here to convert bools to 1s and 0s
    # Pandas DataFrames don't like raw bools, so this is necessary to convert to
    #   a sparse matrix
    # However, this is a poor solution, need to find a better method utilizing 
    #   pandas.DataFrame.astype(int)
    
    data.loc[:,('FEN')] = data.loc[:,('FEN')].transform(lambda fen: convert_fen_to_bitboard(fen)) * 1
    
    
    # WIP
    # this function currently returns a bunch of lists (pandas series)
    # instead of a proper sparse matrix

    
    return data

def preprocess_position_data(data: pandas.DataFrame()) -> pandas.DataFrame():
    
    # This is incredibly slow and bad and I should feel bad
    
    # probably dataframes aren't the best way to go about this,
    # the issue is the kaggle dataframe only has one column for FEN, which
    # when encoded turns into a bunch of columns
    # I can't find a good in place way to shape the dataframe row by row,
    # Maybe ndarray is a better way
        
    outdata = pandas.DataFrame()
    for i in tqdm(data['FEN']):
        outdata = pandas.concat([outdata, convert_fen_to_bitboard(i).to_frame().T])
    
    return outdata.astype(pandas.SparseDtype('bool', False))
    
    
    
# deps: pandas, pickle -> functions.py
def save_dataframe(data: pandas.DataFrame(), filename: str, path=SCRIPTLOCATION):
    """
    Saves a dataframe directly to disk in order to avoid needing to download
    and process the CSV file each time
    """
    with open(path + '/' + filename, 'wb') as fp:
        pickle.dump(data, fp)
        print("saved to %s" % (filename))
        
# deps: pandas, pickle -> functions.py
def load_dataframe(filename: str, path=SCRIPTLOCATION) -> pandas.DataFrame():
    """
    Loads the contents of filename from disk into a pandas DataFrame
    Contents of filename must be a pandas dataframe serialized using pickle
    Save a dataframe using save_dataframe method
    """

    with open(path + '/' + filename, 'rb') as fp:
        return pickle.load(fp)
    

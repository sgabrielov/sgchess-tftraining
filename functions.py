from kaggle.api.kaggle_api_extended import KaggleApi
import json, os, sys, zipfile, chess, pandas, scipy, numpy, pickle


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

# deps: pandas
# --> functions.py


def loadCSV(csv='chessData.csv.zip') -> pandas.DataFrame():
    """Loads the CSV file in the script directory into a pandas DataFrame
    
    """
    return pandas.read_csv(SCRIPTLOCATION + '/' + csv)

def strip_nonnumeric_evaluations(data: pandas.DataFrame()) -> pandas.DataFrame():
    """Deletes all position data that have non numeric elements in evaluation
    """
    return data[data['Evaluation'].str.contains('[^0-9][+-]', regex=True) == False]
    
def preprocess_position_data(data: pandas.DataFrame()) -> pandas.DataFrame():
    """Handles initial data processing before handing off to sklearn"""
    
    # Create a new dataframe with the necessary dimensions
    # The number of features should be represented by the length of the bitboard
    #   string after conversion. Based on the current encoding strategy, this
    #   is going to equal 772
    # Need to find a good way to calculate this length in case the strategy 
    #   changes
    
    # The number of rows can be found using len(data.index)
    
    outdata = pandas.DataFrame()
    
    
    
    data = strip_nonnumeric_evaluations(data)
    data.loc[:,('Evaluation')] = data.loc[:,('Evaluation')].astype('int')
    
    data.loc[:,('FEN')] = data.loc[:,('FEN')].transform(lambda fen: convert_fen_to_bitboard(fen))
    
    
    # WIP
    # this function currently returns a bunch of lists (pandas series)
    # instead of a proper sparse matrix
    
    return data

# deps: pandas, pickle -> functions.py
def save_dataframe(data: pandas.DataFrame(), filename: str):
    """
    Saves a dataframe directly to disk in order to avoid needing to download
    and process the CSV file each time
    """
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)
        print("saved to %s" % (filename))
        
# deps: pandas, pickle -> functions.py
def load_dataframe(filename: str) -> pandas.DataFrame():
    """
    Loads the contents of filename from disk into a pandas DataFrame
    Contents of filename must be a pandas dataframe serialized using pickle
    Save a dataframe using save_dataframe method
    """
    with open(filename, 'rb') as fp:
        return pickle.load(fp)
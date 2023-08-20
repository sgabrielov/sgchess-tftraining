from kaggle.api.kaggle_api_extended import KaggleApi
import json, os, sys, zipfile, chess, pandas, scipy, numpy, pickle
from tqdm import tqdm



# This file contains definitions for all useful library functons.
# Eventually, these will be split into separate files according to which
# modules they depend on 


SCRIPTLOCATION = "/home/ml/sgchess"
# SCRIPTLOCATION = "~"

EVAL_COL_NAME = "Evaluation"
FEN_COL_NAME = "FEN"

if SCRIPTLOCATION not in sys.path:
    sys.path.append(SCRIPTLOCATION)
    
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

def convert_fen_to_bitboard(fen: pandas.core.series.Series) -> pandas.core.series.Series:
    
    
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

def strip_nonnumeric_evaluations(data: pandas.DataFrame(), key=EVAL_COL_NAME) -> pandas.DataFrame():
    """Deletes all position data that have non numeric elements in evaluation
    and converts to int
    """
    return data[data[key].str.contains('[^0-9][+-]', regex=True) == False]
    
def preprocess_position_data(data: pandas.DataFrame()) -> pandas.DataFrame():  
    
    """Handles initial data processing before handing off to sklearn"""
       

    
    # Transform the dataframe
    
    # Create a temporary list while constructing the frame
    outlist = []
    
    # tqdm adds a progress tracker to output
    
    for i in tqdm(data[FEN_COL_NAME]):
        # convert the fen string to a bitboard mapping and save it to the temp list
        outlist.append(convert_fen_to_bitboard(i))
    
    # create the output dataframe and load in the temp data
    outdata = pandas.DataFrame(outlist, index=data[FEN_COL_NAME].index)
    
    # convert the dataframe into sparse type for memory efficiency and return
    return outdata.astype(pandas.SparseDtype('bool', False))   

# def preprocess_position_data(data: pandas.DataFrame()) -> pandas.DataFrame():
    
#     # This is incredibly slow and bad and I should feel bad
    
#     # probably dataframes aren't the best way to go about this,
#     # the issue is the kaggle dataframe only has one column for FEN, which
#     # when encoded turns into a bunch of columns
#     # I can't find a good in place way to shape the dataframe row by row,
#     # Maybe ndarray is a better way
    
#     outlist = []
#     for i in tqdm(data['FEN']):
#         outlist.append(convert_fen_to_bitboard(i))
    
#     outdata = pandas.DataFrame(outlist, index=data['FEN'].index)
#     return outdata.astype(pandas.SparseDtype('bool', False))   
def cast_as(data: pandas.DataFrame(), key=EVAL_COL_NAME, casttype=int):
    """
    Change the type of a column in the dataframe

    Parameters
    ----------
    data : pandas.DataFrame()
        The dataframe
    key : str, optional
        The key of column that should be cast as casttype
    casttype : type, optional
        What type the value should be. The default is int.

    Returns
    -------
    None.

    """
    data.loc[:,(key)] = data.loc[:,(key)].astype(casttype)

def preprocess_position_data_batch(data: pandas.DataFrame(), batch_size=100_000, max_batches=10_000, save_loc=SCRIPTLOCATION, filename="dataprocessed.p"):
    """
    An optional method of handling the data preprocessing in batches
    This is necessary to process an exceptionally large ( > 10M ) dataset
    
    If the batch size is set too small, this could create a lot of files!

    Parameters
    ----------
    data : pandas.DataFrame()
        The dataframe.
    batch_size : int, optional
        The number of rows to process per batch. The default is 100_000: int.
    max_batches : int, optional
        The maximum number of batches (and therefore files) allowed. The default is 10_000: int.
    save_loc : str, optional
        The path to save to. The default is SCRIPTLOCATION: str.
    filename : str, optional
        The name of the file. The default is "dataprocessed.p":str.

    Returns
    -------
    None.

    """
    
    batches = len(data)//batch_size + 1
    if batches > max_batches:
         raise ValueError(f'batches exceeds max_batches: {batches} > {max_batches}')
    if len(data) % batch_size == 0:
        batches = batches - 1
    padding = batches**(1/10)
    for i in range(0, batches):
        
        
        dataprocessed = preprocess_position_data(data.iloc[i*batch_size:(i+1)*batch_size])
        savename = str(i).zfill(padding) + filename
        
        print("saving batch to " + save_loc + "/" +  savename)
        save_dataframe(dataprocessed,  savename, save_loc)
    
    
def load_position_data_batch(filename: str, load_loc=SCRIPTLOCATION) -> pandas.DataFrame():
    """
    Load the dataframe located in batched files that contain filename.
    """
    frames = []
    files = [f for f in os.listdir(load_loc) if filename in f]
    files.sort()
    
    for i in files:
        frames.append(load_dataframe(i, load_loc))
        
    return pandas.concat(frames)
    
    
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
    
def standardize(data: pandas.Series) -> pandas.Series:
    return (data - data.mean()) / data.std()

def destandardize(data: pandas.Series, orig: pandas.Series) -> pandas.Series:
    return data * orig.std() + orig.mean()
 
              
    
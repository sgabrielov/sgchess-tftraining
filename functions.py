from kaggle.api.kaggle_api_extended import KaggleApi
import json, os, sys, zipfile, chess, pandas, scipy, numpy, pickle
from tqdm import tqdm
import math



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
    with open(path+'/'+filename, "r") as infile:
        data = infile.read()
    return json.loads(data)

# deps: 
# import chess

# --> chess_functions.py

def convert_fen_to_bitboard(fen, cols=None) -> pandas.core.series.Series:
    
    
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

    return pandas.Series(outlist, index=cols, dtype=bool)

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
    
def preprocess_position_data_old(data: pandas.DataFrame()) -> pandas.DataFrame():  
    
    """Handles initial data processing before handing off to sklearn"""
       
    
    # Transform the dataframe
    
    # Create a temporary list while constructing the frame
    outlist = []
    
    # tqdm adds a progress tracker to output
    
    for i in tqdm(data[FEN_COL_NAME]):
        # convert the fen string to a bitboard mapping and save it to the temp list
        outlist.append(convert_fen_to_bitboard(i))
    
    # get the list of columns for the new dataframe
    with open(SCRIPTLOCATION + '/cols.json') as infile:
        cols = json.load(infile)
    
    # create the output dataframe and load in the temp data
    outdata = pandas.DataFrame(data=outlist, index=data[FEN_COL_NAME].index, columns=cols)
    
    # convert the dataframe into sparse type for memory efficiency and return
    return outdata.astype(pandas.SparseDtype('bool', False))   

# v2 of preprocess logic
# so far this is working much better
def preprocess_position_data(data: pandas.DataFrame()) -> pandas.DataFrame():
    
    # get column labels from json fie
    with open(SCRIPTLOCATION + '/cols.json') as infile:
        cols = json.load(infile)
    
    # run convert_fen_to_bitboard functions accross all rows of data, the input parameter
    # convert the resulting dataframe into sparse type
    # the columns labels of the new dataframe will be set to cols
    
    # this function can handle input in the form of either a dataframe or a series. 
    # either way it will return a dataframe
    # any other type will raise a ValueError
    if isinstance(data, pandas.core.frame.DataFrame):
        return data[FEN_COL_NAME].apply(lambda fen: convert_fen_to_bitboard(fen, cols))
    elif isinstance(data, pandas.core.series.Series):
        return data.apply(lambda fen: convert_fen_to_bitboard(fen, cols))
    else:
        raise ValueError(f'Unable to handle input data type. Expected pandas Series or DataFrame, received {type(data)}')
        
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
    
    # identify the number of batches/iterations to process/files to create
    batches = len(data)//batch_size + 1
    
    # if the number of iterations exceeds the number allowed by the user
    if batches > max_batches:
         raise ValueError(f'batches exceeds max_batches: {batches} > {max_batches}')
    
    # In order to ensure sortable filenames, pad the prefix with 0s according to
    # the length of the longest prefix
    #   -> is the # of digits of the largest prefix
    #       -> is the base 10 log of batches
    padding = math.ceil(math.log(batches, 10))
    
    # if the batch size is a perfect divisor of the number of data samples
    if len(data) % batch_size == 0:
        batches = batches - 1
    
    # start generating the files
    print("Writing files to " + save_loc + "/")
    print(f'# of records:\t{len(data)}')
    print(f'# of batches:\t{batches}')
    print(f'first filename:\t{str(0).zfill(padding) + filename}')
    print(f'last filename:\t{str(batches).zfill(padding) + filename}')
    
    for i in range(0, batches):
        
        # preprocess each batch
        dataprocessed = preprocess_position_data(data.iloc[i*batch_size:(i+1)*batch_size])
        
        # generate the filename to save to
        savename = str(i).zfill(padding) + filename
        
        # print the save file to the user
        print("saving batch to " + save_loc + "/" +  savename)
        
        # save the batches data to the file
        save_dataframe(dataprocessed,  savename, save_loc)
    
    
def load_position_data_batch(filename: str, load_loc=SCRIPTLOCATION) -> pandas.DataFrame():
    """
    Load the dataframe located in batched files that contain filename.
    """
    frames = []
    # get all the files in load_loc that contain filename
    # there is no mechanism to prevent unwanted/junk files from getting loaded this way
    # the safest way to use this function is to make dedicate directories to save 
    #   to when using save_dataframe
    # and to only attempt to load from these directories
    files = [f for f in os.listdir(load_loc) if filename in f]
    
    # os.listdir returns files out of order. sorting the file list will ensure
    # the indices of the data will be in the same order as when they were saved
    files.sort()
    
    # build a list of partial dataframes
    for i in files:
        frames.append(load_dataframe(i, load_loc))
    
    # concatenate the list of partial dataframes into one and return
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
 
              
    
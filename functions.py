from kaggle.api.kaggle_api_extended import KaggleApi
import json, os, sys, zipfile, chess, pandas

SCRIPTLOCATION = "/home/ml/sgchess"
# SCRIPTLOCATION = "~"

# deps: 
# from kaggle.api.kaggle_api_extended import KaggleApi
# import zipfile

# --> kaggle.py

def download_kaggle_data(url: str, file: str, path: str, competition=False):
    """Authenticate Kaggle connection using ~/.kaggle/kaggle.json
    Download kaggle data located at url/file to path
    
    """
    api = KaggleApi()
    api.authenticate()
    
    if competition:
        api.competition_download_file(url, file, path)
    else:
        #api.dataset_download_file(url, file, path, unzip=True)
        api.dataset_download(url, file, path)
        
    
# deps:
# import json

# --> functions.py

def load_kaggle_dataset_json(filename: str, path: str) -> object:
    """Load config settings from the given json file
    Returns the decoded python object
    """
    with open(path+filename, "r") as infile:
        data = infile.read()
    return json.loads(data)

# deps: 
# import chess

# --> chess_functions.py

def convert_fen_to_bitboard(fen: str) -> list:
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
    
    
    return outlist

def loadCSV(csv='chessData.csv.zip') -> pandas.DataFrame():
    return pandas.read_csv(SCRIPTLOCATION + '/' + csv)

def strip_checkmating_positions(data: pandas.DataFrame()) -> pandas.DataFrame():
    return data[data['Evaluation'].str.contains('#') == False]
    
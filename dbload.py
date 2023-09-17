from functions import loadCSV, load_dataframe, load_position_data_batch, preprocess_position_data, strip_nonnumeric_evaluations, cast_as
import pandas
import mysql.connector
import json
import sys, os
from tqdm import tqdm

from sqlalchemy import create_engine

SCRIPTLOCATION = '/home/ml/sgchess/'
if SCRIPTLOCATION not in sys.path:
    sys.path.append(SCRIPTLOCATION)

with open(SCRIPTLOCATION + 'mysql_connect.json', 'r') as infile:
    conn_settings = json.load(infile)

def store_positions_in_db():
	
    #df = pandas.DataFrame({'name' : ['User 1', 'User 2', 'User 3']})
    
    positions=load_position_data_batch('dataprocessed.p', SCRIPTLOCATION + '/data_processed/')
    
    
    #df = pandas.concat([positions, evals], axis=1)
    
    write_to_db(positions, 'db_positions', **conn_settings)
    
    

def store_evals_in_db():
    evals = load_dataframe('evals.p')
    
    write_to_db(evals, 'db_evals', **conn_settings)
    
    
def write_to_db(data: pandas.DataFrame(), table, user, password, host, database, **conn):
    DATABASE_URI='mysql+mysqlconnector://{u}:{pw}@{s}/{db}'.format(
        u=user, 
        pw=password, 
        s=host, 
        db=database)
    print("initializing engine")
    eng = create_engine(DATABASE_URI)
    print("writing to db")
    data.to_sql(table, eng, if_exists='append', chunksize=100)
    print("done")
    eng.dispose()
    
def write_to_db_batched_from_file(filename: str, table, user, password, host, database, load_loc=SCRIPTLOCATION, **conn):

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
    for i in tqdm(files):
        write_to_db(load_dataframe(i, load_loc), table, user, password, host, database)

    
def main():
    write_to_db_batched_from_file('dataprocessed.p', 'db_positions', **conn_settings, load_loc=SCRIPTLOCATION +'data_processed/')

if __name__ == '__main__':
	main()

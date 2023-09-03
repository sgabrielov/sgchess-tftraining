# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class MySQLGenerator(tf.keras.utils.Sequence):
    def __init__(self, sql_connection, table:str, index_col:str, X_cols:list, y_cols:list, min_row:int, max_row:int, batch_size:int, shuffle=True):
        """
        This class is a generator designed to pull information from a MySQL database
        and feed it in batches to a tensorflow model

        Parameters
        ----------
        sql_connection : mysql.connector.connection.MySQLConnection
            An active connection to a MySQL database.
            Use mysql.connector.MySQLConnection(...)
        table : str
            The name of the table or view to pull data from.
        index_col : str
            The name of the primary key column.
        X_cols : list
            A list of feature columns.
        y_cols : list
            A list of target columns.
        min_row : int
            The starting index of desired data.
        max_row : int
            The final index of desired data.
        batch_size : int
            The number of rows to return for each call to the generator.
        shuffle : TYPE, optional
            Whether to shuffle the data upon retrieval. The default is True.
        Returns
        -------
        None.

        """
        
        # initialize class attributes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_row = min_row
        self.max_row = max_row
        self.sql_connection = sql_connection
        self.X_cols = X_cols
        self.y_cols = y_cols
        self.table = table
        self.index_col = index_col        
        self.n = self.__get_count()
        self.on_epoch_end()
    def __get_count(self):
        
        cursor = self.sql_connection.cursor(buffered = True)        
        query = f'SELECT COUNT(`{self.index_col}`) FROM {self.table} WHERE `{self.index_col}` BETWEEN {self.min_row} AND {self.max_row};'
        cursor.execute(query)
        
        result = cursor.fetchall()
        
        cursor.close()
        
        return result[0][0]
    
    def on_epoch_end(self):
        """
        Retrieves indices from database and shuffles if Shuffle=True

        Returns
        -------
        None.

        """
        
        # Create a cursor
        cursor = self.sql_connection.cursor(buffered = True)
        
        # Construct the query
        query = f'SELECT `{self.index_col}` FROM `{self.table}` WHERE `{self.index_col}` BETWEEN {self.min_row} AND {self.max_row} LIMIT {self.n};'
        
        cursor.execute(query)
        
        # Get indices from cursor
        self.indices = [i[0] for i in cursor.fetchall()]
        
        # Shuffle if necessary
        if(self.shuffle):
            np.random.shuffle(self.indices)
        
        # Close the cursor
        cursor.close()
    
    def __getitem__(self, index):
        
        # Get a list of indices corresponding to the requested data
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        # Create a cursor
        cursor = self.sql_connection.cursor(buffered = True)
        
        # Construct and execute a query to retrieve feature data
        X_query = f'SELECT {",".join(self.X_cols)} FROM `{self.table}` WHERE `{self.index_col}` IN({str(indices)[1:-1]}) LIMIT {self.batch_size};'
        cursor.execute(X_query)
        
        # Retrieve data from the cursor
        X = np.array(cursor.fetchall())
        
        # Construct and execute a query to retrieve target data
        y_query = f'SELECT {",".join(self.y_cols)} FROM `{self.table}` WHERE `{self.index_col}` IN({str(indices)[1:-1]}) LIMIT {self.batch_size};'
        cursor.execute(y_query)
        
        # Retrieve data from the cursor
        y = np.array(cursor.fetchall())
        
        # Close the cursor
        cursor.close()
        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size
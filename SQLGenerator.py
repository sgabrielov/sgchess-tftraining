# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 09:26:08 2023

@author: osiri
"""

import tensorflow as tf
import numpy as np

class SQLGenerator(tf.keras.utils.Sequence):
    def __init__(self, sql_connection, table:str, X_cols:list, y_cols:list, min_row:int, max_row:int, batch_size:int, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_row = min_row
        self.max_row = max_row
        self.n = max_row - min_row
        self.X_cols = X_cols
        self.y_cols = y_cols
        self.table = table
        
        self.sql_connection = sql_connection
        
        
        
        self.on_epoch_end()
        
    def on_epoch_end(self):
        cursor = self.sql_connection.cursor()
        query = "SELECT p.index FROM " + self.table + " AS p WHERE p.index BETWEEN " + str(self.min_row) + " AND " + str(self.max_row) + " LIMIT " + str(self.n) + ";"
        cursor.execute(query)
        self.indices = [i[0] for i in cursor.fetchall()]
        
        if(self.shuffle):
            np.random.shuffle(self.indices)
            
        cursor.close()
    
    def __getitem__(self, index):
        
        
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        cursor = self.sql_connection.cursor(buffered = True)
        X_query = "SELECT " + ",".join(self.cols) + " FROM " + self.position_table + " AS p WHERE p.index IN(" + str(indices)[1:-1] + ") LIMIT " + str(self.batch_size) + ";"
        y_query = "SELECT e.Evaluation FROM " + self.eval_table + " AS e WHERE e.index IN(" +  str(indices)[1:-1] + ") LIMIT " + str(self.batch_size) + ";"
        
        cursor.execute(X_query)
        X = np.array(cursor.fetchall())
        cursor.execute(y_query)
        y = np.array(cursor.fetchall())
        
        
        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size
    
import pandas as pd
import numpy as np
import pymssql

class Data: 

    def __init__(self):
        pass

    def fromCSV(self, way):
        self.table = pd.read_csv(way)
        self.matrix = self.table.pivot(index = 'userId', columns = 'productId', values = 'marketValue')
        self.max = self.matrix.max(axis = 1)
        self.min = self.matrix.min(axis = 1)

    def fromSQL(self, server, username, password, database, query):
        conn = pymssql.connect(server = server, user = username, password= password, database = database)
        self.table = pd.read_sql_query(query, conn)
        self.matrix = self.table.pivot(index = 'userId', columns = 'productId', values = 'marketValue')
        self.max = self.matrix.max(axis = 1)
        self.min = self.matrix.min(axis = 1)
        print(self.matrix)
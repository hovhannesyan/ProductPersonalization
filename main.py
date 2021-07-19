import os
from src.models.Data import Data
from src.predictionModels.CollaborativeFiltering import CollaborativeFiltering

rootDir = os.path.dirname(__file__)

products = Data()

products.fromSQL('server address', 'username', 'password', 'database', '''query''')


#products.fromCSV(os.path.join(rootDir, 'testCSVs', 'test1.csv'))

predictions = CollaborativeFiltering(products)
predictions.train(1000)
predictions.getPredictions()
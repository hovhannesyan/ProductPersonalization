# Product Personalization algorithm
## Description
This algorithm based on [Collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) and written in [PyTorch](https://pytorch.org) framework  
Whole algorithm is in [CollaborativeFiltering.py](https://github.com/hovhannesyan/ProductPersonalization/blob/master/src/predictionModels/CollaborativeFiltering.py) file  
As result the console will write:
```
Recommend user {x} product {y}
```
## Modules installation
1. PyTorch: 
```
pip install pytorch
```
2. NumPy: 
```
pip install numpy
```
3. Pandas: 
```
pip install pandas
```
4. pymssql: 
```
pip install pymsssql
```
## How to use
After cloning this repository you should configure [main.py](ProductPersonalization/main.py) file:
1. Set MSSQL connection and SQL query
```
products.fromSQL('server address', 'username', 'password', 'database', '''SQL query''')
```
> In SQL query you must get table with 3 coloumns named:
> * userId
> * productId
> * marketValue
2. Just run [main.py](https://github.com/hovhannesyan/ProductPersonalization/blob/master/main.py) file
```
python main.py
```
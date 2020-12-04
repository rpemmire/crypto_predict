# crypto_predict
Final Project for Brown Deep Learning (CS1470)

Our project was a structured data classification problem, using blockchain data to build
transaction networks and then forecasting transactions between accounts.

-This repository contains a code and data folder. In pre-pre-processing, we used the script
getfirst100k.py on blockchain data from https://senseable2015-6.mit.edu/bitcoin/ to turn our
data into weighted edges format.

-We then used Node2vec.py to learn and output embeddings and graph structures for each of our
10 graphs. These were stored in Data/Node2Vec_outputs. Predict.py contains the model which can
then take in the saved node embeddings and learn to forecast transactions between 2 accounts.

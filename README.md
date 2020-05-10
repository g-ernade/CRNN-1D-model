# CRNN-1D model
 relevant code for CRNN-1D model

### data

Data folder contains training data and testing data mentioned in the paper

### models

Models is a module imported in `train.py`. This module includes models in comparisons with CRNN-1D and CRNN-1D itself.

### evalu

Evlau is a module imported in `evaluation.py`. This module includes various evaluation methods adopted in the paper.

### training_methods

Training_methods is a module imported in `train.py`. This module includes the main training method including splitting the data used for training into training set and validation set.

### trained_file

Trained_file is a folder that contains all the hdf5 model file we trained on the server.

### evaluation.py

 `evaluation.py` is a python script we used for result analysis

### train.py

`train.py` is a a python script we used for training different models.

### dic_test.pkl

`dic_test.pkl` is a packed binary file for a python dict that is used for token transition for all the models. This file is compressed using pickle module in python.
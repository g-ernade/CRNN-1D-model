from evalu.Evalu import *


model_dict={'cmu':'./trained_file/cmu.hdf5',
            'CRNN-1D':'./trained_file/CRNN-1D.hdf5',
            'mit':'./trained_file/mit.hdf5'
            ,'nyu':'./trained_file/nyu.hdf5',
            'endgame':'./trained_file/endgame9.hdf5'}#'model name':'model file path'
black_path='./data/dga_test_10w.csv'
white_path='./data/alexa_10w.csv'
dict_path='dic_test.pkl'
run(model_dict,black_path,white_path,dict_path)
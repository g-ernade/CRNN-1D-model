from training_methods.Training_Methods import *
from models.Models import *

data_path_black = './data/dga_train_90w.csv'
data_path_white = './data/alexa_90w.csv'
data_white = pd.read_csv(data_path_white)
data_black = pd.read_csv(data_path_black)
data = pd.concat([data_black, data_white], axis=0, ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)

with open('dic_test.pkl', 'rb') as f_ptr:
    dic_for_encrypt = pickle.load(f_ptr)

batch_size=64
epoches=30

train_validation(crnn1d,data,dic_for_encrypt,epoches,batch_size)

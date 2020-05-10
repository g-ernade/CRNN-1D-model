import pickle
import time
from tensorflow.python.keras.preprocessing import sequence
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import roc_curve, auc
from tensorflow.python.keras.callbacks import ModelCheckpoint,TensorBoard
import os
import tensorflow as tf
from datetime import datetime

def dns_encryptor(dic, dns, max_len=75):  # 75 equals max length
    '''
    Encrypt dga domains into vectors
    :param path_dic: python dict for encryption
    :param dns: dga data to be encrypted
    :return: np.array([[],[],[]...])
    '''
    print('>>>encrypting data with maxlen=', max_len)

    vaild_chars_dic = dic
    max_len = max_len

    dns_encrypt = []
    for x in dns:
        tmp = []
        for y in x:
            tmp.append(vaild_chars_dic.get(y, 0))
        dns_encrypt.append(tmp)
    out = sequence.pad_sequences(dns_encrypt, max_len)
    print(type(out))
    return out


def train_validation(model_func, data, dic_for_encrypt: dict, epochs: int, \
                  batch_size: int, max_len: int = 75) -> None:
    # training results are saved to ./results/modelnames

    if not os.path.exists("./results"):
        os.mkdir('./results')


    np.random.seed(42)
    model_name = input('input model names ')

    if not os.path.exists("./results"+'/'+model_name):
        os.mkdir("./results"+'/'+model_name)
    model_container_path="./results"+'/'+model_name+'/'


    time_str = str(time.asctime(time.localtime(time.time()))).replace(':', '_')  # begin time for training
    dns = [x['Domain'] for index, x in data.iterrows()]  # Read domain string
    label = [x['Label'] for index, x in data.iterrows()]  # Read labels (1 for black and 0 for white
    max_features = len(dic_for_encrypt) + 1  # Extra 1 for padding
    dns_encrypted = dns_encryptor(dic_for_encrypt, dns, max_len)  # Encrypting dns
    model = model_func(max_features, max_len, use_gap=True)

    cpt_file_path=os.path.join(model_container_path,model_name+'_'+time_str+'{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpoint=ModelCheckpoint(cpt_file_path,monitor='val_acc',verbose=1)#callback0

    TensorBoardcallback = tf.keras.callbacks.TensorBoard(log_dir='./logs',
            histogram_freq=1, batch_size=32,
            write_graph=True, write_grads=False, write_images=True,
            embeddings_freq=0, embeddings_layer_names=None,
            embeddings_metadata=None, embeddings_data=None, update_freq=500)

    cbk_ls=[checkpoint,TensorBoardcallback] #callback list
    dns_train=np.array(dns_encrypted)
    label_train=np.array(label)

    print('>>>train begin')
    start_time=time.perf_counter()
    history=model.fit(dns_train,label_train,batch_size=batch_size,epochs=epochs,validation_split=0.2,\
              callbacks=cbk_ls)
    end_time=time.perf_counter()
    train_time=end_time-start_time
    print('>>>training time is{}'.format(train_time))




    file_path=os.path.join(model_container_path,model_name+' '+time_str+'.txt')

    with open(file_path,'w') as fptr:
        fptr.write(model_name+'training time:'+str(train_time)+'s')
    return None


def import_test_tm():
    print('you have imported Training_Methods successfully:)')

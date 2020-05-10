import numpy as np
import pandas as pd
from tensorflow.python.keras.models import load_model


from tensorflow.python.keras.layers import Dense,Embedding, BatchNormalization, \
     Activation, Input
import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
import pickle
from sklearn.metrics import roc_curve,auc,accuracy_score,\
    f1_score,precision_score,recall_score,confusion_matrix
import time
from matplotlib import pyplot as plt


def hdf5_cracker(modelpath:str):
    try:
        model = load_model(modelpath)
        print(model.summary())
        return model
    except:
        print('>>>failed to read the model')
        return -1

def dns_encryptor(dic:dict, dns, max_len=75):  # 75是最大序列长度
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

def make_test_df(black_sample_path:str,white_sample_path:str):
    df_b=pd.read_csv(black_sample_path)
    df_w=pd.read_csv(white_sample_path)
    df=pd.concat([df_b,df_w],axis=0, ignore_index=True)
    return df

def main(model_name,model_path,black_path,white_path,dict_path,FPR_threshold=0.001):
    model=hdf5_cracker(model_path)
    if model == -1:
        return
    test_data=make_test_df(black_path,white_path)
    dns = [x['Domain'] for index, x in test_data.iterrows()]  # read domain string
    label = [x['Label'] for index, x in test_data.iterrows()]  # read labels(1 for white and 0 for black
    label=np.array(label)#convert from python list to numpy array
    with open(dict_path, 'rb') as fptr:
        dic_for_encrypt = pickle.load(fptr)  # unpickle the dict
    dns_encrypted = dns_encryptor(dic_for_encrypt, dns, max_len=75)
    start_time = time.time()
    predict = model.predict(dns_encrypted,verbose=1)
    end_time = time.time()
    print('\n', model_name, ':')
    predicting_time=end_time-start_time
    print('!!!predicting_time= ', predicting_time)

    fpr, tpr, thresholds = roc_curve(label, predict)
    #cal auc@1%
    integral = 0
    i=0
    while tpr[i+1] <= 0.99:
        integral += (tpr[i] + tpr[i + 1]) * (fpr[i + 1] - fpr[i]) / 2  # compute the integral
        i += 1
    FPR=fpr[i];TPR=tpr[i]
    print(">>>AUC@1%= ",integral*100)
    #control fpr at 0.001 and calculate, f1, rercall, accuracy, precision, auc
    i=0
    while fpr[i]<FPR_threshold:
        i+=1
    print('>>>FPR= ',fpr[i],';TPR= ',tpr[i])
    auc_=auc(fpr,tpr)
    accuracy_score_=accuracy_score(label,predict>thresholds[i])

    confusion=confusion_matrix(label,predict>thresholds[i])
    TP = confusion[0, 0]
    FN = confusion[0, 1]
    FP = confusion[1, 0]
    TN = confusion[1, 1]
    i = 0
    while fpr[i+1] < 0.0001:
        i += 1
    print('>>>AUC= ', auc_,';accuracy= ',accuracy_score_\
          ,'\nTP=',TP,';FN=',FN,';FP=',FP,';TN=',TN)
    print('\n')
    out_dict={'name':model_name,'auc':auc_,'accuracy':accuracy_score_,'TP':TP,'FN':FN,'FP':FP,'TN':TN,'FPR':FPR,\
              'TPR':TPR,'time':predicting_time,'auc@1%':integral*100}
    return fpr,tpr,out_dict

def run(modeldict,black_path,white_path,dict_path):
    results=[]
    for name,path in modeldict.items():
        x,y,dic=main(name,path,black_path,white_path,dict_path)
        results.append(dic)
        plt.semilogx(x,y,label=name)
    plt.legend(fontsize='large')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('aucsemilogx.png')
    plt.show()
    with open('results of all models.txt','w+') as fptr:
        for item in results:
            fptr.write(str(item))
            print(item)
            fptr.write('\n')







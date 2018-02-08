import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np
import sys
from PIL import Image
import io
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json
from keras.preprocessing import sequence
from scipy import interp
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

basePath="/home/bkcs/Desktop/dung/v2/"
eth_Close_model_id = 1
eth_Volume_model_id = 2
eth_High_model_id =3
eth_Low_model_id = 4
bt_Close_model_id = 5
bt_Volume_model_id = 6
bt_High_model_id = 7
bt_Low_model_id= 8
class BtcPredict():
    def _init_(self):
        print("This is bitcoin predict!")
    #Load Model
    def loadModel(self,model,weight,id): 
        # load json and create model
        json_file = open(model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        if(id == eth_Close_model_id):
            self.eth_Close_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.eth_Close_model.load_weights(weight) 
        elif(id==eth_Volume_model_id):
            self.eth_Volume_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.eth_Volume_model.load_weights(weight) 
        elif(id==eth_High_model_id):
            self.eth_High_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.eth_High_model.load_weights(weight) 
        elif(id==eth_Low_model_id):
            self.eth_Low_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.eth_Low_model.load_weights(weight)
        elif(id== bt_Close_model_id):
            self.bt_Close_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.bt_Close_model.load_weights(weight) 
        elif(id== bt_Volume_model_id):
            self.bt_Volume_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.bt_Volume_model.load_weights(weight) 
        # print("Loaded model from disk")
        elif(id== bt_High_model_id):
            self.bt_High_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.bt_High_model.load_weights(weight) 
        # print("Loaded model from disk")
        elif(id== bt_Low_model_id):
            self.bt_Low_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.bt_Low_model.load_weights(weight) 
        # print("Loaded model from disk")
    def predict(self,data,id):
        if(id == eth_Close_model_id):
            return self.eth_Close_model.predict(data)
        elif(id==eth_Volume_model_id):
            return self.eth_Volume_model.predict(data)
        elif(id==eth_High_model_id):
            return self.eth_High_model.predict(data)
        elif(id==eth_Low_model_id):
            return self.eth_Low_model.predict(data)
        elif(id==bt_Close_model_id):
            return self.bt_Close_model.predict(data)
        elif(id==bt_Volume_model_id):
            return self.bt_Volume_model.predict(data)
        elif(id==bt_High_model_id):
            return self.bt_High_model.predict(data)
        elif(id==bt_Low_model_id):
            return self.bt_Low_model.predict(data)

if __name__ == "__main__":
    eth_Close_model = basePath+"eth_Close_model.json"
    eth_Close_model_weight = basePath+"eth_Close_model.h5"

    eth_Volume_model = basePath+"eth_Volume_model.json"
    eth_Volume_model_weight = basePath+"eth_Volume_model.h5"


    eth_High_model = basePath+"eth_High_model.json"
    eth_High_model_weight = basePath+"eth_High_model.h5"

    eth_Low_model = basePath+"eth_Low_model.json"
    eth_Low_model_weight = basePath+"eth_Low_model.h5"

    bt_Close_model = basePath+"bt_Close_model.json"
    bt_Close_model_weight = basePath+"bt_Close_model.h5"

    bt_Volume_model = basePath+"bt_Volume_model.json"
    bt_Volume_model_weight = basePath+"bt_Volume_model.h5"

    bt_High_model = basePath+"bt_High_model.json"
    bt_High_model_weight = basePath+"bt_High_model.h5"

    bt_Low_model = basePath+"bt_Low_model.json"
    bt_Low_model_weight = basePath+"bt_Low_model.h5"



    btc_predict =BtcPredict()

    btc_predict.loadModel(eth_Close_model,eth_Close_model_weight,eth_Close_model_id)
    btc_predict.loadModel(eth_Volume_model,eth_Volume_model_weight,eth_Volume_model_id)
    btc_predict.loadModel(eth_High_model,eth_High_model_weight,eth_High_model_id)
    btc_predict.loadModel(eth_Low_model,eth_Low_model_weight,eth_Low_model_id)
    btc_predict.loadModel(bt_Close_model,bt_Close_model_weight,bt_Close_model_id)
    btc_predict.loadModel(bt_Volume_model,bt_Volume_model_weight,bt_Volume_model_id)
    btc_predict.loadModel(bt_High_model,bt_High_model_weight,bt_High_model_id)
    btc_predict.loadModel(bt_Low_model,bt_Low_model_weight,bt_Low_model_id)

    data = pd.read_csv("lastdata.csv")
    data = np.array(data)
    data.shape = (1,20,8)
    # print(data)
    window_len = 20
    predict_date=pd.read_csv('date.csv')

    test_set = pd.read_csv("test_set.csv")

    # print(test_set['bt_Close'].values[:-window_len])
    predict = []
    predict_price=[]
    i=0
    while i<171:
        eth_Close_predict = btc_predict.predict(data,eth_Close_model_id)
        eth_Volume_predict = btc_predict.predict(data,eth_Volume_model_id)
        eth_High_predict = btc_predict.predict(data,eth_High_model_id)
        eth_Low_predict = btc_predict.predict(data,eth_Low_model_id)
        bt_Close_predict = btc_predict.predict(data,bt_Close_model_id)
        bt_Volume_predict = btc_predict.predict(data,bt_Volume_model_id)
        bt_High_predict = btc_predict.predict(data,bt_High_model_id)
        bt_Low_predict = btc_predict.predict(data,bt_Low_model_id)
        # bt_volatility_predict = btc_predict.predict(data,bt_volatility_model_id)
        bt_price_predict = ((np.transpose(bt_Close_predict)+0.9) * test_set['bt_Close'].values[:-window_len])[0][-1]
        predict.append([bt_Close_predict[0][0]])
        predict_price.append(bt_price_predict)
        # test_set['bt_Close'].values[:-window_len][-1] = bt_price_predict
        data= np.delete(data[0], 0,0)
        last_row = np.array([eth_Close_predict[0][0], eth_Volume_predict[0][0]
                            , eth_High_predict[0][0] , eth_Low_predict[0][0]
                            , bt_Close_predict[0][0] , bt_Volume_predict[0][0] 
                            , bt_High_predict[0][0] , bt_Low_predict[0][0] ])
        data = np.concatenate((data, [last_row]), axis=0)
        data.shape = (1,20,8)
        i = i+1
    print(predict)

    print(predict_price)


    
    # print()

    fig, ax1 = plt.subplots(1,1)
    ax1.set_xticks([datetime.date(2018,i+1,1) for i in range(12)])
    ax1.set_xticklabels([datetime.date(2018,i+1,1).strftime('%b %d %Y')  for i in range(12)])
    # ax1.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime),
    #          test_set['eth_Close'][window_len:], label='Actual')
    ax1.plot(predict_date.head(171)['date'].astype(datetime.datetime),
             predict_price, label='Predicted')
    ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(predict)+1)-\
                (test_set['eth_Close'].values[window_len:])/(test_set['eth_Close'].values[:-window_len]))), 
                 xy=(0.75, 0.9),  xycoords='axes fraction',
                xytext=(0.75, 0.9), textcoords='axes fraction')
    ax1.set_title('Test Set: Single Timepoint Prediction',fontsize=13)
    ax1.set_ylabel('Ethereum Price ($)',fontsize=12)
    ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
    plt.show()

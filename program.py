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

basePath="/home/bkcs/Desktop/dung/"
eth_Close_model_id = 1
eth_Volume_model_id = 2
eth_close_off_high_model_id =3
eth_volatility_model_id = 4
bt_Close_model_id = 5
bt_Volume_model_id = 6
bt_close_off_high_model_id = 7
bt_volatility_model_id = 8
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
        elif(id==eth_close_off_high_model_id):
            self.eth_close_off_high_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.eth_close_off_high_model.load_weights(weight)
        elif(id== eth_volatility_model_id):
            self.eth_volatility_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.eth_volatility_model.load_weights(weight) 
        elif(id== bt_Close_model_id):
            self.bt_Close_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.bt_Close_model.load_weights(weight) 
        elif(id== bt_Volume_model_id):
            self.bt_Volume_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.bt_Volume_model.load_weights(weight) 
        elif(id==bt_close_off_high_model_id):
            self.bt_close_off_high_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.bt_close_off_high_model.load_weights(weight)
        elif(id== bt_volatility_model_id):
            self.bt_volatility_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.bt_volatility_model.load_weights(weight) 

        # print("Loaded model from disk")
    def predict(self,data,id):
        if(id == eth_Close_model_id):
            return self.eth_Close_model.predict(data)
        elif(id==eth_Volume_model_id):
            return self.eth_Volume_model.predict(data)
        elif(id==eth_close_off_high_model_id):
            return self.eth_close_off_high_model.predict(data)
        elif(id==eth_volatility_model_id):
            return self.eth_volatility_model.predict(data)
        elif(id==bt_Close_model_id):
            return self.bt_Close_model.predict(data)
        elif(id==bt_Volume_model_id):
            return self.bt_Volume_model.predict(data)
        elif(id==bt_close_off_high_model_id):
            return self.bt_close_off_high_model.predict(data)
        elif(id==bt_volatility_model_id):
            return self.bt_volatility_model.predict(data)

if __name__ == "__main__":
    eth_Close_model = basePath+"eth_Close_model.json"
    eth_Close_model_weight = basePath+"eth_Close_model.h5"

    eth_Volume_model = basePath+"eth_Volume_model.json"
    eth_Volume_model_weight = basePath+"eth_Volume_model.h5"

    eth_close_off_high_model = basePath+"eth_close_off_high_model.json"
    eth_close_off_high_model_weight = basePath+"eth_close_off_high_model.h5"

    eth_volatility_model = basePath+"eth_volatility_model.json"
    eth_volatility_model_weight = basePath+"eth_volatility_model.h5"

    bt_Close_model = basePath+"bt_Close_model.json"
    bt_Close_model_weight = basePath+"bt_Close_model.h5"

    bt_Volume_model = basePath+"bt_Volume_model.json"
    bt_Volume_model_weight = basePath+"bt_Volume_model.h5"

    bt_close_off_high_model = basePath+"bt_close_off_high_model.json"
    bt_close_off_high_model_weight = basePath+"bt_close_off_high_model.h5"

    bt_volatility_model = basePath+"bt_volatility_model.json"
    bt_volatility_model_weight = basePath+"bt_volatility_model.h5"

    btc_predict =BtcPredict()

    btc_predict.loadModel(eth_Close_model,eth_Close_model_weight,eth_Close_model_id)
    btc_predict.loadModel(eth_Volume_model,eth_Volume_model_weight,eth_Volume_model_id)
    btc_predict.loadModel(eth_close_off_high_model,eth_close_off_high_model_weight,eth_close_off_high_model_id)
    btc_predict.loadModel(eth_volatility_model,eth_volatility_model_weight,eth_volatility_model_id)
    btc_predict.loadModel(bt_Close_model,bt_Close_model_weight,bt_Close_model_id)
    btc_predict.loadModel(bt_Volume_model,bt_Volume_model_weight,bt_Volume_model_id)
    btc_predict.loadModel(bt_close_off_high_model,bt_close_off_high_model_weight,bt_close_off_high_model_id)
    btc_predict.loadModel(bt_volatility_model,bt_volatility_model_weight,bt_volatility_model_id)

    data = pd.read_csv("lastdata.csv")
    data = np.array(data)
    data.shape = (1,10,8)
    predict = []
    i=0
    while i<150:
        eth_Close_predict = btc_predict.predict(data,eth_Close_model_id)
        eth_Volume_predict = btc_predict.predict(data,eth_Volume_model_id)
        eth_close_off_high_predict = btc_predict.predict(data,eth_close_off_high_model_id)
        eth_volatility_predict = btc_predict.predict(data,eth_volatility_model_id)
        bt_Close_predict = btc_predict.predict(data,bt_Close_model_id)
        bt_Volume_predict = btc_predict.predict(data,bt_Volume_model_id)
        bt_close_off_high_predict = btc_predict.predict(data,bt_close_off_high_model_id)
        bt_volatility_predict = btc_predict.predict(data,bt_volatility_model_id)

        predict.append([eth_Close_predict[0][0]])

        data= np.delete(data[0], 0,0)
        last_row = np.array([eth_Close_predict[0][0], eth_Volume_predict[0][0] 
                            , eth_close_off_high_predict[0][0] , eth_volatility_predict[0][0] 
                            , bt_Close_predict[0][0] , bt_Volume_predict[0][0] 
                            , bt_close_off_high_predict[0][0] , bt_volatility_predict[0][0] ])
        data = np.concatenate((data, [last_row]), axis=0)
        data.shape = (1,10,8)
        i = i+1
    print(predict)
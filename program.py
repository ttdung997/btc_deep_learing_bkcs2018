
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

class BitcoinPredict():
    def _init_(self):
        print("I am a Bitcoin predict!")
        
    def loadModel(self,model,weight): 
        # load json and create model
        json_file = open(model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model_binary = model_from_json(loaded_model_json)
        # load weights into new model
        self.model_binary.load_weights(weight)
        print("Loaded model from disk")
        
model = "Model/model_binary.json"
model_weight = "Model/model_binary.h5"
BitcoinPredict = BitcoinPredict()
BitcoinPredict.loadModel(model,model_weight)
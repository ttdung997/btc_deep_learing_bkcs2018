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


def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]
# convert the date string to the correct date format
# bitcoin_market_info.to_csv("btc.csv")
bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))
# when Volume is equal to '-' convert it to 0
bitcoin_market_info.loc[bitcoin_market_info['Volume']=="-",'Volume']=0
# convert to int
bitcoin_market_info['Volume'] = bitcoin_market_info['Volume'].astype('int64')
# look at the first few rows
bitcoin_market_info.head()
# print(bitcoin_market_info.head())

# get market info for ethereum from the start of 2016 to the current day
eth_market_info = pd.read_html("https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]
# convert the date string to the correct date format
eth_market_info = eth_market_info.assign(Date=pd.to_datetime(eth_market_info['Date']))
# look at the first few rows
eth_market_info.head()
# eth_market_info.to_csv("eth.csv")

if sys.version_info[0] < 3:
    import urllib2 as urllib
    bt_img = urllib.urlopen("http://logok.org/wp-content/uploads/2016/10/Bitcoin-Logo-640x480.png")
    eth_img = urllib.urlopen("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Ethereum_logo_2014.svg/256px-Ethereum_logo_2014.svg.png")
else:
    import urllib
    bt_img = urllib.request.urlopen("http://logok.org/wp-content/uploads/2016/10/Bitcoin-Logo-640x480.png")
    eth_img = urllib.request.urlopen("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Ethereum_logo_2014.svg/256px-Ethereum_logo_2014.svg.png")

image_file = io.BytesIO(bt_img.read())
bitcoin_im = Image.open(image_file)

image_file = io.BytesIO(eth_img.read())
eth_im = Image.open(image_file)
width_eth_im , height_eth_im  = eth_im.size
eth_im = eth_im.resize((int(eth_im.size[0]*0.8), int(eth_im.size[1]*0.8)), Image.ANTIALIAS)

bitcoin_market_info.columns =[bitcoin_market_info.columns[0]]+['bt_'+i for i in bitcoin_market_info.columns[1:]]
eth_market_info.columns =[eth_market_info.columns[0]]+['eth_'+i for i in eth_market_info.columns[1:]]

market_info = pd.merge(bitcoin_market_info,eth_market_info, on=['Date'])
market_info = market_info[market_info['Date']>='2016-01-01']
for coins in ['bt_', 'eth_']: 
    kwargs = { coins+'day_diff': lambda x: (x[coins+'Close']-x[coins+'Open'])/x[coins+'Open']}
    market_info = market_info.assign(**kwargs)
market_info.head()

split_date = '2017-12-01'

for coins in ['bt_', 'eth_']: 
    kwargs = { coins+'close_off_high': lambda x: 2*(x[coins+'High']- x[coins+'Close'])/(x[coins+'High']-x[coins+'Low'])-1,
            coins+'volatility': lambda x: (x[coins+'High']- x[coins+'Low'])/(x[coins+'Open'])}
    market_info = market_info.assign(**kwargs)


model_data = market_info[['Date']+[coin+metric for coin in ['bt_', 'eth_'] 
                                   for metric in ['Close','Volume','close_off_high','volatility','High','Low']]]
# need to reverse the data frame so that subsequent rows represent later timepoints
model_data = model_data.sort_values(by='Date')

model_data = model_data.drop('eth_close_off_high',1)
model_data = model_data.drop('bt_close_off_high',1)
model_data = model_data.drop('eth_volatility',1)
model_data = model_data.drop('bt_volatility',1)

print (model_data.head())

# we don't need the date columns anymore
training_set, test_set = model_data[model_data['Date']<split_date], model_data[model_data['Date']>=split_date]
training_set = training_set.drop('Date', 1)
test_set = test_set.drop('Date', 1)
split_date2 = '2017-08-01'
model_data[model_data['Date']>=split_date2].to_csv("test_set.csv")
print("created test set to csv")
window_len = 20
norm_cols = [coin+metric for coin in ['bt_', 'eth_'] for metric in ['Close','Volume']]

# print(len(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime)))

# print (model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime))
LSTM_training_inputs = []
for i in range(len(training_set)-window_len):
    temp_set = training_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_training_inputs.append(temp_set)
# LSTM_training_outputs = (training_set['eth_Close'][window_len:].values/training_set['eth_Close'][:-window_len].values)-1

LSTM_training_outputs=LSTM_training_inputs

LSTM_test_inputs = []
for i in range(len(test_set)-window_len):
    temp_set = test_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    # print(temp_set)
    LSTM_test_inputs.append(temp_set)
    LSTM_last_input=temp_set
    # print (LSTM_test_inputs)
LSTM_test_outputs = (test_set['eth_Close'][window_len:].values/test_set['eth_Close'][:-window_len].values)-1

LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)
print(LSTM_training_inputs)
print(LSTM_training_inputs.shape[1])
print(LSTM_training_inputs.shape[2])
LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)
LSTM_last_input.to_csv("lastdata2.csv")
print("created csv data file")
LSTM_last_input = LSTM_test_inputs[-1]
LSTM_last_input.shape = (1,20,8)

np.random.seed(202)

# model output is next price normalised to 10th previous closing price
# LSTM_training_outputs = (training_set['eth_Close'][window_len:].values/training_set['eth_Close'][:-window_len].values)-1
# train model on data
# LSTM_training_outputs = (training_set['eth_Close'][window_len:].values/training_set['eth_Close'][:-window_len].values)-1
# note: eth_history contains information on the training error per epoch
# eth_history = eth_model.fit(LSTM_training_inputs, LSTM_training_outputs, 
#                             epochs=20, batch_size=1, verbose=2, shuffle=True)
print("this is eth_Close model")
LSTM_training_eth_Close_outputs = (training_set['eth_Close'][window_len:].values/training_set['eth_Close'][:-window_len].values)-1
eth_Close_model =build_model(LSTM_training_inputs, output_size=1, neurons = 20)
eth_Close_model.fit(LSTM_training_inputs, LSTM_training_eth_Close_outputs, 
                            epochs=20, batch_size=1, verbose=2, shuffle=True)

print("this is eth_Volume model")
LSTM_training_eth_Volume_outputs = (training_set['eth_Volume'][window_len:].values/training_set['eth_Volume'][:-window_len].values)-1
eth_Volume_model =build_model(LSTM_training_inputs, output_size=1, neurons = 30)
eth_Volume_model.fit(LSTM_training_inputs, LSTM_training_eth_Volume_outputs, 
                            epochs=20, batch_size=1, verbose=2, shuffle=True)

LSTM_training_eth_High_outputs = (training_set['eth_High'][window_len:].values/training_set['eth_High'][:-window_len].values)-1
eth_High_model =build_model(LSTM_training_inputs, output_size=1, neurons = 30)
eth_history = eth_High_model.fit(LSTM_training_inputs, LSTM_training_eth_High_outputs, 
                            epochs=20, batch_size=1, verbose=2, shuffle=True)
print("this is eth_Low model")
LSTM_training_eth_Low_outputs = (training_set['eth_Low'][window_len:].values/training_set['eth_Low'][:-window_len].values)-1
eth_Low_model =build_model(LSTM_training_inputs, output_size=1, neurons = 30)
eth_Low_model.fit(LSTM_training_inputs, LSTM_training_eth_Low_outputs, 
                            epochs=20, batch_size=1, verbose=2, shuffle=True)

print("this is bt_Close model")
LSTM_training_bt_Close_outputs = (training_set['bt_Close'][window_len:].values/training_set['bt_Close'][:-window_len].values)-1
bt_Close_model =build_model(LSTM_training_inputs, output_size=1, neurons = 30)
bt_Close_model.fit(LSTM_training_inputs, LSTM_training_bt_Close_outputs, 
                            epochs=20, batch_size=1, verbose=2, shuffle=True)

print("this is bt_Volume model")
LSTM_training_bt_Volume_outputs = (training_set['bt_Volume'][window_len:].values/training_set['bt_Volume'][:-window_len].values)-1
bt_Volume_model =build_model(LSTM_training_inputs, output_size=1, neurons = 30)
bt_Volume_model.fit(LSTM_training_inputs, LSTM_training_bt_Volume_outputs, 
                            epochs=20, batch_size=1, verbose=2, shuffle=True)

LSTM_training_bt_High_outputs = (training_set['bt_High'][window_len:].values/training_set['bt_High'][:-window_len].values)-1
bt_High_model =build_model(LSTM_training_inputs, output_size=1, neurons = 30)
bt_history = bt_High_model.fit(LSTM_training_inputs, LSTM_training_bt_High_outputs, 
                            epochs=20, batch_size=1, verbose=2, shuffle=True)
print("this is bt_Low model")
LSTM_training_bt_Low_outputs = (training_set['bt_Low'][window_len:].values/training_set['bt_Low'][:-window_len].values)-1
bt_Low_model =build_model(LSTM_training_inputs, output_size=1, neurons = 30)
bt_Low_model.fit(LSTM_training_inputs, LSTM_training_bt_Low_outputs, 
                            epochs=20, batch_size=1, verbose=2, shuffle=True)


model_json = eth_Close_model.to_json()
model_output = "eth_Close_model.json"
weight_output = "eth_Close_model.h5"
with open(model_output, "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        eth_Close_model.save_weights(weight_output)

model_json = eth_Volume_model.to_json()
model_output = "eth_Volume_model.json"
weight_output = "eth_Volume_model.h5"
with open(model_output, "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        eth_Volume_model.save_weights(weight_output)

model_json = eth_High_model.to_json()
model_output = "eth_High_model.json"
weight_output = "eth_High_model.h5"
with open(model_output, "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        eth_High_model.save_weights(weight_output)

model_json = eth_Low_model.to_json()
model_output = "eth_Low_model.json"
weight_output = "eth_Low_model.h5"
with open(model_output, "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        eth_Low_model.save_weights(weight_output)

model_json = bt_Close_model.to_json()
model_output = "bt_Close_model.json"
weight_output = "bt_Close_model.h5"
with open(model_output, "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        bt_Close_model.save_weights(weight_output)

model_json = bt_Volume_model.to_json()
model_output = "bt_Volume_model.json"
weight_output = "bt_Volume_model.h5"
with open(model_output, "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        bt_Volume_model.save_weights(weight_output)

model_json = bt_High_model.to_json()
model_output = "bt_High_model.json"
weight_output = "bt_High_model.h5"
with open(model_output, "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        bt_High_model.save_weights(weight_output)

model_json = bt_Low_model.to_json()
model_output = "bt_Low_model.json"
weight_output = "bt_Low_model.h5"
with open(model_output, "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        bt_Low_model.save_weights(weight_output)


print("Saved model to disk")



# New_input = LSTM_last_input
# # print(New_input)
# # print(New_input[0][9])
# New_eth_Close =eth_Close_model.predict(LSTM_last_input)

# New_eth_Volume =eth_Volume_model.predict(LSTM_last_input)

# # New_eth_close_off_high =eth_close_off_high_model.predict(LSTM_last_input)

# New_eth_volatilityeth =eth_volatility_model.predict(LSTM_last_input)

# New_bt_Close =bt_Close_model.predict(LSTM_last_input)

# New_bt_Volume =bt_Volume_model.predict(LSTM_last_input)

# # New_bt_close_off_high =bt_close_off_high_model.predict(LSTM_last_input)

# New_bt_volatilitybt_close =bt_volatility_model.predict(LSTM_last_input)

# New_input[0][9][0] = New_eth_Close

# New_input[0][9][1] = New_eth_Volume

# # New_input[0][9][2] = eth_close_off_high_model.predict(LSTM_last_input)


# # New_input[0][9][2] = New_eth_volatilityeth

# New_input[0][9][2] = New_bt_Close

# New_input[0][9][3] = New_bt_Volume

# # New_input[0][9][6] = bt_close_off_high_model.predict(LSTM_last_input)


# # New_input[0][9][5] =New_bt_volatilitybt_close

# # print(New_input)
# print(New_input)
# print(eth_Close_model.predict(New_input))


# # # print(LSTM_test_inputs[-1])
# # print("this is total test")
# # print(LSTM_test_inputs)
# # print("this is the total predict")
# # print(eth_Close_model.predict(LSTM_test_inputs)) 

# # print("this is the last test")
# # print(LSTM_last_input)
# # print("this is the last predict")
# # print(eth_Close_model.predict(LSTM_last_input))



# # predict_date=pd.read_csv('date.csv')

# # fig, ax1 = plt.subplots(1,1)
# # ax1.set_xticks([datetime.date(2018,i+1,1) for i in range(12)])
# # ax1.set_xticklabels([datetime.date(2018,i+1,1).strftime('%b %d %Y')  for i in range(12)])
# # # ax1.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime),
# # #          test_set['eth_Close'][window_len:], label='Actual')
# # ax1.plot(predict_date.tail(53)['date'].astype(datetime.datetime),
# #          ((np.transpose(eth_Close_model.predict(LSTM_test_inputs))+1) * test_set['eth_Close'].values[:-window_len])[0], 
# #          label='Predicted')
# # ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(eth_Close_model.predict(LSTM_test_inputs))+1)-\
# #             (test_set['eth_Close'].values[window_len:])/(test_set['eth_Close'].values[:-window_len]))), 
# #              xy=(0.75, 0.9),  xycoords='axes fraction',
# #             xytext=(0.75, 0.9), textcoords='axes fraction')
# # ax1.set_title('Test Set: Single Timepoint Prediction',fontsize=13)
# # ax1.set_ylabel('Ethereum Price ($)',fontsize=12)
# # ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
# # plt.show()



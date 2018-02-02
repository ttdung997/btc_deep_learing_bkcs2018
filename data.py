
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np
# t = (2018, 12, 30, 17, 3, 38, 1, 48, 0)
# t = time.mktime(t)


# print(pd.DatetimeIndex.dayofyear)
# print(time.strftime("%Y%m%d",time.gmtime(t)))
# get market info for bitcoin from the start of 2016 to the current day
bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]
# convert the date string to the correct date format
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

import sys
from PIL import Image
import io

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

# fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})
# ax1.set_ylabel('Closing Price ($)',fontsize=12)
# ax2.set_ylabel('Volume ($ bn)',fontsize=12)
# ax2.set_yticks([int('%d000000000'%i) for i in range(10)])
# ax2.set_yticklabels(range(10))
# ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
# ax1.set_xticklabels('')
# ax2.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
# ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,7]])
# ax1.plot(bitcoin_market_info['Date'].astype(datetime.datetime),bitcoin_market_info['bt_Open'])
# ax2.bar(bitcoin_market_info['Date'].astype(datetime.datetime).values, bitcoin_market_info['bt_Volume'].values)
# fig.tight_layout()
# fig.figimage(bitcoin_im, 100, 120, zorder=3,alpha=.5)
# plt.show()
# market_in

# fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})
# #ax1.set_yscale('log')
# ax1.set_ylabel('Closing Price ($)',fontsize=12)
# ax2.set_ylabel('Volume ($ bn)',fontsize=12)
# ax2.set_yticks([int('%d000000000'%i) for i in range(10)])
# ax2.set_yticklabels(range(10))
# ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
# ax1.set_xticklabels('')
# ax2.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
# ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,7]])
# ax1.plot(eth_market_info['Date'].astype(datetime.datetime),eth_market_info['eth_Open'])
# ax2.bar(eth_market_info['Date'].astype(datetime.datetime).values, eth_market_info['eth_Volume'].values)
# fig.tight_layout()
# fig.figimage(eth_im, 300, 180, zorder=3, alpha=.6)
# plt.show()


market_info = pd.merge(bitcoin_market_info,eth_market_info, on=['Date'])
market_info = market_info[market_info['Date']>='2016-01-01']
for coins in ['bt_', 'eth_']: 
    kwargs = { coins+'day_diff': lambda x: (x[coins+'Close']-x[coins+'Open'])/x[coins+'Open']}
    market_info = market_info.assign(**kwargs)
market_info.head()

split_date = '2017-12-01'


# fig, (ax1, ax2) = plt.subplots(2,1)
# ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
# ax1.set_xticklabels('')
# ax2.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
# ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,7]])
# ax1.plot(market_info[market_info['Date'] < split_date]['Date'].astype(datetime.datetime),
#          market_info[market_info['Date'] < split_date]['bt_Close'], 
#          color='#B08FC7', label='Training')
# ax1.plot(market_info[market_info['Date'] >= split_date]['Date'].astype(datetime.datetime),
#          market_info[market_info['Date'] >= split_date]['bt_Close'], 
#          color='#8FBAC8', label='Test')
# ax2.plot(market_info[market_info['Date'] < split_date]['Date'].astype(datetime.datetime),
#          market_info[market_info['Date'] < split_date]['eth_Close'], 
#          color='#B08FC7')
# ax2.plot(market_info[market_info['Date'] >= split_date]['Date'].astype(datetime.datetime),
#          market_info[market_info['Date'] >= split_date]['eth_Close'], color='#8FBAC8')
# ax1.set_xticklabels('')
# ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
# ax2.set_ylabel('Ethereum Price ($)',fontsize=12)
# plt.tight_layout()
# ax1.legend(bbox_to_anchor=(0.03, 1), loc=2, borderaxespad=0., prop={'size': 14})
# fig.figimage(bitcoin_im.resize((int(bitcoin_im.size[0]*0.65), int(bitcoin_im.size[1]*0.65)), Image.ANTIALIAS), 
#              200, 260, zorder=3,alpha=.5)
# fig.figimage(eth_im.resize((int(eth_im.size[0]*0.65), int(eth_im.size[1]*0.65)), Image.ANTIALIAS), 
#              350, 40, zorder=3,alpha=.5)
# plt.show()

# trivial lag model: P_t = P_(t-1)
# fig, (ax1, ax2) = plt.subplots(2,1)
# ax1.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
# ax1.set_xticklabels('')
# ax2.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
# ax2.set_xticklabels([datetime.date(2017,i+1,1).strftime('%b %d %Y')  for i in range(12)])
# ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
#          market_info[market_info['Date']>= split_date]['bt_Close'].values, label='Actual')
# ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
#           market_info[market_info['Date']>= datetime.datetime.strptime(split_date, '%Y-%m-%d') - 
#                       datetime.timedelta(days=1)]['bt_Close'][1:].values, label='Predicted')
# ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
# ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
# ax1.set_title('Simple Lag Model (Test Set)')
# ax2.set_ylabel('Etherum Price ($)',fontsize=12)
# ax2.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
#          market_info[market_info['Date']>= split_date]['eth_Close'].values, label='Actual')
# ax2.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
#           market_info[market_info['Date']>= datetime.datetime.strptime(split_date, '%Y-%m-%d') - 
#                       datetime.timedelta(days=1)]['eth_Close'][1:].values, label='Predicted')
# fig.tight_layout()
# plt.show()

for coins in ['bt_', 'eth_']: 
    kwargs = { coins+'close_off_high': lambda x: 2*(x[coins+'High']- x[coins+'Close'])/(x[coins+'High']-x[coins+'Low'])-1,
            coins+'volatility': lambda x: (x[coins+'High']- x[coins+'Low'])/(x[coins+'Open'])}
    market_info = market_info.assign(**kwargs)


model_data = market_info[['Date']+[coin+metric for coin in ['bt_', 'eth_'] 
                                   for metric in ['Close','Volume','close_off_high','volatility']]]
# need to reverse the data frame so that subsequent rows represent later timepoints
model_data = model_data.sort_values(by='Date')
print (model_data.head())
# print(model_data.head())


# we don't need the date columns anymore
training_set, test_set = model_data[model_data['Date']<split_date], model_data[model_data['Date']>=split_date]
training_set = training_set.drop('Date', 1)
test_set = test_set.drop('Date', 1)


window_len = 10
norm_cols = [coin+metric for coin in ['bt_', 'eth_'] for metric in ['Close','Volume']]

# print (model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime))
LSTM_training_inputs = []
for i in range(len(training_set)-window_len):
    temp_set = training_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_training_inputs.append(temp_set)
LSTM_training_outputs = (training_set['eth_Close'][window_len:].values/training_set['eth_Close'][:-window_len].values)-1

LSTM_test_inputs = []
for i in range(len(test_set)-window_len):
    temp_set = test_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    # print(temp_set)
    LSTM_test_inputs.append(temp_set)
    # print (LSTM_test_inputs)
LSTM_test_outputs = (test_set['eth_Close'][window_len:].values/test_set['eth_Close'][:-window_len].values)-1


# I find it easier to work with numpy arrays rather than pandas dataframes
# especially as we now only have numerical data
LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)
 # print(LSTM_training_inputs)
LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)
# print(LSTM_test_inputs)

# import the relevant Keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

np.random.seed(202)
# initialise model architecture
eth_model = build_model(LSTM_training_inputs, output_size=1, neurons = 20)
# model output is next price normalised to 10th previous closing price
LSTM_training_outputs = (training_set['eth_Close'][window_len:].values/training_set['eth_Close'][:-window_len].values)-1
# train model on data
# note: eth_history contains information on the training error per epoch
eth_history = eth_model.fit(LSTM_training_inputs, LSTM_training_outputs, 
                            epochs=10, batch_size=1, verbose=2, shuffle=True)
# print("training input")
# print(LSTM_training_inputs)
# print("training output")
# print(LSTM_training_outputs)
# print("test input")
# print(LSTM_test_inputs)
# print("test output")
# print(LSTM_test_outputs)

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# fig, ax1 = plt.subplots(1,1)
# ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,5,9]])
# ax1.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,5,9]])
# ax1.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
#          training_set['eth_Close'][window_len:], label='Actual')
# ax1.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
#          ((np.transpose(eth_model.predict(LSTM_training_inputs))+1) * training_set['eth_Close'].values[:-window_len])[0], 
#          label='Predicted')
# ax1.set_title('Training Set: Single Timepoint Prediction')
# ax1.set_ylabel('Ethereum Price ($)',fontsize=12)
# ax1.legend(bbox_to_anchor=(0.15, 1), loc=2, borderaxespad=0., prop={'size': 14})
# ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(eth_model.predict(LSTM_training_inputs))+1)-\
#             (training_set['eth_Close'].values[window_len:])/(training_set['eth_Close'].values[:-window_len]))), 
#              xy=(0.75, 0.9),  xycoords='axes fraction',
#             xytext=(0.75, 0.9), textcoords='axes fraction')
# # # figure inset code taken from http://akuederle.com/matplotlib-zoomed-up-inset
# # axins = zoomed_inset_axes(ax1, 3.35, loc=10) # zoom-factor: 3.35, location: centre
# # axins.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,5,9]])
# # axins.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
# #          training_set['eth_Close'][window_len:], label='Actual')
# # axins.plot(model_data[model_data['Date']< split_date]['Date'][window_len:].astype(datetime.datetime),
# #          ((np.transpose(eth_model.predict(LSTM_training_inputs))+1) * training_set['eth_Close'].values[:-window_len])[0], 
# #          label='Predicted')
# # axins.set_xlim([datetime.date(2017, 3, 1), datetime.date(2017, 5, 1)])
# # axins.set_ylim([10,60])
# # axins.set_xticklabels('')
# # mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
# plt.show()
# split_date = '2018-01-01'

# model_json = eth_model.to_json()
# model_output = "eth_model.json"
# weight_output = "eth_model.h5"

# with open(model_output, "w") as json_file:
#         json_file.write(model_json)
#         # serialize weights to HDF5
#         eth_model.save_weights(weight_output)
# print("Saved model to disk")
# print (test_set['eth_Close'][window_len:])
# print(((np.transpose(eth_model.predict(LSTM_test_inputs))+1) * test_set['eth_Close'].values[:-window_len])[0])

predict_date=pd.read_csv('date.csv')
# print (predict_date.tail(52)['date'].astype(datetime.datetime))
# print (len(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime)))
# print((np.transpose(eth_model.predict(LSTM_test_inputs))+1) * test_set['eth_Close'].values[:-window_len])[0]
print("This is the test input ")
print(LSTM_test_inputs)
print("This is the test output ")
print (eth_model.predict(LSTM_test_inputs))
print("This is some think?????")
print(test_set['eth_Close'].values[:-window_len])[0]
print("this is the value not x")
print(np.transpose(eth_model.predict(LSTM_test_inputs))+1)
print("this is the value")
print(((np.transpose(eth_model.predict(LSTM_test_inputs))+1) * test_set['eth_Close'].values[:-window_len])[0])

fig, ax1 = plt.subplots(1,1)
ax1.set_xticks([datetime.date(2018,i+1,1) for i in range(12)])
ax1.set_xticklabels([datetime.date(2018,i+1,1).strftime('%b %d %Y')  for i in range(12)])
# ax1.plot(model_data[model_data['Date']>= split_date]['Date'][window_len:].astype(datetime.datetime),
#          test_set['eth_Close'][window_len:], label='Actual')
ax1.plot(predict_date.tail(52)['date'].astype(datetime.datetime),
         ((np.transpose(eth_model.predict(LSTM_test_inputs))+1) * test_set['eth_Close'].values[:-window_len])[0], 
         label='Predicted')
ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(eth_model.predict(LSTM_test_inputs))+1)-\
            (test_set['eth_Close'].values[window_len:])/(test_set['eth_Close'].values[:-window_len]))), 
             xy=(0.75, 0.9),  xycoords='axes fraction',
            xytext=(0.75, 0.9), textcoords='axes fraction')
ax1.set_title('Test Set: Single Timepoint Prediction',fontsize=13)
ax1.set_ylabel('Ethereum Price ($)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
plt.show()

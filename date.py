import time
from datetime import date
from calendar import monthrange
import pandas as pd

date=[]
for month in range(1,13):
	for day in range(1,monthrange(2018,month)[1]+1):
		if(day < 10 and month <10):
			date.append('2018-0'+str(month)+'-0'+str(day)+' 00:00:00')
		elif(day < 10):
			date.append('2018-'+str(month)+'-0'+str(day)+' 00:00:00')
		elif(month < 10):
			date.append('2018-0'+str(month)+'-'+str(day)+' 00:00:00')
		else:
			date.append('2018-'+str(month)+'-'+str(day)+' 00:00:00')
		data = {'date': date}
ResultDf = pd.DataFrame(data)
ResultDf.to_csv('date.csv')
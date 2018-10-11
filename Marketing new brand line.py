
# coding: utf-8

# # Marketing strategy for a new brand line
# 
# ## Table of contents
# <ul><font color=blue>
# <li><a href="#problem">Question and problem definition</a></li>
# <li><a href="#wrangle">Wrangle, prepare, cleanse the data</a></li>
# <li><a href="#analyze">Analyze, identify patterns, and explore the data</a></li>
# <li><a href="#model">Model, predict and solve the problem</a></li>
# <li><a href="#conclusion">Report and present the problem solving steps and final solution</a></li>
# </ul>
# 
# 
# 

# <a id='problem'></a>
# ### Question and problem definition
# Our client is a business platform which serves online marketing campaigns for many brands. One of the purposes of these campaigns is to incentivise online purchases of new products.
# One of the brands that they work with has a catalogue of products with different prices, but none of them are within the range between 30 and 40 pounds. This particular brand wants to fill the gap in this price range by releasing a new line.
# Our mission is to use existing data to evaluate how to best market the new line and predict the sales.
# The dataset provided covered the period from 2018, July 1st to August 14th, but has several missing data points we need to provide in order to fit a model which will predict the sales in the following 2-3 weeks 

# In[1]:


#This code is to import the libraries we will ptentially be using during the execution of the tasks I have been assigned to
# data analysis and wrangling
import pandas as pd
import pickle
import numpy as np
import random as rnd
from datetime import timedelta
from scipy import stats

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[3]:


#Import of data into a dataframe called df
df=pd.read_pickle('case_study.pickle')
df.head()


# <a id='wrangle'></a>
# ### Wrangle, prepare, cleanse the data
# In this section we will be exploring the dataset to answer qustions like:
# <ul>
# <li>Which features are available in the dataset? </li>
# <li>Which features are categorical and which ones are numerical?</li>
# <li>Which features have missing values and how can we handle them? </li>
# <li>which ones can we drop?</li>
#     </ul>

# In[4]:


#data exploration
df.info()


# In[5]:


#converting type "object" to "datetime" for column "date" 
df['date'] = df['date'].astype('datetime64')
df.info()
#Sorting the values by 'date'
df.sort_values(by='date', inplace=True)


# In[6]:


#Let's give a look to the time-window from 8th August to 14th August
df.loc[(df['date'] >= '2018-08-08') & (df['date'] <= '2018-08-14')]


# In[7]:


#The dataset contains a total of 27091 entries. Features "County" and "Country" may be droppped form the dataset since
#1.more than 50% of the rows entries are missing 
#2.they may not contribute to determine the sales
df=df.drop(['County', 'Country'], axis=1)


# In[8]:


#Describing statistical metrics of numerical features
df.describe()


# <a id='analyze'></a>
# ### Analyze, identify patterns, and explore the data
# In this session I will be looking much deeper into data trends, exploring their statistical distribution and eventually correcting outliers and nans

# In[9]:


#Features "avg_price" and "total_products" are missing roughly 2000 data points. Let's give a closer look to these 2 features
#Checking frequency distribution for 'total_product'
sns.boxplot(x=df['total_products'])


# In[10]:


#The box plot shows presence of outliers. To isolate and remove the outliers using the interquartile range (IQR):
Q1 =  df.total_products.quantile(0.25)
Q3 =  df.total_products.quantile(0.75)
IQR = Q3 - Q1
 df =  df[~(( df.total_products < (Q1 - 1.5 * IQR)) |( df.total_products > (Q3 + 1.5 * IQR)))]
 df.describe()


# I dropped roughly 1300 data points that is about 5% of the total. While I might loose some specific  patterns I still preserve the main trend.

# In[11]:


#Checking distribution of 'total_products'
df['total_products'].plot.hist(figsize=(30,10), bins=10)


# In[12]:


#Since the data from 'total_products' don't look to follow a normal distribution I decide to impute the 
#median (which is also the most common value of the total products sold each session) in the missing da-
#ta points
df['total_products']=df['total_products'].fillna(df['total_products'].median())
df.describe()


# In[13]:


#Checking frequency distribution for average price
df['avg_price'].plot.hist(figsize=(30,10), bins=10)


# In[14]:


#since the data from 'avg_price'looks to follow a normal distribution I will impute mean values for 
#missing datapoints
Beamly_df['avg_price']=Beamly_df['avg_price'].fillna(Beamly_df['avg_price'].mean())
Beamly_df.describe()


# In[15]:


#Let's give a look again to the time-window  from August 8th to August 14th
df.loc[(df['date'] >= '2018-08-08') & (df['date'] <= '2018-08-14')]


# In[16]:


#I noticed 2018-08-14 have a weird trend compare to the other days (avg_hour,min_hour, max_hour are all 
#equal to 0 and the numbers of rows(=sales) is also lower than the other previous days). I decide to 
#not consider data coming from this date
 df= df[ df['date'] != '2018-08-14']


# In[17]:


#describing statistical distribution of categorical features
df.describe(include=['O'])


# In[18]:


#let's check the categories of 'productBand'...
df['productBand'].unique()


# In[19]:


#...and convert them in numerical ordinal ones
df['productBand'] = df['productBand'].map( {'price_missing': 0, 'lessThan10': 1, 'between10and20': 2, 'between20and30': 3, 'between40and50': 4, 'moreThan50': 5} ).astype(int)


# In[20]:


#let's check the categories of 'device_name'
df['device_name'].unique()


# In[21]:


#one device_name category is "Unknown". Let's check how many entries are present for each device_name
df.groupby(['device_name']).count()


# In[22]:


#I decided to not consider the "Unknown" category for our analysis, since it has only 119 entries
 df= df[ df['device_name'] != 'Unknown']
 df['device_name']= df['device_name'].map({'Mobile':0, 'Tablet':1,'Desktop and Laptop':2})


# In[23]:


#I drop the features 'city' and 'region', since they won't be used for the modelling stage
df.drop(['city', 'region'], axis=1, inplace=True)


# In[24]:


#I assume the frequency of sales depends on the day of the week. Therefore I will create a new variable ('day_of_week')
#for the day of the week 
df['day_of_week']=df['date'].dt.dayofweek


# In[25]:


#Let's give a look how data distributions look like after the cleaning stage
df.hist(bins = 40, figsize = (20, 10));


# <a id='model'></a>
# ### Model, predict and solve the problem
# Now I will focus on answering the questions for this challenge through use of machine learning tools.
# In this section I will:
# <ol type="1">
# <li>Build a predictive model for the sales from August 14th to August 31st</li>
# <li>Simulate the sales after introducing the new brand line which has price range between 30 and 40s pound</li>
# <li>Predict future sales using the new line of price ranges</li>
#     </ol>
#  

# 1.Build a predictive model for the sales from August 14th to August 31st

# In[26]:


#Let's give a look how the time-serie of daily sales looks like
daily_trend=pd.DataFrame(df.groupby(['date']).agg({'total_products':'sum'}))
daily_trend.plot()
plt.ylabel("Daily sales")


# In[27]:


#I will train Prophet machine learning algorithm on sales data
from fbprophet import Prophet
data_prophet = daily_trend.copy()
data_prophet = pd.DataFrame(data_prophet)
data_prophet.reset_index(drop=False, inplace=True)
data_prophet.columns =['ds','y']
data_prophet


# In[31]:


#Using trained Prophet model to predict the fututre 18 days
m = Prophet()
m.fit(data_prophet)
future = m.make_future_dataframe(periods=18, freq='D')
forecast = m.predict(future)
m.plot(forecast)
plt.savefig('forecast.png', bbox_inches='tight')
plt.show()

#daily_trend.plot()


# In[792]:


#let's visualize the trends of the sales
m.plot_components(forecast)


# There is a clear trend of growth for the sales with time. On a weekly base, Tuesday is the day with more sales. After Tuesday, sales drop and reach a minimum on Friday

# In[793]:


forecast.columns


# In[794]:


#now let's see the values for the future 18 days
forecasted_values = forecast[['ds', 'yhat']].tail(18)
forecasted_values = forecasted_values.set_index('ds')
forecasted_values.columns = ['y']
forecasted_values


# In[795]:


#calculating total of sales
sum(forecasted_values['y'])


# 2.Simulate the sales after introducing the new brand line which has price range between 30 and 40s pound

# In[796]:


#Our final goal is to see if the new brand line is going to be beneficial for the sales
#let's calculate how many products the client has sold in the time window of the given dataset and without
#using the new bran line
sum(df['total_products'])


# In[797]:


#And let's calculate the revenue for the same as above
Total_revenue=df.total_products*df.avg_price
sum(Total_revenue)


# In[808]:


#Now I want to build a predictive model for the sales, based on the dataset features.
#I decided to split some redundant features like 'productBand'(we already have 'avg_price',
#'min_hour'and 'max_hour'(we have 'avg_hour', that has a similar distrobution as shown by histogram plots).
#I will split the time-series between training and testing dataset in a ratio 2:1. The model will be
#therefore trained on the a period of 1 month..

 df_train =  df.loc[( df['date'] >= '2018-07-01')|( df['date']<= '2018-07-31')]
Y_train =  df_train['total_products']
X_train =  df_train.drop(['total_products', 'date','productBand','min_hour','max_hour'], axis=1)  


# In[809]:


#and test on the following 14 days
 df_test =  df.loc[( df['date'] >= '2018-08-01')|( df['date']< '2018-08-14')]
Y_test =  df_test['total_products']
X_test =  df_test.drop(['total_products', 'date','productBand','min_hour','max_hour'], axis=1)


# In[810]:


df_train.groupby('total_products').sum()


# In[819]:


#Let's evaluate which machine learning algorythm is more accurate using R2 as metric
from sklearn.metrics import r2_score

models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('RT', DecisionTreeClassifier()))
models.append(('DF', RandomForestClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('CRT', DecisionTreeRegressor()))
models.append(('RDF', RandomForestRegressor()))
models.append(('LNR', LinearRegression()))
models.append(('RID', Ridge()))
models.append(('LAR', Lasso()))


# In[820]:


# evaluate each model in turn
results = []
names = []
for name, model in models:
    
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
     
    results.append(r2)
   
    names.append(name)
    msg = "%s: %f" % (name, r2.mean())
   
    print(msg)


# In[821]:


#Decision Tree algorithm is giving the best accuracy. Let's create the regressor for it and call it DTR
DTR = DecisionTreeRegressor(random_state = 46)
DTR.fit(X_train, Y_train);
print(DTR.score(X_test, Y_test))


# In[822]:


#I am checking the lenght of the dataframe before creating a new range of average prices 
df.shape


# In[823]:


#I am now creatin a new columns for new average prices of products. This serie of value will have a 
#normal distribution based on the mean and standard deviation of the existing 'avg_price' data
df['avg_price_random_30to40'] =np.random.normal(23.428529 ,13.437024, 25614)


# In[824]:


#Let's check how the new dataframe looks like
df.describe()


# In[825]:


#The 'avg_price_random_30to40' feature look to have negative values, which I am going to replace with 
#the mean for this feature
avg_positive=[]
for i in df['avg_price_random_30to40']:
    if i >0:
           avg_positive.append(i)
    else:
           avg_positive.append(df['avg_price_random_30to40'].mean())
#replacing the vg_price_random_30to40' column with the new values                             
df['avg_price_random_30to40']=avg_positive


# In[826]:


#let's compare the 2 distribution of average prices of products
df['avg_price'].plot.hist(figsize=(30,10), bins=10)
df['avg_price_random_30to40'].plot.hist(figsize=(30,10), bins=10)


# In[827]:


#I will now replace 'avg_price'column with the new set of price range
df['avg_price']=df['avg_price_random_30to40']


# In[833]:


#and I will predict the sales for this new line using the previously built and trained regressor 'DTR'
X_newline = df.drop(['total_products', 'date', 'avg_price_random_30to40','productBand','min_hour','max_hour'], axis=1)
Y_pred_newline = DTR.predict(X_newline) 
    


# In[834]:


#let's calculate the total sales for this new brand line
sum(Y_pred_newline)


# In[835]:


#and also the revenue
Total_revenue=Y_pred_newline*df.avg_price
sum(Total_revenue)


# In[837]:


#let's check which device is more used for the sales
df.groupby('device_name').agg({'avg_price':'mean', 'total_products':'sum'})


# Both sales and revenue have been increased by the introduction of the new brand line:
# Total sales: from 32269 to ~40000 pounds;
# Total revenue: from 740343 to ~900000 pounds

# 3.Predict future sales using the new line of price ranges

# In[838]:


#let's add the new predicted total sales to the initial dataframe in a column named 'total_products_new_line'
df['total_products_new_line']=Y_pred_newline
Y_pred_newline


# In[839]:


daily_trend_new_line=pd.DataFrame(df.groupby(['date']).agg({'total_products_new_line':'sum'}))
daily_trend_new_line


# In[840]:


#I will train Prophet machine learning algorythm on the new dataset
data_prophet_new = daily_trend_new_line.copy()
data_prophet_new = pd.DataFrame(data_prophet_new)
data_prophet_new.reset_index(drop=False, inplace=True)
data_prophet_new.columns =['ds','y']
data_prophet_new.sum()


# In[841]:


#Using trained Prophet model to predict the future 18 days
m = Prophet()
m.fit(data_prophet_new)
future = m.make_future_dataframe(periods=18, freq='D')
forecast_new = m.predict(future)
m.plot(forecast_new)
daily_trend_new_line.plot()


# In[842]:


#now let's see the values for the future 18 days
forecasted_values_new = forecast_new[['ds', 'yhat']].tail(18)
forecasted_values_new = forecasted_values_new.set_index('ds')
forecasted_values_new.columns = ['y']
forecasted_values_new


# In[843]:


#calculating total of sales
sum(forecasted_values_new['y'])


# <a id='conclusion'></a>
# ### Report and present the problem solving steps and final solution
# In this challenge I was able to use a dataset containing informations relative to sales from 2018-07-01 to 2018-08-13 to :
# <ol type="1">
# <li>Correct and impute missing values based on their statistical distribution</li>
# <li>Build a model to predict the sales for the following time window of 18 days (from 2018-08-14 to 2018-08-31)</li>
# <li>Simulate the sales upon introduction of a new brand line which was covering a range of average price between 30 and 40 pounds, previously missing from the dataset</li>
# <li>Evaluate the benefits of the introduction of the new line on sales and revenue</li>
# <li>Isolate the best platform for marketing the new brand line</li>
# <li>Predict the future sales upon marketing of the new brand line, evaluating the benefits on the overall sales </li>
# </ol>

# 1.Correct and impute missing values based on their statistical distribution
# The main significant changes I brought to the dataset were:
# <ul>
# <li>dropping sales of products that were judged as outliers with interquartile range metric. This made me drop a good 5% of data points, but probably significantly improve the model</li>
# <li>replacing missing values for sales and average price based on their statistical distribution</li>
# <li>dropping partial data relative to a date (2018-08-14)</li>
# <li>introducing a new feature based on the day of the week data were recorded for </li>
# </ul>
# 
# 

# 2.Build a model to predict the sales for the following time window of 18 days (from 2018-08-14 to 2018-08-31)
# 
# I used Prophet, a forecasting open source tool from Facebook, to predict the future sales.
# 
# Unfortunately the provided dataset covers a time window that is too short to correct for seasonality or other trends.
# Despite that, I was able to predict sales, identify a trend of overall growth during time with maximal sales frequency on Tuesday and to evaluate the total revenue.
# 
# Other algorithms should also be tested, such as Autoregressive Integrated Moving average (ARIMA)

# 3.Simulate the sales upon introduction of a new brand line which was covering a range of average price between 30 and 40 pounds, previously missing from the dataset
# 
# I created a new average price feature comprehensive of the price range 30-40 pounds, by assuming a normal distribution with mean and standard deviation equal to those of the pre-existing 'avg_price' feature.
# Some of the new average prices were negative so I replaced them with the mean of the distribution.
# This approach is stochastic, so it might not predicted as precisely as I am assuming
# 

# 4.Evaluate the benefits of the introduction of the new line of products on sales and revenue
# 
# Using DecisionTreeRegressor I was able to estimate total sales and total revenue for previous price range and compared them with those coming from the introduction of the new price-range line. The new line produces an increase in sales of about 25% (from ~32 to 40 thousands) which corrisponds to an increase of the revenue of about 200000 pounds
# 
# 

# 5.Isolate the best platform for marketing the new brand line
# 
# Both visualization and analysis of data trends showed that the majority of the sales are operated on mobiles, followed by laptops and lastly by tablets.
# Therefore, the best 2 platforms to market the new line will be mobiles and laptops

# 6.Predict the future sales upon marketing of the new brand line, evaluating the benefits in term of sales and revenue
# 
# Training Prophet with the predicted new sales data, allowed me to predict what is going to happen to the sales after introducing the new line for the time-window from August 14th to August 31st.
# Again introduction of the new price range products improved the total sales of about 25-30%. 

# Therefore introduction of the new price range of products looks to be beneficial in terms of sales and revenue and could be further pushed by focusing the marketing on mobiles and laptops devices.

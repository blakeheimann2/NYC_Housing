import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from pandas.plotting import scatter_matrix

from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import BaggingRegressor # I think this bagging method can be used with various types of models
from sklearn import tree

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from scipy import stats

df = pd.DataFrame(pd.read_csv('NYC_final_data.csv'))
del df['ease-ment']
del df['Unnamed: 0']
df = df.drop_duplicates()

df.columns[df.isnull().any()]
del df['cb_website']
del df['moe_under18_rate_boro']
del df['pop_acs']
del df['shared_puma_cd']
del df['the_geom']
del df['the_geom_webmercator']


df['land square feet'] = df['land square feet'].replace(0,np.nan)
df['gross square feet'] = df['land square feet'].replace(0,np.nan)
df['land square feet']=df['land square feet'].fillna(df['land square feet'].mean())
df['gross square feet']=df['gross square feet'].fillna(df['gross square feet'].mean())

numeric_data=df.select_dtypes(include=[np.number])
######################
import seaborn as sns
corrmat = numeric_data.corr()
k = 20 #number of variables for heatmap
cols = corrmat.nlargest(k, 'sale_price')['sale_price'].index
cm = np.corrcoef(numeric_data[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

plt.figure(figsize=(15,6))
sns.boxplot(x='sale_price', data=df)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of SALE PRICE in USD')
plt.show()
#see distribution
sns.distplot(df['sale_price'])
plt.show()
#drop outlying data

mean = np.mean(df['sale_price'])
sd = np.std(df['sale_price'])
low_cap =  mean - 1 * sd
high_cap =  mean + 1* sd
data = df[(df['sale_price'] > low_cap) & (df['sale_price'] < high_cap)]

sns.distplot(data['sale_price'])
plt.show()
from sklearn.preprocessing import StandardScaler
sales= np.log(data['sale_price'])
print(sales.skew())
sns.distplot(sales)
plt.show

sns.boxplot(x='gross square feet', data=data,showfliers=False)
plt.show()
sns.boxplot(x='land square feet', data=data,showfliers=False)

mean = np.mean(df['gross square feet'])
sd = np.std(df['gross square feet'])
low_cap =  mean - 1 * sd
high_cap =  mean + 1 * sd
data = df[(df['gross square feet'] > low_cap) & (df['gross square feet'] < high_cap)]
sns.boxplot(x='gross square feet', data=data,showfliers=False)
plt.show()

sns.regplot(x='gross square feet', y='sale_price', data=data, fit_reg=False, scatter_kws={'alpha':0.3})
axes = plt.gca()
axes.set_ylim([0,50000000])
plt.show()

sns.regplot(x='land square feet', y='sale_price', data=data, fit_reg=False, scatter_kws={'alpha':0.3})
axes = plt.gca()
axes.set_ylim([0,50000000])
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='commercial units', y='sale_price', data=data)
plt.title('Commercial Units vs Sale Price')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='residential units', y='sale_price', data=data)
plt.title('Residential Units vs Sale Price')
plt.show()

data['tax class at present'].unique()
data['tax class at time of sale'].unique()
data['building class category'].unique()
pivot=data.pivot_table(index='building class category', values='sale_price', aggfunc=np.median)
pivot.plot(kind='bar', color='Green')

#based on 

data[["building class category", "sale_price"]].groupby(['building class category'], as_index=False).count().sort_values(by='sale_price', ascending=False)
#grab subset of building types..?
data['borough'].unique()


del data['ADDRESS']
del data['APARTMENT NUMBER']

from scipy.stats import skew
skewed = data[numeric_data.columns].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[abs(skewed) > 0.75]
skewed = skewed.index
data[skewed] = np.log1p(data[skewed])


scaler = StandardScaler()
scaler.fit(data[numeric_data.columns])
scaled = scaler.transform(data[numeric_data.columns])
for i, col in enumerate(numeric_data.columns):
       data[col] = scaled[:,i]
NYC1 = data.loc[:, ['sale_price',  'borough', 'crime_count', 'residential units','commercial units', 'year built', 'cd_short_title', 'building class at time of sale', 'building class category', 'tax class at time of sale','gross square feet', 'land square feet','lot_area_commercial_office',	'lot_area_industrial_manufacturing',	'lot_area_mixed_use',	'lot_area_open_space',	'lot_area_other_no_data',	'lot_area_parking',	'lot_area_public_facility_institution',	'lot_area_res_1_2_family_bldg',	'lot_area_res_multifamily_elevator',	'lot_area_res_multifamily_walkup'	,'lot_area_transportation_utility', 'lot_area_vacant'
,'pct_hh_rent_burd', 'pop_change_00_10', 'poverty_rate', 'total_lot_area']]
NYC1 = pd.get_dummies(NYC1, columns=['cd_short_title', 'borough', 'building class at time of sale', 'building class category',  'tax class at time of sale'])




######################################################

NYC1 = data.loc[:, ['sale_price',  'borough', 'crime_count', 'residential units','commercial units', 'year built', 'cd_short_title', 'building class at time of sale', 'building_class_category', 'tax class at time of sale','gross square feet', 'land square feet','lot_area_commercial_office',	'lot_area_industrial_manufacturing',	'lot_area_mixed_use',	'lot_area_open_space',	'lot_area_other_no_data',	'lot_area_parking',	'lot_area_public_facility_institution',	'lot_area_res_1_2_family_bldg',	'lot_area_res_multifamily_elevator',	'lot_area_res_multifamily_walkup'	,'lot_area_transportation_utility', 'lot_area_vacant'
,'pct_hh_rent_burd', 'pop_change_00_10', 'poverty_rate', 'total_lot_area']]
NYC1 = pd.get_dummies(NYC1, columns=['cd_short_title', 'borough', 'building class at time of sale', 'building_class_category',  'tax class at time of sale'])
NYC1 = NYC1.loc[:,['sale_price','tax class at time of sale_4','tax class at time of sale_2','tax class at time of sale_1','building_class_category_44_CONDO_PARKING',
'building_class_category_45_CONDO_HOTEL','building_class_category_46_CONDO_STORE_BUILDINGS',
'building_class_category_36_OUTDOOR_RECREATIONAL_FACILITIES',
'building_class_category_30_WAREHOUSES','building_class_category_29_COMMERCIAL_GARAGES','building_class_category_22_STORE_BUILDINGS',
'building_class_category_21_OFFICE_BUILDINGS','building_class_category_05_TAX_CLASS_1_VACANT_LAND','building_class_category_01_ONE_FAMILY_DWELLINGS','borough_3','borough_2','borough_1','cd_short_title_Brooklyn CD 15',
'cd_short_title_Brooklyn CD 12','cd_short_title_Bronx CD 1','total_lot_area','pop_change_00_10','lot_area_vacant','lot_area_transportation_utility',
'lot_area_res_multifamily_walkup','lot_area_res_multifamily_elevator','lot_area_res_1_2_family_bldg',
'lot_area_public_facility_institution','lot_area_open_space','lot_area_industrial_manufacturing',
'lot_area_commercial_office','gross square feet','year built','commercial units','residential units']]




X, y = NYC1.iloc[:, 1:], NYC1.sale_price
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)  


# RMSE
def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))

print("ExtraTrees:")  #now score of 83
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor(random_state=1)
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)  
print(score)
rmse(y_test,y_pred)


print("Gbooster:")  #cv not bad .20
from sklearn.ensemble import GradientBoostingRegressor
model= GradientBoostingRegressor()
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)  
print(score)
rmse(y_test,y_pred)

print("BagginRF:") #cross val is bad (neg)
from sklearn.ensemble import RandomForestRegressor
model = BaggingRegressor(RandomForestRegressor(random_state=1))
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)  
print(score)

print("NNET:")
model = MLPRegressor(alpha=.00001,max_iter=5000) #cv of 18 
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)  
print(score)

from sklearn.neighbors import KNeighborsRegressor #weak on cv .05
model = KNeighborsRegressor(5)
model.fit(X_train, y_train)  
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)
print(score)
rmse(y_test,y_pred)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)  
y_pred = model.predict(X_test)
print(score)
rmse(y_test,y_pred)

from sklearn.linear_model import Ridge
model = Ridge()
model.fit(X_train,y_train)
score = model.score(X_test,y_test)  
y_pred = model.predict(X_test)
print(score)
rmse(y_test,y_pred)

from sklearn.linear_model import Lasso
model = Lasso()
model.fit(X_train,y_train)
score = model.score(X_test,y_test)  
y_pred = model.predict(X_test)
print(score)
rmse(y_test,y_pred)


from sklearn.model_selection import KFold, cross_val_score

kfold = model_selection.KFold(n_splits=10, random_state=1)
cv_results01 = model_selection.cross_val_score(model, X_test,y_test, cv=kfold, scoring='r2')
cv_results02 = model_selection.cross_val_score(model, X_test,y_test, cv=kfold, scoring='neg_mean_squared_error')
cv_results03 = model_selection.cross_val_score(model, X_test,y_test, cv=kfold, scoring='neg_mean_absolute_error')

cv_results01.mean()
cv_results02.mean()
cv_results03.mean()

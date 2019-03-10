import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

NYC = pd.DataFrame(pd.read_csv('NYC_final_data_abridged_.csv'))
NYC.columns = NYC.columns.str.strip().str.lower().str.replace(' ', '_')

del NYC['apartment_number']
del NYC['address']
del NYC['ease-ment']
del NYC['unnamed:_0']
del NYC['cartodb_id']
del NYC['borocd']
del NYC['cd_full_title']
del NYC['lot']
del NYC['zip_code']
del NYC['shared_puma']

NYC = NYC.drop(columns=['acs_tooltip','cb_email','cb_website', 'cd_son_fy2018', 'moe_bach_deg',	'moe_bach_deg_boro',	'moe_bach_deg_nyc',	'moe_foreign_born',	'moe_hh_rent_burd',	'moe_hh_rent_burd_boro',	'moe_hh_rent_burd_nyc',	'moe_lep_rate',	'moe_lep_rate_boro',	'moe_lep_rate_nyc',	'moe_mean_commute',	'moe_mean_commute_boro',	'moe_mean_commute_nyc',	'moe_over65_rate', 'moe_over65_rate_boro',	'moe_over65_rate_nyc',	'moe_poverty_rate',	'moe_under18_rate',	'moe_unemnployment',	'moe_unemployment_boro',	'moe_unemployment_cd', 'moe_under18_rate_boro', 'moe_under18_rate_nyc',	'neighborhood',	'neighborhoods'], axis = 1)
NYC = NYC.drop(columns=['shared_puma_cd', 'son_issue_1', 'son_issue_2', 'son_issue_3', 'the_geom', 'the_geom_webmercator', 'puma10', 'pop_acs'])
NYC1 = pd.get_dummies(NYC, columns=['cd_short_title', 'borough','building_class_at_present', 'building_class_at_time_of_sale', 'building_class_category',  'tax_class_at_time_of_sale', 'tax_class_at_present'])
NYC1 = NYC1.drop(columns = ['sale_date'], axis =1)
NYC1 = NYC1.drop(index = 56791) #issue data point

nullcols = NYC1.columns[NYC1.isnull().any()]
NYC1 = NYC1.drop(columns = nullcols)

''' #ATTEMPTED TO LOOK AT ONLY MONTHLY SUBSET
NYC1['sale_date'] = pd.to_datetime(NYC1['sale_date'])
NYC1['sale_year'] = NYC1['sale_date'].dt.year
NYC1['sale_month'] = NYC1['sale_date'].dt.month
NYC_Oct = NYC1[(NYC1['sale_month'] == 10) & (NYC1['sale_year'] == 2018)]
NYC_Oct = NYC_Oct.drop(columns = ['sale_date', 'sale_month', 'sale_year'], axis = 1)
'''

####PLOTTING DATA##
# Top 20 Heatmap
k = 20
corrmat = NYC1.corr()
cols = corrmat.nlargest(k, 'sale_price')['sale_price'].index
cm = np.corrcoef(NYC1[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


#Univariate regressions with highly correlated data points
sns.jointplot(x=NYC1['gross_square_feet'], y=NYC1['sale_price'], kind='reg')  #key variable
sns.jointplot(x=(NYC1['residential_units']), y=NYC1['sale_price'], kind='reg')
sns.jointplot(x=NYC1['acres'], y=NYC1['sale_price'], kind='reg')
sns.jointplot(x=(NYC1['commercial_units']), y=NYC1['sale_price'], kind='reg') # key variable
sns.jointplot(x=(NYC1['total_units']), y=NYC1['sale_price'], kind='reg') # key variable


#More visualizations
sns.distplot(NYC1['sale_price'])
print(NYC1['sale_price'].skew())
logsales = np.log(NYC1['sale_price'])
sns.distplot(logsales)
print(logsales.skew())


sns.boxplot(x='gross_square_feet', data=NYC1,showfliers=False)
sns.boxplot(x='total_units', data=NYC1,showfliers=False)
sns.boxplot(x='land_square_feet', data=NYC1,showfliers=False)

NYC1 = NYC1[NYC1['land_square_feet'] != 0]
NYC1 = NYC1[NYC1['gross_square_feet'] != 0]
NYC1 = NYC1[NYC1['total_units'] != 0]


plt.figure(figsize=(10,6))
sns.boxplot(x='total_units', y='sale_price', data=NYC1)
plt.title('Total Units vs Sale Price')
plt.show()


sns.boxplot(x='commercial_units', y='sale_price', data=NYC1)
plt.title('Commercial Units vs Sale Price')
plt.show()


sns.boxplot(x='residential_units', y='sale_price', data=NYC1)
plt.title('Residential Units vs Sale Price')
plt.show()


sns.barplot(x ='tax_class_at_time_of_sale', y='sale_price', data=NYC)#use NYC dataframe because we one hot coded these variables in NYC1 already
plt.show()


sns.barplot(x ='tax_class_at_present', y='sale_price', data=NYC) #use NYC dataframe because we one hot coded these variables in NYC1 already
plt.show()



####modelling### If Random Forest Ends Up being solid, may want to see how it performs without scaling..

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
X, y = np.array(NYC1.iloc[:, 1:]), np.array(NYC1.sale_price)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
kf.get_n_splits(X)
print(kf)
cvtest = []
cvtrain = []
for train_index, test_index in kf.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   scalerx = StandardScaler().fit(X_train)
   X_test = scalerx.transform(X_test)
   X_train = scalerx.transform(X_train)
   scalery = StandardScaler().fit(np.array(y_train).reshape(-1, 1))
   y_test = scalery.transform(np.array(y_test).reshape(-1, 1))
   y_train = scalery.transform(np.array(y_train).reshape(-1, 1))
   model = RandomForestRegressor(max_depth=15, max_features="sqrt", bootstrap=True, n_estimators=1500, random_state=42)
   model.fit(X_train, y_train)
   testscores = model.score(X_test, y_test)
   trainscores = model.score(X_train, y_train)
   print('Training Score:')
   print(trainscores)
   print('Testing Score:')
   print(testscores)
   cvtest.append(testscores)
   cvtrain.append(testscores)

print('Mean CV Test Score:')
print(np.mean(np.array(cvtest)))


from sklearn.ensemble import ExtraTreesRegressor
kf = KFold(n_splits=10, shuffle=True, random_state=42)
kf.get_n_splits(X)
print(kf)
cvtest = []
cvtrain = []
for train_index, test_index in kf.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   scalerx = StandardScaler().fit(X_train)
   X_test = scalerx.transform(X_test)
   X_train = scalerx.transform(X_train)
   scalery = StandardScaler().fit(np.array(y_train).reshape(-1, 1))
   y_test = scalery.transform(np.array(y_test).reshape(-1, 1))
   y_train = scalery.transform(np.array(y_train).reshape(-1, 1))
   model = ExtraTreesRegressor(max_depth=10, max_features="sqrt", n_estimators=2000, random_state=42)
   model.fit(X_train, y_train)
   testscores = model.score(X_test, y_test)
   trainscores = model.score(X_train, y_train)
   print('Training Score:')
   print(trainscores)
   print('Testing Score:')
   print(testscores)
   cvtest.append(testscores)
   cvtrain.append(testscores)


print('Mean CV Test Score:')
print(np.mean(np.array(cvtest)))


#Look at data once last time now scaled
from scipy.stats import skew
skew(X_test)
skew(y_test)
skew(X_train)
skew(y_train)
sns.distplot(y_test)
sns.distplot(y_train) #if data results are poor we can subset to a price range... and remove many of the outliers
##Another interesting manipulation would be to calculate price per gross sq feet


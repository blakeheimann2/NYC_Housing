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

X, y = NYC1.iloc[:, 1:], NYC1.sale_price
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#y_test = np.log(y_test) gives shitty results
#y_train = np.log(y_train)
scalery =  StandardScaler().fit(np.array(y_train).reshape(-1, 1))
y_test = scalery.transform(np.array(y_test).reshape(-1, 1))
y_train = scalery.transform(np.array(y_train).reshape(-1, 1))

#Look at data once last time now scaled
from scipy.stats import skew
skew(X_test)
skew(y_test)
skew(X_train)
skew(y_train)
sns.distplot(y_test)
sns.distplot(y_train) #if data results are poor we can subset to a price range... and remove many of the outliers
##Another interesting manipulation would be to calculate price per gross sq feet

'''#LINEAR REGRESSION WITH OUTPUT INFO
import statsmodels.api as sm
model = sm.OLS(y_train, X_train).fit()
model.summary()'''


#Lets run a few models... just to explore..
print("Gbooster:")  #..36; cv gives 56 #terrible when scaled
from sklearn.ensemble import GradientBoostingRegressor
model= GradientBoostingRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('MSE')
print(mean_squared_error(y_test, y_pred))
print("MAE")
print(mean_absolute_error(y_test, y_pred))
print("R^2")
print(r2_score(y_test, y_pred))


from sklearn.ensemble import RandomForestRegressor #good when all vars are scaled
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('MSE')
print(mean_squared_error(y_test, y_pred))
print("MAE")
print(mean_absolute_error(y_test, y_pred))
print("R^2")
print(r2_score(y_test, y_pred))

from sklearn.neural_network import MLPRegressor
print("NNET:")
model = MLPRegressor(alpha=.00001,max_iter=5000) #57; cv .01
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('MSE')
print(mean_squared_error(y_test, y_pred))
print("MAE")
print(mean_absolute_error(y_test, y_pred))
print("R^2")
print(r2_score(y_test, y_pred))

print("KNN")
from sklearn.neighbors import KNeighborsRegressor #23, cv 13
model = KNeighborsRegressor(10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('MSE')
print(mean_squared_error(y_test, y_pred))
print("MAE")
print(mean_absolute_error(y_test, y_pred))
print("R^2")
print(r2_score(y_test, y_pred))
'''
print('SVR') #also bad on CV (neg)
model = SVR(kernel = "linear", gamma=.1, C=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('MSE')
print(mean_squared_error(y_test, y_pred))
print("MAE")
print(mean_absolute_error(y_test, y_pred))
print("R^2")
print(r2_score(y_test, y_pred))
'''
print("ExtraTrees:")  #25
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('MSE')
print(mean_squared_error(y_test, y_pred))
print("MAE")
print(mean_absolute_error(y_test, y_pred))
print("R^2")
print(r2_score(y_test, y_pred))

print("Ridge Regression:")  ##GOOD SCORE BUT PROBABLY SIGNIFICANT COLINEARITY BETWEEN EXOGENOUS VARIABLES
from sklearn.linear_model import Ridge
model = Ridge()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('MSE')
print(mean_squared_error(y_test, y_pred))
print("MAE")
print(mean_absolute_error(y_test, y_pred))
print("R^2")
print(r2_score(y_test, y_pred))

print("ExtraTrees:")  #25
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('MSE')
print(mean_squared_error(y_test, y_pred))
print("MAE")
print(mean_absolute_error(y_test, y_pred))
print("R^2")
print(r2_score(y_test, y_pred))

##NOT FINALIZED BELOW
'''

###############################
#NOW LETS ITERATE THROUGH USING CROSS VALIDATION AND PLOT R2 VALUES##
#Be prepared this can take a while - maybe 20 min or more...
scoring = 'r2'

print("creating list of models")
#Create a list of models that will run in turn to get output and later determine best model
models = []
#models.append(('LR', LogisticRegression()))
models.append(('Ridge', Ridge()))
#models.append(('LDA', LinearDiscriminantAnalysis())) commenting out due to colinearity
#models.append(('SVR', SVR(kernel = "linear", gamma=.1, C=10)))
#models.append(('KNN', KNeighborsRegressor(n_neighbors=10))) Would include but takes waaaaay too long
#models.append(('Bagged-CART', BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))))
#models.append(('NB', GaussianNB())) #data supposed to be normal
models.append(('RF', RandomForestRegressor(random_state=42)))
models.append(('NNET', MLPRegressor(alpha=.00001,max_iter=5000))) #takes a while to run but good model
#models.append(('ABoost', AdaBoostRegressor()))
models.append(('Gboost', GradientBoostingRegressor())) #takes a while to run
#models.append(('BNB', BernoulliNB()))
#models.append(('MNB', MultinomialNB()))
models.append(('ExTrees', ExtraTreesRegressor(random_state=1))) #terrible negative with CV

results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=0)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure()
fig.suptitle('R^2 Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(cv_results)
ax.set_xticklabels(names)
plt.show()


####Results
###SAY WE WANT TO ENSEMBLE MULTIPLE MODELS... THOUGH FOR THIS USE CASE, IT IS PROB NOT NECESSARY

#################BLENDING### Does not work right now

X, y = NYC_Oct.iloc[:, 1:], NYC_Oct.sale_price

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

X_val = X_val.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


model1 = BaggingRegressor(RandomForestRegressor(random_state=1))
model1.fit(X_train, y_train)
val_pred1=model1.predict(X_val)
test_pred1=model1.predict(X_test)
val_pred1=pd.DataFrame(val_pred1)
test_pred1=pd.DataFrame(test_pred1)

model2 = KNeighborsRegressor(10)
model2.fit(X_train,y_train)
val_pred2=model2.predict(X_val)
test_pred2=model2.predict(X_test)
val_pred2=pd.DataFrame(val_pred2)
test_pred2=pd.DataFrame(test_pred2)

model3 = GradientBoostingRegressor()
model3.fit(X_train, y_train)
val_pred3=model3.predict(X_val)
test_pred3=model3.predict(X_test)
val_pred3=pd.DataFrame(val_pred3)
test_pred3=pd.DataFrame(test_pred3)


#df_val=pd.concat([X_val, val_pred1,val_pred2, val_pred3],axis=1,join_axes=[X_val.index])
#df_test=pd.concat([X_test, test_pred1,test_pred2, test_pred3],axis=1,join_axes=[X_test.index])

#df_val=pd.concat([X_val.loc[:,['gross_square_feet', 'commercial_units','residential_units']], val_pred1,val_pred2, val_pred3],axis=1,join_axes=[X_val.index])
#df_test=pd.concat([X_test.loc[:,['gross_square_feet', 'commercial_units','residential_units']], test_pred1,test_pred2, test_pred3],axis=1,join_axes=[X_test.index])

#the best model comes from only using the outputs for regression
df_val=pd.concat([val_pred1, val_pred3],axis=1,join_axes=[X_val.index])
df_test=pd.concat([test_pred1, test_pred3],axis=1,join_axes=[X_test.index])

df_val.shape
df_test.shape
y_test.shape


#df_val = df_val.dropna()    #Gboost has one NAN at index 24902, so we need to drop that data point on all DFs
#df_test = df_test.dropna()
#y_test = y_test.drop(index=11375)
y_test = y_test.drop(index=11375)
df_test = df_test.drop(index=11375)


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
model = LinearRegression()
model.fit(df_val,(y_val))
model.score(df_test,(y_test))  
y_pred = model.predict(df_test)
####if we standardize and use KNN we get 45 R2, if we use linear we get 

print('MSE')
print(mean_squared_error(y_test, y_pred))
print("MAE")
print(mean_squared_error(y_test, y_pred))
print("R^2")
print(r2_score(y_test, y_pred))
print("explained Var")
print(explained_variance_score(y_test, y_pred))
print("median absolute error")
print(median_absolute_error(y_test, y_pred))

from sklearn.model_selection import KFold

kfold = model_selection.KFold(n_splits=10, random_state=0)
cv_results01 = model_selection.cross_val_score(model, df_test,y_test, cv=kfold, scoring='r2')
cv_results02 = model_selection.cross_val_score(model, df_test,y_test, cv=kfold, scoring='neg_mean_squared_error')
cv_results03 = model_selection.cross_val_score(model, df_test,y_test, cv=kfold, scoring='neg_mean_absolute_error')

#Create Dataframe of results to export
array1 = np.array(['Blended KNN & ExTrees', cv_results01.mean(), cv_results01.std(), cv_results02.mean(), cv_results02.std(), cv_results03.mean(), cv_results03.std()])
array2 = np.array(['Model' ,'R^2', 'R^2_Stdev','Neg_MSE','Neg_MSE_Stdev', 'Neg_MAE', 'Neg_MAE_Stdev'  ])
BlendResults0 = pd.DataFrame(array1).transpose()
BlendCols = pd.DataFrame(array2).transpose()
BlendResults = BlendCols.append(BlendResults0)
headers = BlendResults.iloc[0]
BlendResults = pd.DataFrame(BlendResults.values[1:], columns=headers)

'''

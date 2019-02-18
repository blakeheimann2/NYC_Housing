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

#######################


NYC = pd.DataFrame(pd.read_csv('NYC_final_data.csv'))
NYC1 = NYC.loc[:, ['sale_price',  'borough', 'crime_count', 'residential units','commercial units', 'year built', 'cd_short_title', 'building class at time of sale', 'building_class_category', 'tax class at time of sale','gross square feet', 'land square feet','lot_area_commercial_office',	'lot_area_industrial_manufacturing',	'lot_area_mixed_use',	'lot_area_open_space',	'lot_area_other_no_data',	'lot_area_parking',	'lot_area_public_facility_institution',	'lot_area_res_1_2_family_bldg',	'lot_area_res_multifamily_elevator',	'lot_area_res_multifamily_walkup'	,'lot_area_transportation_utility', 'lot_area_vacant'
,'pct_hh_rent_burd', 'pop_change_00_10', 'poverty_rate', 'total_lot_area']]
NYC1 = pd.get_dummies(NYC1, columns=['cd_short_title', 'borough', 'building class at time of sale', 'building_class_category',  'tax class at time of sale'])

#attempt modelling by boroughs... or attempt by building class type --- might want to include date.
# actually, lets model subsets by date

NYC1['sale_date'] = pd.to_datetime(NYC1['sale_date'])
NYC1['sale_year'] = NYC1['sale_date'].dt.year
NYC1['sale_month'] = NYC1['sale_date'].dt.month

NYC_Oct = NYC1[(NYC1['sale_month'] == 10) & (NYC1['sale_year'] == 2018)]
NYC_Oct = NYC_Oct.drop(columns = ['sale_date', 'sale_month', 'sale_year'], axis = 1)



NYC1 = NYC1.loc[:,['sale_price','tax class at time of sale_4','tax class at time of sale_2','tax class at time of sale_1','building_class_category_43_CONDO_OFFICE_BUILDINGS','building_class_category_44_CONDO_PARKING',
'building_class_category_45_CONDO_HOTEL','building_class_category_46_CONDO_STORE_BUILDINGS',
'building_class_category_36_OUTDOOR_RECREATIONAL_FACILITIES','building_class_category_31_COMMERCIAL_VACANT_LAND',
'building_class_category_30_WAREHOUSES','building_class_category_29_COMMERCIAL_GARAGES','building_class_category_22_STORE_BUILDINGS',
'building_class_category_21_OFFICE_BUILDINGS','building_class_category_05_TAX_CLASS_1_VACANT_LAND','building_class_category_01_ONE_FAMILY_DWELLINGS','borough_3','borough_2','borough_1','cd_short_title_Brooklyn CD 15',
'cd_short_title_Brooklyn CD 12','cd_short_title_Bronx CD 1','total_lot_area','pop_change_00_10','lot_area_vacant','lot_area_transportation_utility',
'lot_area_res_multifamily_walkup','lot_area_res_multifamily_elevator','lot_area_res_1_2_family_bldg',
'lot_area_public_facility_institution','lot_area_open_space','lot_area_industrial_manufacturing',
'lot_area_commercial_office','gross square feet','year built','commercial units','residential units']]

NYC_Oct.columns[NYC_Oct.isna().any()].tolist()

####PLOTTING DATA##
import seaborn as sns
corrmat = NYC1.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()
# Top 10 Heatmap

k = 20 #number of variables for heatmap
cols = corrmat.nlargest(k, 'sale_price')['sale_price'].index
cm = np.corrcoef(NYC1[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


sns.jointplot(x=NYC1['gross square feet'], y=NYC1['sale_price'], kind='reg')  #key variable
sns.jointplot(x=(NYC1['residential units']), y=NYC1['sale_price'], kind='reg')
sns.jointplot(x=NYC_Oct['acres'], y=NYC_Oct['sale_price'], kind='reg')
sns.jointplot(x=(NYC1['commercial units']), y=NYC1['sale_price'], kind='reg') # key variable
sns.jointplot(x=(NYC1['commercial units']), y=NYC1['sale_price'], kind='reg')
sns.jointplot(x=(NYC1['tax class at time of sale_4']), y=NYC1['sale_price'], kind='reg')

####modelling###
from sklearn.preprocessing import normalize
X, y = NYC1.iloc[:, 1:], NYC1.sale_price
X = normalize(X)


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)  

NYC_Oct = NYC_Oct[NYC_Oct['sale_price'] > 10000]

###################################3
import statsmodels.api as sm
model = sm.OLS(y_train, X_train).fit()
model.summary()


print("bagging tree:") #cross val is bad (neg)
from sklearn import tree
model = BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)  
print(score)

from sklearn.model_selection import KFold, cross_val_score

kfold = model_selection.KFold(n_splits=10, random_state=1)
cv_results01 = model_selection.cross_val_score(model, X_test,y_test, cv=kfold, scoring='r2')
cv_results02 = model_selection.cross_val_score(model, X_test,y_test, cv=kfold, scoring='neg_mean_squared_error')
cv_results03 = model_selection.cross_val_score(model, X_test,y_test, cv=kfold, scoring='neg_mean_absolute_error')

cv_results01.mean()
cv_results02.mean()
cv_results03.mean()

print("Gbooster:")  #.19
from sklearn.ensemble import GradientBoostingRegressor
model= GradientBoostingRegressor()
model.fit(X_train, y_train)
score = model.score(X_test,y_test)

y_pred = model.predict(X_test)  
print(score)

print("BagginRF:") #cross val is bad (neg)
from sklearn.ensemble import RandomForestRegressor
model = BaggingRegressor(RandomForestRegressor(random_state=1))
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)  
print(score)

print("NNET:")
model = MLPRegressor(alpha=.00001,max_iter=5000) #good model .78
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)  
print(score)

from sklearn.neighbors import KNeighborsRegressor #weak on cv .25
model = KNeighborsRegressor(5)
model.fit(X_train, y_train)  
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)
print(score)

print('SVR') #also bad on CV (neg)
model = SVR(kernel = "linear", gamma=.1, C=10)
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)  
print(score)


print("ExtraTrees:")  #now score of 83
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor(random_state=1)
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)  
print(score)

#score of 78
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
print(explained_variance_score(y_test, y_pred))
print(median_absolute_error(y_test, y_pred))


#for Random Forrest Features
feature_importances = pd.DataFrame(model.feature_importances_,index = X_train.columns, 
columns=['importance']).sort_values('importance',ascending=False)



print("starting to run the regression model now")

from sklearn.ensemble import BaggingRegressor # I think this bagging method can be used with various types of models

scoring = 'r2'

print("creating list of models") 
#Create a list of models that will run in turn to get output and later determine best model
models = []

#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis())) commenting out due to colinearity
#models.append(('SVM', SVR(kernel = "rbf", gamma=.1, C=10)))
models.append(('KNN', KNeighborsRegressor(n_neighbors=10,algorithm="kd_tree")))
#models.append(('Bagged-CART', BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))))
#models.append(('NB', GaussianNB())) #data supposed to be normal
#models.append(('RF', BaggingRegressor(RandomForestRegressor(random_state=1))))
#models.append(('NNET', MLPRegressor(alpha=.00001,max_iter=5000))) #takes a while to run but good model
#models.append(('ABoost', AdaBoostRegressor()))
#models.append(('Gboost', GradientBoostingRegressor())) #takes a while to run 
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


###################################3


print("bagging tree:") #cross val is bad (neg)
from sklearn import tree
model = BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)  
print(score)

from sklearn.model_selection import KFold, cross_val_score

kfold = model_selection.KFold(n_splits=10, random_state=0)
cv_results01 = model_selection.cross_val_score(model, X_test,y_test, cv=kfold, scoring='r2')
cv_results02 = model_selection.cross_val_score(model, X_test,y_test, cv=kfold, scoring='neg_mean_squared_error')
cv_results03 = model_selection.cross_val_score(model, X_test,y_test, cv=kfold, scoring='neg_mean_absolute_error')

cv_results01.mean()
cv_results02.mean()
cv_results03.mean()

print("Gbooster:")  #.19
from sklearn.ensemble import GradientBoostingRegressor
model= GradientBoostingRegressor()
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)  
print(score)

print("BagginRF:") #cross val is bad (neg)
from sklearn.ensemble import RandomForestRegressor
model = BaggingRegressor(RandomForestRegressor(random_state=1))
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)  
print(score)

print("NNET:")
model = MLPRegressor(alpha=.00001,max_iter=5000) #good model .78
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)  
print(score)

from sklearn.neighbors import KNeighborsRegressor #weak on cv .25
model = KNeighborsRegressor(5)
model.fit(X_train, y_train)  
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)
print(score)

print('SVR') #also bad on CV (neg)
model = SVR(kernel = "rbf", gamma=.1, C=10)
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)  
print(score)


print("ExtraTrees:")  #now score of 83
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor(random_state=1)
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)  
print(score)

#score of 78
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
print(explained_variance_score(y_test, y_pred))
print(median_absolute_error(y_test, y_pred))


#for Random Forrest Features
feature_importances = pd.DataFrame(model.feature_importances_,index = X_train.columns, 
columns=['importance']).sort_values('importance',ascending=False)

#################BLENDING### Does not work right now

X, y = NYC_Oct.iloc[:, 1:], NYC_Oct.sale_price

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

X_val = X_val.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


model1 = Ridge()
model1.fit(X_train, y_train)
val_pred1=model1.predict(X_val)
test_pred1=model1.predict(X_test)
val_pred1=pd.DataFrame(val_pred1)
test_pred1=pd.DataFrame(test_pred1)

model2 = MLPRegressor(alpha=.00001,max_iter=5000)
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


df_val=pd.concat([X_val, val_pred1,val_pred2, val_pred3],axis=1,join_axes=[X_val.index])
df_test=pd.concat([X_test, test_pred1,test_pred2, test_pred3],axis=1,join_axes=[X_test.index])

df_val = df_val.dropna()    #Gboost has one NAN at index 24902, so we need to drop that data point on all DFs
df_test = df_test.dropna()
y_test = y_test.drop(index=24902)

'''
df_val=pd.concat([X_val, val_pred1,val_pred2],axis=1)
df_test=pd.concat([X_test, test_pred1,test_pred2],axis=1)
'''
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(df_val,y_val)
model.score(df_test,y_test)  
y_pred = model.predict(df_test)


print('MSE')
print(mean_squared_error(y_test, y_pred))
print("MAE")
print(mean_absolute_error(y_test, y_pred))
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


##CV view
fig = plt.figure()
fig.suptitle('R^2 of Blended Model')
ax = fig.add_subplot(111)
plt.boxplot(cv_results01)
ax.set_xticklabels('Model')
plt.show()
fig.savefig('BlendR2.png')

fig = plt.figure()
fig.suptitle('MSE of Blended Model')
ax = fig.add_subplot(111)
plt.boxplot(cv_results02)
ax.set_xticklabels('Model')
plt.show()
fig.savefig('BlendMSE.png')

fig = plt.figure()
fig.suptitle('MAE of Blended Model')
ax = fig.add_subplot(111)
plt.boxplot(cv_results03)
ax.set_xticklabels('Model')
plt.show()
fig.savefig('BlendMAE.png')


'''
############################################

from sklearn.ensemble import VotingClassifier

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = MLPRegressor(alpha=.00001,max_iter=5000)
estimators.append(('MLP', model1))
model2 = ExtraTreesRegressor(random_state=1)
estimators.append(('Etrees', model2))
model3 = GradientBoostingRegressor()
estimators.append(('Gboost', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, y, cv=kfold)   ###error cannot cast float to int
print(results.mean())
'''


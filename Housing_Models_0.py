import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import BaggingRegressor # I think this bagging method can be used with various types of models
from sklearn import tree





pd.set_option('display.expand_frame_repr', False)


Manhattan = pd.DataFrame(pd.read_csv('rollingsales_manhattan.csv'))
Brooklyn = pd.DataFrame(pd.read_csv('rollingsales_brooklyn.csv'))
Queens = pd.DataFrame(pd.read_csv('rollingsales_queens.csv'))
Bronx = pd.DataFrame(pd.read_csv('rollingsales_bronx.csv'))
StatenI = pd.DataFrame(pd.read_csv('rollingsales_statenisland.csv'))

Zip = pd.DataFrame(pd.read_csv('zip_to_zcta10_nyc_revised.csv'))
ZCTAtoPUMA = pd.DataFrame(pd.read_csv('nyc_zcta10_to_puma10.csv'))
PUMAtoNeigh = pd.DataFrame(pd.read_csv('nyc_puma_neighborhood.csv'))

Indicators = pd.DataFrame(pd.read_csv('Neighborhood_Indicators_nycgov_nanszero.csv'))


#only for BK, QUeens, and Staten Island
Sales = Brooklyn.append(Queens)
Sales = Sales.append(StatenI)
Sales = Sales.append(Manhattan)
Sales = Sales.append(Bronx)
Sales.drop_duplicates() #total 81115 sales

#do not touch
Sales.columns = map(str.lower, Sales.columns)
Zip = Zip.rename(columns = {'zipcode' : 'zip code'})
Zip = Zip.rename(columns = {'zcta5' : 'zcta'})
ZCTAtoPUMA = ZCTAtoPUMA.rename(columns = {'zcta10' : 'zcta'})
Sales_zip = pd.merge(Sales, Zip, how='left', on='zip code')
Sales_zip_zcta = pd.merge(Sales_zip, ZCTAtoPUMA, how='left', on='zcta' )


Indicators = Indicators.rename(columns = {'puma':'puma10'})
Data = pd.merge(Sales_zip_zcta, Indicators, how='left', on='puma10')
#Data.to_csv('Salesdata.csv')

###Need to figure out Manhattan and Bronx b/c Dups
Data.shared_puma.describe()
#sort by shared puma....
Data.loc[Data['shared_puma'] != False, ['shared_puma']]

#This is final subset of non-shared Pumas... i.e. no dups
Nonshared = Data.loc[Data['shared_puma'] == False]
len(Nonshared) # 75491, 219 columns
Nonshared_final = Nonshared.drop([ 'zcta','ziptype', 'postalcity','bcode', 'note', 'Unnamed: 6', 'stateco', 'alloc', 'pumaname', 'nameshort', 'per_in_puma', 'per_of_puma'], axis=1)



##NOW sort all sales dataset by True or Nan for sharedpuma
###Then join that dataset to Indicators Manually..
#NANs are the dups from the final join before so we do not include those values
Shared = Data.loc[Data['shared_puma'] == True]

####Does not add up to correct total 
#Get Shared list and drop dups
newshared = Shared[['borough', 'neighborhood', 'building class category', 'tax class at present', 'block', 'lot', 'ease-ment', 'building class at present', 'address', 'apartment number', 'zip code', 'residential units', 'commercial units', 'total units', 'land square feet', 'gross square feet', 'year built', 'tax class at time of sale', 'building class at time of sale', ' sale price ', 'sale date']]
Shared = newshared.drop_duplicates() #gets 5624 -> results in correct total. Now need to manually join these##
Shared['cd_short_title'] = Shared['neighborhood']
#Shared.to_csv('Shared.csv')

#manually join the shared Puma districs.
#current final Data Sets - Nonshared - good to go; Shared - join manually and then good.

Manhattan_shared = Shared[Shared.borough == 1]
Bronx_shared = Shared[Shared.borough == 2]
#FIDI
Manhattan_shared['cd_short_title'][Manhattan_shared.cd_short_title == 'FINANCIAL'] = "Manhattan CD 1"
Manhattan_shared['cd_short_title'][Manhattan_shared.cd_short_title == 'CIVIC CENTER'] = "Manhattan CD 1"
Manhattan_shared['cd_short_title'][Manhattan_shared.cd_short_title == 'TRIBECA'] = "Manhattan CD 1"

#GREENWICH
Manhattan_shared['cd_short_title'][Manhattan_shared.cd_short_title == 'GREENWICH VILLAGE-CENTRAL'] = "Manhattan CD 2"
Manhattan_shared['cd_short_title'][Manhattan_shared.cd_short_title == 'GREENWICH VILLAGE-WEST'] = "Manhattan CD 2"
Manhattan_shared['cd_short_title'][Manhattan_shared.cd_short_title == 'SOHO'] = "Manhattan CD 2"
Manhattan_shared['cd_short_title'][Manhattan_shared.cd_short_title == 'LITTLE ITALY'] = "Manhattan CD 2"

#Clinton/Chelsea
Manhattan_shared['cd_short_title'][Manhattan_shared.cd_short_title == 'CHELSEA'] = "Manhattan CD 4"
Manhattan_shared['cd_short_title'][Manhattan_shared.cd_short_title == 'CLINTON'] = "Manhattan CD 4"
Manhattan_shared['cd_short_title'][Manhattan_shared.cd_short_title == 'JAVITS CENTER'] = "Manhattan CD 4"

#LES/Chinatown
Manhattan_shared['cd_short_title'][Manhattan_shared.cd_short_title == 'CHINATOWN'] = "Manhattan CD 3"
Manhattan_shared['cd_short_title'][Manhattan_shared.cd_short_title == 'SOUTHBRIDGE'] = "Manhattan CD 3"

#MIDTOWN
Manhattan_shared['cd_short_title'][Manhattan_shared.cd_short_title == 'MIDTOWN WEST'] = "Manhattan CD 5"
Manhattan_shared['cd_short_title'][Manhattan_shared.cd_short_title == 'FLATIRON'] = "Manhattan CD 5"
Manhattan_shared['cd_short_title'][Manhattan_shared.cd_short_title == 'FASHION'] = "Manhattan CD 5"

#UWS
Manhattan_shared['cd_short_title'][Manhattan_shared.cd_short_title == 'UPPER WEST SIDE (59-79)'] = "Manhattan CD 7"
#Manhattan_shared.to_csv('Manhattan_shared.csv')

#join on neighborhood to Indicators
Manhattan_final = pd.merge(Manhattan_shared, Indicators, how="left", on='cd_short_title')
#results in 4550 rows x 207 col

##Manually Join the Bronx Hoods

#Bronx Hoods to CDs
Bronx_shared['cd_short_title'][Bronx_shared.cd_short_title == 'BATHGATE'] = "Bronx CD 3"
Bronx_shared['cd_short_title'][Bronx_shared.cd_short_title == 'BAYCHESTER'] = "Bronx CD 12"
Bronx_shared['cd_short_title'][Bronx_shared.cd_short_title == 'BELMONT'] = "Bronx CD 6"

Bronx_shared['cd_short_title'][Bronx_shared.cd_short_title == 'BRONXDALE'] = "Bronx CD 12"
Bronx_shared['cd_short_title'][Bronx_shared.cd_short_title == 'CROTONA PARK'] = "Bronx CD 3"
Bronx_shared['cd_short_title'][Bronx_shared.cd_short_title == 'EAST TREMONT'] = "Bronx CD 6"
Bronx_shared['cd_short_title'][Bronx_shared.cd_short_title == 'HUNTS POINT'] = "Bronx CD 2"


Bronx_shared['cd_short_title'][Bronx_shared.cd_short_title == 'MELROSE/CONCOURSE'] = "Bronx CD 4"
Bronx_shared['cd_short_title'][Bronx_shared.cd_short_title == 'MOTT HAVEN/PORT MORRIS'] = "Bronx CD 1"
Bronx_shared['cd_short_title'][Bronx_shared.cd_short_title == 'PARKCHESTER'] = "Bronx CD 9"
Bronx_shared['cd_short_title'][Bronx_shared.cd_short_title == 'MOUNT HOPE/MOUNT EDEN'] = "Bronx CD 4"

#Now for longwood and Morrisania by zip code..
Bronx_shared['cd_short_title'][(Bronx_shared['cd_short_title'] == 'MORRISANIA/LONGWOOD') & (Bronx_shared['zip code'] == 10451)] = 'Bronx CD 1'
Bronx_shared['cd_short_title'][(Bronx_shared['cd_short_title'] == 'MORRISANIA/LONGWOOD') & (Bronx_shared['zip code'] == 10454)] = 'Bronx CD 2'
Bronx_shared['cd_short_title'][(Bronx_shared['cd_short_title'] == 'MORRISANIA/LONGWOOD') & (Bronx_shared['zip code'] == 10455)] = 'Bronx CD 1'
Bronx_shared['cd_short_title'][(Bronx_shared['cd_short_title'] == 'MORRISANIA/LONGWOOD') & (Bronx_shared['zip code'] == 10456)] = 'Bronx CD 3'
Bronx_shared['cd_short_title'][(Bronx_shared['cd_short_title'] == 'MORRISANIA/LONGWOOD') & (Bronx_shared['zip code'] == 10459)] = 'Bronx CD 2'
Bronx_shared['cd_short_title'][(Bronx_shared['cd_short_title'] == 'MORRISANIA/LONGWOOD') & (Bronx_shared['zip code'] == 10460)] = 'Bronx CD 3'

Bronx_final = pd.merge(Bronx_shared, Indicators, how="left", on='cd_short_title')

#Holy shit I hope this works.
#And it does: 1074 rows x207 columns


#now align all data
#Manhattan_final.to_csv('Manhattan_final.csv')
#Bronx_final.to_csv('Bronx_final.csv')
#Nonshared_final.to_csv('Nonshared_final.csv')


#Need to align all data so columns are in the correct order... then append to the bottom
Bronx_final = Bronx_final.sort_index(axis = 1, ascending = 'True')
Manhattan_final = Manhattan_final.sort_index(axis = 1, ascending = 'True')
Nonshared_final = Nonshared_final.sort_index(axis = 1, ascending = 'True')

Final_data = Nonshared_final
Final_data = Final_data.append(Manhattan_final)
Final_data = Final_data.append(Bronx_final)
describe = Final_data.describe()
#describe.to_csv('Final_data_describe.csv')

#remove nulls from Sale Price by first getting rid of symbols in string then convert to numeric
Final_data[' sale price ']  = Final_data[' sale price '].str.replace(r'\D', '')
Final_data[' sale price '] = pd.to_numeric(Final_data[' sale price '])
#drop where is null
#need to double check this, cuts data nearly in half
df = Final_data.dropna(axis=0, subset=[' sale price '])
df = df[df[" sale price "] != 1]
df[' sale price '].describe()
NYC = df.rename(columns = {'sale date' : 'sale_date'})
NYC = NYC.rename(columns = {' sale price ' : 'sale_price'})
NYC.to_csv('NYC_final_data.csv')
#Gives final data points to 56876

'''
#drop completely irrelevant data for Modelling
NYC.drop(['cb_email', 'sale_date', 'ease-ment', 'cb_website','cd_full_title', 'acs_tooltip', 'neighborhoods','son_issue_1', 'son_issue_2', 'son_issue_3', 'shared_puma_cd', 'address', 'apartment number', 'cd_son_fy2018' ], axis=1, inplace=True)
NYC.drop(['the_geom','the_geom_webmercator', 'moe_under18_rate_nyc', 'moe_under18_rate_boro', 'pop_acs'], axis=1, inplace=True)
#additional removal
NYC.drop(['shared_puma', 'moe_poverty_rate', 'poverty_rate_boro', 'poverty_rate_nyc'], axis=1,inplace=True)
NYC.drop(['moe_bach_deg', 'pct_bach_deg_boro', 'moe_bach_deg_boro', 'pct_bach_deg_nyc', 'moe_bach_deg_nyc', 'moe_unemployment_cd', 'unemployment_boro', 'moe_unemployment_boro', 'unemployment_nyc', 'moe_unemnployment'], axis=1, inplace=True)
NYC.drop(['moe_mean_commute', 'mean_commute_boro', 'moe_mean_commute_boro', 'mean_commute_nyc', 'moe_mean_commute_nyc', 'moe_hh_rent_burd', 'pct_hh_rent_burd_boro', 'moe_hh_rent_burd_boro', 'pct_hh_rent_burd_nyc', 'moe_hh_rent_burd_nyc', 'pct_clean_strts_boro', 'pct_clean_strts_nyc', 'crime_per_1000_boro', 'crime_per_1000_nyc', 'crime_count_boro', 'crime_count_nyc'], axis=1,inplace=True)
NYC.drop(['lots_other_no_data', 'lots_total', 'lot_area___res_1_2_family_bldg', 'lot_area___res_multifamily_walkup', 'lot_area___res_multifamily_elevator', 'lot_area___mixed_use', 'lot_area___commercial_office', 'lot_area___industrial_manufacturing', 'lot_area___transportation_utility', 'lot_area___public_facility_institution', 'lot_area___open_space', 'lot_area___parking', 'lot_area___vacant', 'lot_area___other_no_data', 'lot_area___other_no_data', 'pct_lot_area___other_no_data', 'total_lot_area', 'neighborhoods'],axis=1, inplace=True)
NYC.drop(['moe_foreign_born', 'moe_lep_rate', 'lep_rate_boro', 'moe_lep_rate_boro', 'lep_rate_nyc', 'moe_lep_rate_nyc', 'moe_under18_rate', 'under18_rate_boro', 'moe_under18_rate_boro', 'under18_rate_nyc', 'moe_under18_rate_nyc', 'moe_over65_rate', 'over65_rate_boro', 'moe_over65_rate_boro', 'over65_rate_nyc', 'moe_over65_rate_nyc'], axis=1, inplace=True)
'''
'''
#Select Variables to include in model
NYC1 =NYC.loc[:, ['sale_price', 'year built', 'cd_short_title', 'building class at present', 'building class at time of sale', 'building class category', 'tax class at present', 'tax class at time of sale','gross square feet']]

#One-hot code the Categorical Data -CD short/full title, neighborhood, building class, building class at time of sale, 'tax class at present', 'tax class at time of sale'

NYC1 = pd.get_dummies(NYC1, columns=['cd_short_title', 'building class at present', 'building class at time of sale', 'building class category', 'tax class at present', 'tax class at time of sale'])
##NYC1.to_csv('NYC_data.csv')
print("done with preprocessing")

###NOW TIME TO MODEL#####

NYC1 = NYC1.sample(frac=.05)
NYC1.sale_price = np.log(NYC1.sale_price)

X, y = NYC1.iloc[:, 1:], NYC1.sale_price

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  


print("starting to run the regression model now")

from sklearn.ensemble import BaggingRegressor # I think this bagging method can be used with various types of models

print("bagging tree:")
from sklearn import tree
model = BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)  
print(score)

print("Gbooster:")
from sklearn.ensemble import GradientBoostingRegressor
model= GradientBoostingRegressor()
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
y_pred = model.predict(X_test)  
print(score)

print("BagginRF:")
from sklearn.ensemble import RandomForestRegressor
model = BaggingRegressor(RandomForestRegressor(random_state=1))
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




scoring = 'r2'

print("creating list of models") 
#Create a list of models that will run in turn to get output and later determine best model
models = []

#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis())) commenting out due to colinearity
models.append(('SVM', SVR(kernel = "rbf", gamma=.1, C=10)))
models.append(('KNN', KNeighborsRegressor(n_neighbors=10,algorithm="kd_tree")))
#models.append(('Bagged-CART', BaggingRegressor(DecisionTreeRegressor(min_samples_split=40, criterion="entropy"))))
#models.append(('NB', GaussianNB())) #data supposed to be normal
models.append(('RF', RandomForestRegressor(min_samples_split = 10, n_estimators=100, random_state=0)))
models.append(('NNET', MLPRegressor(alpha=.0000001,max_iter=500))) #takes a while to run but good model
#models.append(('ABoost', AdaBoostRegressor()))
#models.append(('Gboost', GradientBoostingRegressor())) #takes a while to run 
#models.append(('BNB', BernoulliNB()))
#models.append(('MNB', MultinomialNB()))


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
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
'''


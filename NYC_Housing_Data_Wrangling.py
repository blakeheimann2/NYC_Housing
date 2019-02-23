import pandas as pd
import numpy as np
pd.set_option('display.expand_frame_repr', False)

#Import Datasets for manipulation and aggregation
Manhattan = pd.DataFrame(pd.read_csv('rollingsales_manhattan.csv'))
Brooklyn = pd.DataFrame(pd.read_csv('rollingsales_brooklyn.csv'))
Queens = pd.DataFrame(pd.read_csv('rollingsales_queens.csv'))
Bronx = pd.DataFrame(pd.read_csv('rollingsales_bronx.csv'))
StatenI = pd.DataFrame(pd.read_csv('rollingsales_statenisland.csv'))
Zip = pd.DataFrame(pd.read_csv('zip_to_zcta10_nyc_revised.csv'))
ZCTAtoPUMA = pd.DataFrame(pd.read_csv('nyc_zcta10_to_puma10.csv'))
PUMAtoNeigh = pd.DataFrame(pd.read_csv('nyc_puma_neighborhood.csv'))
Indicators = pd.DataFrame(pd.read_csv('Neighborhood_Indicators_nycgov.csv'))

#Append Sales data for all boroughs
Sales = Brooklyn.append(Queens)
Sales = Sales.append(StatenI)
Sales = Sales.append(Manhattan)
Sales = Sales.append(Bronx)
Sales = Sales.drop_duplicates()
len(Sales) # final number we want 81115
#Manipulate the columns and column names to align for the join
Sales.columns = map(str.lower, Sales.columns)
Zip = Zip.rename(columns = {'zipcode' : 'zip code'})
Zip = Zip.rename(columns = {'zcta5' : 'zcta'})
ZCTAtoPUMA = ZCTAtoPUMA.rename(columns = {'zcta10' : 'zcta'})
Indicators = Indicators.rename(columns = {'puma':'puma10'})

#Join the Data Sets
Sales_zip = pd.merge(Sales, Zip, how='left', on='zip code')
Sales_zip_zcta = pd.merge(Sales_zip, ZCTAtoPUMA, how='left', on='zcta')
Data = pd.merge(Sales_zip_zcta, Indicators, how='left', on='puma10')
len(Data) #gives incorrect total so we have dups due to shared PUMA

#Now we need to split out the data between the shared PUMA and the nonshared to avoid dups
Nonshared = Data.loc[Data['shared_puma'] == False]
Nonshared_final = Nonshared.drop([ 'zcta','ziptype', 'postalcity','bcode', 'note', 'Unnamed: 6', 'stateco', 'alloc', 'pumaname', 'nameshort', 'per_in_puma', 'per_of_puma'], axis=1)
Shared = Data.loc[Data['shared_puma'] == True]
len(Nonshared) #75491, 219 columns
len(Shared) #incorrect


#Correct the Shared list and drop dups to get right total
newshared = Shared[['borough', 'neighborhood', 'building class category', 'tax class at present', 'block', 'lot', 'ease-ment', 'building class at present', 'address', 'apartment number', 'zip code', 'residential units', 'commercial units', 'total units', 'land square feet', 'gross square feet', 'year built', 'tax class at time of sale', 'building class at time of sale', ' sale price ', 'sale date']]
Shared = newshared.drop_duplicates() #gets 5624 -> results in correct total. Now need to manually join these##

#creat a new column for next steps of Joining manually
Shared['cd_short_title'] = Shared['neighborhood']

#Current Status: Nonshared - good to go; Shared - need to join manually
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

#join on neighborhood to Indicators
Manhattan_final = pd.merge(Manhattan_shared, Indicators, how="left", on='cd_short_title')
#results in 4550 rows x 207 col


##Manually Join the Bronx Neighborhoods
#Bronx neighborhoods to CDs
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

#Need to align all data so columns are in the correct order... then append all the data together
Bronx_final = Bronx_final.sort_index(axis = 1, ascending = 'True')
Manhattan_final = Manhattan_final.sort_index(axis = 1, ascending = 'True')
Nonshared_final = Nonshared_final.sort_index(axis = 1, ascending = 'True')
Final_data = Nonshared_final
Final_data = Final_data.append(Manhattan_final)
Final_data = Final_data.append(Bronx_final)
describe = Final_data.describe()

#remove nulls and low values from Sale Price by first getting rid of symbols in string then convert to numeric
Final_data[' sale price ']  = Final_data[' sale price '].str.replace(r'\D', '')
Final_data[' sale price '] = pd.to_numeric(Final_data[' sale price '])
df = Final_data.dropna(axis=0, subset=[' sale price '])
df = df[df[" sale price "] != 1]

#This cuts our data nearly in half
df[' sale price '].describe()
NYC = df.rename(columns = {'sale date' : 'sale_date'})
NYC = NYC.rename(columns = {' sale price ' : 'sale_price'})
NYC.to_csv('NYC_final_data_abridged_.csv')
#Gives final data points to 56876



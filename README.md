# NYC Property Sales:
## Overview, Data Wrangling, Visualizations and Model Predictions

### Overview
This Project's purpose is to formulate a machine learning model with reasonable predictive power in explaining real estate pricing in NYC.
I start by marrying neighborhood and demographic data to property sales data. From there, I start to clean and visualize the data before any modelling is done. Once the data is prepped, cleaned, and summarized in visualizations, I begin modelling using SciKit Learn. Python files are included in the Repo and some of the Tableau Visualizations are/will be shown below. The results of various models tend to be somewhat underwhelming, with R^2 values typically below .5 when cross validation is applied.Thus, this project will continue to undergo various iterations to yield improvements in predictive power.

### Data Wrangling
#### This has its own file, separate from the prep and modelling.
The data for this project is sourced from [NYU Furman Center](http://furmancenter.org/neighborhoods) and [NYC Department of Finance](https://www1.nyc.gov/site/finance/taxes/property-rolling-sales-data.page). The Furman Center supplies neighborhood indicator data that consists of information like poverty rate, population, crime, and other socioeconomic factors that may play a role in sale prices of a particular neighborhood. The NYC Department of Finance supplies rolling sales data, which give sale prices and a few other data points specific to the property being transacted. I take sales data for Dec 2017 through Nov 2018 and join it to the neighborhood (Community District) data from the Furman Center using Zip Codes and PUMA codes. These data points do not join flawlessly due to a few community districts sharing mutiple PUMAs. These are a minority and I did my best to manually map these.

### Data Cleaning 
#### The remaining work is done in the 'Prep_Viz_Modelling' file
Once the data has been put together in a consumable format, I start focusing on prepping it for use in visualizations and ML models. I find that there are a lot of nulls throughout the data set, and I still need to convert the categorical data using 'One-Hot-Coding'. This is completed, and several columns of data are dropped that introduce significant nulls or becuase they contain zero values, as well as one issue data point. Once dropped, I can begin modelling, but it is ideal to explore the data further using visualizations beforehand. Much of this is done in tableau, but some heatmaps, univariate regressions, and other visuals are produced in the python code. 


### Preparation and Modelling

The first thing I do is split the data into test and train sets. To further prep the data for modelling (outside of dropping issue data points with zeros or NANs) I start by scaling it using StandardScaler(). This is done on the train data, and the test data is then transformed using the parameters from the training set to maintain integrity. 

From here, I can start modelling, but I see that the data isnt near normal with large skew values. Without dropping outliers or additional data points, I the other option I have to attempt to better standardize is to take the log. In doing so, I found that the models performed more poorly, so I stuck with the current scaling.

### Modelling (Incomplete)

I start by throwing a few models such as random forrest, gradient boosting, K-nearest neighbors, as well as a ridge regression and multilayer perceptron without any specified hyperparameters just to see performance metrics on R^2, MSE, and MAE. The random forrest model seems promising with an R^2 of approx. .77, while the others have weak explanatory power at or below an R^2 of .30. 

**Next Steps:**
Currently, I have code that iterates through various models to test them (that takes way too long to run), as well as code to blend multiple models together in an attempt to yield better model performance. My plan is to explore blending these models, as well as use the GridSearchCV function from Sci-Kit Learn to find optimal hyperparameters (though this is computationally expensive).

*Due to my poor documentation with exploring log transformations for standardization I may also run that again*







# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 20:21:23 2022

@author: Utilisateur
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor




df = pd.read_csv("C:/Users/Utilisateur/ML Models_ Projects/complete_project/eda_data.csv")
df

# Choose relevant columns
df.columns

df_model = df[["Rating","Type of ownership",
               "Industry","Revenue","Sector",
                "Competitor_Count",
               "Employer_Provided", "same_state",
               "Size", "Average_Salary", "Hourly",
               "Company_age", "Spark_yn",
               "Job_State","Python_yn",
               "AWS", "Excel_yn",
               "Seniority", "Job_simp", "Desc_len"]]
df_model

# Get dummy variables
df_dum = pd.get_dummies(df_model)

# Train test split
X = df_dum.drop("Average_Salary", axis=1)
y = df_dum.Average_Salary.values

X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# multiple linear regression
""" Stast Models"""
X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)
model.fit().summary()

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

# Lasso regression bcz the data is so sparse avd we'll need tp normalize it
lm_l = Lasso(alpha=0.13)
lm_l.fit(X_train, y_train)
np.mean(cross_val_score(lm_l, X_train, y_train, 
                        scoring='neg_mean_absolute_error', cv=3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(
            lml, X_train, y_train,
            scoring='neg_mean_absolute_error', cv=3)))

plt.plot(alpha, error)

err = tuple(zip(alpha, error))
df_err = pd.DataFrame(err, columns=['alpha','error'])
df_err[df_err.error == max(df_err.error)]

# From the dataframe and the plot we can see that the peek is 0.13, which means it is the best value aloha can take

""" To run codes like jupyter in blocks use: #%% """

# Random Forest
rf = RandomForestRegressor()

np.mean(cross_val_score(rf, X_train, y_train,
                        scoring='neg_mean_absolute_error',
                        cv=3))

    
# Tune Models using gridsearchCV
params = {'n_estimators':range(10,300,10),
          'criterion':('mse','mae'),
          'max_features':('auto','sqrt','log2')}
gs = GridSearchCV(rf, params,
                  scoring='neg_mean_absolute_error', cv=3)

gs.fit(X_train, y_train)


gs.best_score_
gs.best_estimator_


# Test Ensembles

tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm)
mean_absolute_error(y_test,tpred_lml)
mean_absolute_error(y_test,tpred_rf)

mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)

import pickle
pickl = {'model': gs.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(np.array(list(X_test.iloc[1,:])).reshape(1,-1))[0]

list(X_test.iloc[1,:])
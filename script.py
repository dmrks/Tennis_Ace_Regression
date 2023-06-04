import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# load and investigate the data here:

df = pd.read_csv("tennis_stats.csv")
print(df.head())
print(df.columns)
print(df.dtypes)

# 3 perform exploratory analysis here:

features = df[['FirstServeReturnPointsWon']]
outcome = df[['Winnings']]

plt.scatter(features,outcome)
plt.show()
plt.clf()


## perform single feature linear regressions here
line_fitter = LinearRegression()

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)

line_fitter.fit(features_train, outcome_train)
line_fitter.score(features_test,outcome_test)

prediction = line_fitter.predict(features_test)

plt.scatter(outcome_test,prediction, alpha=0.4)
plt.show()
plt.clf()

#5 Build Model  Are BreakPointsOpportunities a Predictor for Wins?

feature2 = df[['BreakPointsOpportunities']]
outcome2 = df[['Wins']]

plt.scatter(feature2,outcome2)
plt.show()
plt.clf()

# Train and Fit Model

line_fitter = LinearRegression()

features_train, features_test, outcome_train, outcome_test = train_test_split(feature2, outcome2, train_size = 0.8)

line_fitter.fit(features_train, outcome_train)
line_fitter.score(features_test,outcome_test)

# Score Model on TestData

line_fitter.score(features_test,outcome_test)

# Plot Real Outcomes vs Prediction

prediction = line_fitter.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)
plt.show()
plt.clf()

# Create a few linear regression models that use two features to predict yearly earnings

#6 Build Model  Are BreakPointsOpportunities a Predictor for Wins?

feature3 = df[['BreakPointsOpportunities',"FirstServeReturnPointsWon"]]
outcome3 = df[['Winnings']]


# Train and Fit Model

line_fitter = LinearRegression()

features_train, features_test, outcome_train, outcome_test = train_test_split(feature3 , outcome3, train_size = 0.8)

line_fitter.fit(features_train, outcome_train)
line_fitter.score(features_test,outcome_test)

# Score Model on TestData

line_fitter.score(features_test,outcome_test)

# Plot Real Outcomes vs Prediction

prediction = line_fitter.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)
plt.show()
plt.clf()






















import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = pd.read_csv('train.csv')
x = data.iloc[:,0:20]
y = data.iloc[:,-1]

bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(x.columns)
scores = pd.concat([dfcolumns,dfscores], axis=1)
scores.columns = ['specs', 'score']
print(scores.nlargest(5, 'score'))
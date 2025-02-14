import numpy as np
import random
import math
import matplotlib.pyplot as plt 
# import xgboost and sklearn stuff:
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay


# functions to generate random data:
from generate_example_data_1D import *

# random seed:
seed = 1234

# Generate 10k points in [-10,10]: 
N = 100000
xmin = -10
xmax = 10
# signal
muS = 1.0
sigmaS = 1.0
idS = 1 # identifier for signal
Sweight = 10.0 # weight for signal
S, LS, wS = GenerateGaussianData(muS,sigmaS,xmin,xmax,N, idS, Sweight)
# background
muB = -2.0
sigmaB = 1.0
idB = 0
# identifier for background
Bweight = 100.0 # weight for background
B, LB, wB = GenerateGaussianData(muB,sigmaB,xmin,xmax,N, idB, Bweight)

# concatenate lists:
X = np.array(S + B)
L = np.array(LS + LB)
W = np.array(wS + wB)

#print(X)

# create testing and training samples:
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, L, W, test_size=0.5,random_state=seed)

# train XGBoost model:
model = xgb.XGBClassifier()
model.fit(X_train, y_train,sample_weight=w_train)
#print(model)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
# in this case:
# (0,0): background-as-background -> True negative
# (0,1): background-as-signal (mis-id) -> False positive
# (1,1): signal-as-signal -> True positive
# (1,0): signal-as-background (mis-id) -> False negative
confmatrix = confusion_matrix(y_test, predictions)

# signal efficiency:
eff_S = confmatrix[1][0]/(confmatrix[1][0] + confmatrix[1][1])
eff_B = confmatrix[0][0]/(confmatrix[0][0] + confmatrix[0][1])

print('Signal efficiency=', eff_S)
print('Background Efficiency=', 1-eff_B)

# calculate "significance"
print('Initial significance=', Sweight/np.sqrt(Bweight))
print('Final significance=', Sweight*eff_S/np.sqrt(Bweight*(1-eff_B)))


# ROC curve:
y_score = model.fit(X_train, y_train,sample_weight=w_train).predict_proba(X_test)
#fig, ax = plt.subplots() # create the elements required for matplotlib. This creates a figure containing a single axes.
display = RocCurveDisplay.from_predictions(
    y_test,
    y_score[:,1],
    name=f"background",
    color="darkorange",
    plot_chance_level=True,
    despine=True,
)
_ = display.ax_.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="ROC curve",
)
#plt.show() # show the plot here
plt.savefig('xgboost_example_roc.pdf')

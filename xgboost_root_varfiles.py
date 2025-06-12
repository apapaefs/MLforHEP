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


# functions to load root varfiles from HwSim
from read_root_varfiles import *

# random seed:
seed = 1234

# Branching ratios:
BR_z_ellell = 3.3632E-2 #  Z -> lepton lepton (one flavour)
BR_w_ellnu = 10.86E-2 # W -> lepton+neutrino (one flavour)
BR_z_vv = 0.2 # Z -> neutrino neutrino (all flavours)
BR_z_qq = 0.116 + 0.156 + 0.1203 + 0.1512 # Z -> qq
BR_h_bb = 0.5824
BR_h_gamgam = 0.00229

# factors to apply to signal and background (K-factors and BRs)
btagging = 0.85
sig_factors = 2.0 * BR_h_bb**3 * btagging**6
bkg_factors = 2.0 * btagging**6 # BRs already applied. The k-factor is uniform

# Luminosity (inv fb):
Lumi = 20000
# initial total weight of events (before the analysis that created the _var.root files):
initial_S = 10000
initial_B = 864960

# load signal and backgrounds
# NOTE THAT: the weights will also be multiplied by the total cross section for the process! 

# load signal:
idS=1 # id number for signal
xsS=2.8979 # signal cross section
S, LS, wS = read_ROOT_varfile('./rootdata/HW-7_SM_var.smear.root', idS, xsS)
Sweight = Lumi * np.sum(wS)/initial_S * sig_factors # calculate total expected number of events

# load background(s):
idB=0 # id number for background
xsB=28.328254252903694E3 # background cross section (fb)
B, LB, wB =  read_ROOT_varfile('./rootdata/HW-all_events_6b_100_var.smear.root', idB, xsB)
Bweight = Lumi * np.sum(wB)/initial_B * bkg_factors # calculate total expected number of events

print('Signal pre-efficiency=', np.sum(wS)/initial_S/xsS)
print('Background pre-efficiency=', np.sum(wB)/initial_B/xsB)

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
# in this case signal = 1, background = 0
# in this case:
# (0,0): background-as-background -> True negative
# (0,1): background-as-signal (mis-id) -> False positive
# (1,1): signal-as-signal -> True positive
# (1,0): signal-as-background (mis-id) -> False negative
confmatrix = confusion_matrix(y_test, predictions)
print('confusion matrix:')
print(confmatrix)
# signal efficiency = number of signal events identified as signal divided by total number of signal events
eff_S = confmatrix[1][1]/(confmatrix[1][0] + confmatrix[1][1])
# background efficiency = number of background events identified as signal divided by total number of background events
eff_B = confmatrix[0][1]/(confmatrix[0][0] + confmatrix[0][1])

print('Luminosity=', Lumi)

# initial cross sections into final state:
print('Initial signal cross section=', sig_factors*np.sum(wS)/initial_S)
print('Initial background cross section=', bkg_factors*np.sum(wB)/initial_B)
print('-')
# calculate "significance"
print('Initial significance=', Sweight/np.sqrt(Bweight))
print('-')
# print analysis efficiencies
print('Signal efficiency=', eff_S)
print('Background Efficiency=', eff_B)
print('Final signal cross section=', sig_factors*np.sum(wS)/initial_S*eff_S)
print('Final background cross section=', bkg_factors*np.sum(wB)/initial_B*eff_B)
print('Final significance=', Sweight*eff_S/np.sqrt(Bweight*eff_B))
print('-')
# calculate 95% C.L. limit on expected number of events: 
S2sigma = np.sqrt(Bweight*eff_B) * 2
print('95% C.L. limit on number of signal events=', S2sigma)
print('95% C.L. limit on signal cross section in given final state=', S2sigma/Lumi, 'fb')


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

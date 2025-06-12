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

# Energy for the analysis (TeV):
Energy = 100 
# Luminosity (inv fb):
Lumi = 20000


# load signal and backgrounds
# NOTE THAT: the weights will also be multiplied by the total cross section for the process! 

# initial total weight of events (before the analysis that created the _var.root files):
initial_S = 10000
# load signal:
idS=0 # id number for signal
xsS=2.8979 # signal cross section
S, LS, wS = read_ROOT_varfile('./rootdata/HW-7_SM_var.smear.root', idS, xsS)
Sweight = Lumi * np.sum(wS)/initial_S * sig_factors # calculate total expected number of events
print('Signal pre-efficiency=', np.sum(wS)/initial_S/xsS)

# initial values for arrays used in training: 
X = S
L = LS
W = wS

# load background(s):
Backgrounds = []
Backgrounds.append('all_events_6b')
Backgrounds.append('pp_zbbbb')
#Backgrounds.append('pp_zzbb')
#Backgrounds.append('pp_hzbb')
#Backgrounds.append('pp_hhbb')
#Backgrounds.append('gg_hzz')
#Backgrounds.append('gg_zzz')
#Backgrounds.append('pp_hhz')
#Backgrounds.append('gg_hhz')
Backgrounds_xsec = {}
Backgrounds_xsec[(100, 'all_events_6b')] = 28.328254252903694E3 # cross section for 6b background in fb (100 TeV)
Backgrounds_xsec[(100, 'pp_zbbbb')] = 642.6825598 # cross section for zbbbb background in fb (100 TeV)
#Backgrounds_xsec[(100, 'pp_zzbb')] = 24.50364084 # cross section for pp_zzbb background in fb (100 TeV)
#Backgrounds_xsec[(100, 'pp_hzbb')] = 4.934954467 # cross section for pp_hzbb background in fb (100 TeV)
#Backgrounds_xsec[(100, 'pp_hhz')] = 0.300644936 # cross section for pp_hhbb background in fb (100 TeV)
#Backgrounds_xsec[(100, 'pp_hhbb')] = 0.047885626 # cross section for pp_hhbb background in fb (100 TeV)
#Backgrounds_xsec[(100, 'gg_hzz')] = 4.140101372 # cross section for gg_hzz background in fb (100 TeV)
#Backgrounds_xsec[(100, 'gg_zzz')] = 3.3162  # cross section for gg_zzz background in fb (100 TeV)
#Backgrounds_xsec[(100, 'gg_hhz')] = 1.32481746  # cross section for gg_hhz background in fb (100 TeV)


# locaiton of the _var root files for the backgrounds:
Background_files = {}
Background_files[(100, 'all_events_6b')] = './rootdata/HW-all_events_6b_100_var.smear.root'
Background_files[(100, 'pp_zbbbb')] = './rootdata/HW-pp_zbbbb_100_var.smear.root'

# initial weight of Monte Carlo events (at the start of the analysis that generated the var root files):
initial_B = {}
initial_B['all_events_6b'] = 864960
initial_B['pp_zbbbb'] = 98721

# initial actual (i.e. at luminosity) number of events for backgrounds
initial_NB = {}

# background ids:
idB = {}
idB['all_events_6b'] = 1
idB['pp_zbbbb'] = 2

Bweight = 0 
for bkg in Backgrounds:
    xsB=Backgrounds_xsec[(Energy, bkg)] # background cross sections (fb)
    B, LB, wB =  read_ROOT_varfile(Background_files[(Energy, bkg)], idB[bkg], Backgrounds_xsec[(Energy, bkg)])
    initial_NB[bkg] =  Lumi * np.sum(wB)/initial_B[bkg] * bkg_factors # calculate total expected number of events in each background
    Bweight += initial_NB[bkg] # incremenet to total expected number of events
    print('Background pre-efficiency', bkg, np.sum(wB)/initial_B[bkg]/Backgrounds_xsec[(Energy, bkg)])
    # concatenate lists:
    X = X + B
    L = L + LB
    W = W + wB

# convert to numpy arrays: 
X = np.array(X)
L = np.array(L)
W = np.array(W)

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
# in this case signal = 0, backgrounds = i = 1, 2,...
# (0,0): signal-as-signal -> True positive
# (0,i): background-as-signal (mis-id) -> False positive
confmatrix = confusion_matrix(y_test, predictions)
print('confusion matrix:')
print(confmatrix)
# signal efficiency:
total_S = 0
for j in range(len(Backgrounds)+1):
    total_S += confmatrix[0][j]
eff_S = confmatrix[0][0]/total_S # signal identified as signal divided by total number of signal events
# background effiencies:
eff_B = {}
for bkg in Backgrounds:
    total_B = 0
    for j in range(len(Backgrounds)+1):
        total_B += confmatrix[idB[bkg]][j]
    eff_B[bkg] = confmatrix[idB[bkg]][0]/total_B


print('Luminosity=', Lumi)

# initial cross sections into final state:
print('Initial signal cross section=', sig_factors*np.sum(wS)/initial_S)
print('Initial background cross section=', Bweight/Lumi)
print('-')
# calculate "significance"
print('Initial significance=', Sweight/np.sqrt(Bweight))
print('-')
# print analysis efficiencies
print('Signal efficiency=', eff_S)
print('Background Efficiencies=', eff_B)
print('-')
print('Final signal cross section=', sig_factors*np.sum(wS)/initial_S*eff_S)
# calculate the number of events for the background after the analysis:
final_NB = {}
final_NB_total = 0
for bkg in Backgrounds:
    final_NB[bkg] = initial_NB[bkg] * eff_B[bkg]
    #print('\tNumber of events in', bkg,final_NB[bkg], 'after analysis')
    final_NB_total += final_NB[bkg]
print('Final background cross section=', final_NB_total/Lumi)
print('Final significance=', Sweight*eff_S/np.sqrt(final_NB_total))
print('-')
# calculate 95% C.L. limit on expected number of events: 
S2sigma = np.sqrt(final_NB_total) * 2
print('95% C.L. limit on number of signal events=', S2sigma)
print('95% C.L. limit on signal cross section in given final state=', S2sigma/Lumi, 'fb')


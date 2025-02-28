import ROOT
from tqdm import tqdm
import numpy as np

# reads ROOT var files from HwSim
# maxevents is the maximum number of events to use
# identifier is the identifier for the event sample
# xsec is the cross section
def read_ROOT_varfile(inFileName, identifier, xsec, maxevents=1E6):
    # indentifier list:
    identifiers = []
    # weights list:
    weights = []
    # read the ROOT file and get the data
    inFile = ROOT.TFile.Open(inFileName ,"READ")
    treein = inFile.Get("Data2")
    print(inFileName, "contains:", treein.GetEntries(), "events")

    # if the maxevents is greater than the number of entries, reset to the max
    if maxevents > treein.GetEntries():
        maxevents = treein.GetEntries()
    print('Getting', maxevents, 'events from', inFileName)
    # loop and push the entries into a list
    events = []
    for entryNum in tqdm(range(0,maxevents)):
        variables = []
        # get the entry from the tree
        treein.GetEntry(entryNum)
        # get the variables array:
        variablesin = getattr(treein,"variables")
        for i, var in enumerate(variablesin):
            if i != 0: # skip the weight
                variables.append(var)
            else: 
                weights.append(var*xsec) # append the weight in separate list: MULTIPLY BY GIVEN XSEC
        events.append(variables)
        identifiers.append(identifier)
    return events, identifiers, weights

# TEST: 

#signal_events, signal_id, signal_weights = read_ROOT_varfile('./rootdata/HW-7_SM_var.smear.root', 1, 1.0)
#for i, var in enumerate(signal_events):
#    print(i, [v for v in var])
#background_events, background_id, background_weights = read_ROOT_varfile('./rootdata/HW-all_events_6b_100_var.smear.root', 2, 1.0)
#for i, var in enumerate(background_events):
#    print(i, [v for v in var])

#import libraries
import matplotlib.pyplot as plt
import awkward as ak
import pandas as pd
import numpy as np
import uproot
from hist import Hist, axis
import os
#Setup

#Constants
current_dir = os.getcwd()
tree_name= "Delphes"  #all of them are named delphes
luminescence = 300 #fb

#Signal

signal_folder = "signal_folder"
signal_dir = os.path.join(current_dir, signal_folder)# Path to Signals_DMSimpl_Spin0 folder
signal_files = {     #times 1000 to convert from pb to fb
            "DMsimpl_spin0_Y0gg1_DeltaEta2.root": 1.64749 * 1000,
            
            }


#Background
background_folders = {#"BKG_Wjets_WToLNu" : 47744.85 * 1000,
                       "Wlnu_jets": 61082.51494 * 1000,
                       "Znunu_jets": 12995.62137 * 1000}  



#will store data to be used in final "cut chart"
num_cuts = 0
cuts_strings = []
Signal_numevents = []
Background_numevents = []
significances = []

## Bins

binning = {
    "PT": {"range": (20, 1000), "bins": 14},  # (300 - 20) / 20 = 14 bins
    "Eta*Eta": {"range": (-25, 25), "bins": 500},  # (25 - (-25)) / 0.1 = 500 bins
    "Delta_Eta": {"range": (0, 10), "bins": 100},  # (10 - 0) / 0.1 = 100 bins
    "Transverse": {"range": (0, 500), "bins": 100},  # (2000 - 500) / 20 = 75 bins
    "Eta": {"range": (-5, 5), "bins": 100},  # (5 - (-5)) / 0.1 = 100 bins
    "Phi": {"range": (-(np.pi), np.pi), "bins": 63},  
    "Invariant": {"range": (50, 3000), "bins": 148}  # (3000 - 50) / 20 = 147.5, rounded to 148 bins
}

print(signal_dir)
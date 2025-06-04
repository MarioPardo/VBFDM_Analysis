'''
#       This file stores functions related to ROOT file IO:
             * getting weights from root file
             * reading data from root file
#
#
'''

import calculation_functions
import matplotlib.pyplot as plt
import awkward as ak
import pandas as pd
import numpy as np
import uproot
import os
from hist import Hist, axis

#Our library files
import calculation_functions

## Variables
#  Variables for current analysis context
from setup_variables import luminescence,tree_name, binning,  signal_files, signal_dir, current_dir, background_dir, background_folders



#############################  FUNCTIONS  ##########################################3


# # # # # # # Gathering Data Functions # # # # # 


#Get tree from root file
def openTree(tree_filepath):

    ''' This function opens a ROOT file and extracts the relevant branches from the Delphes tree.
    It returns a pandas DataFrame containing the selected branches.
    The branches we want to keep are:
    - MissingET.MET                
    - MissingET.Phi
    - Jet.PT
    - Jet.Phi
    - Jet.Eta
    - Jet.Mass
    ''' 
    file = uproot.open(tree_filepath)
    tree = file[tree_name]

    #What we want to keep
    branches_wanted = [
        "MissingET.MET",
        "MissingET.Phi",
        "Jet.PT",
        "Jet.Phi",
        "Jet.Eta",
        "Jet.Mass"
    ]
    df = tree.arrays(branches_wanted,library="pd")

    return df



# Get Weights from every file in given folder"
def get_signal_weights(directory=signal_dir, file_effectivearea_dir=signal_files,luminescence=luminescence):
    '''
    Get weights from every root file at directory
    For each file, the weight is calculated using luminescence and the corresponding effective area defined
    in the dictionary 'file_effectivearea_dir
    '''

    signal_weight_list = []

    # Iterate over each signal file
    for signal_file in sorted(os.listdir(directory)):  #sorted so that it's the same order each time
        
        if signal_file.endswith(".root"):
            file_path = os.path.join(directory, signal_file)
            i=0
            # print("step number",i)
            i+=1
            
            # Check if the root file is in the signal_files dictionary
            if signal_file in signal_files:
                print(file_path)
                signal_df = openTree(file_path)
                signal_temp = signal_df["MissingET.MET"].values
                #print(signal_df.head(3))

                numSigEvents = len(signal_temp)
                tempSigWeight = calculation_functions.calculateWeight(num_events=numSigEvents,effective_area= file_effectivearea_dir[signal_file],lumi=luminescence)
                signal_weight_list.append(tempSigWeight)
    

    return signal_weight_list
    
# Get Weights from every file in given folders
def get_background_weights(directory=background_dir, folder_effectivearea_dir=background_folders,luminescence=luminescence):
    '''
    Get weights from every root file, in every background folder in "folder_effective_area", inside of directory
    For each file, the weight is calculated using luminescence and the corresponding effective area defined
    in the dictionary 'file_effectivearea_dir
    '''

    background_weight_list = []

    for folder_name in sorted(os.listdir(directory)):
        folder_path = os.path.join(directory, folder_name)
        
        # Check if the current item is a directory and if its name is in the background_folders dictionary
        if os.path.isdir(folder_path) and (folder_name in folder_effectivearea_dir):
            print("Folder name:", folder_name)
            cross_section = folder_effectivearea_dir[folder_name]
            
            for root_file in sorted(os.listdir(folder_path)):
                if root_file.endswith(".root"):
                    print(" filename: ", root_file)
                    file_path = os.path.join(folder_path, root_file)
                    background_df = openTree(file_path) 
                    background_temp = background_df["MissingET.MET"].values #arbitraty branch 

                    numBkgEvents = len(background_temp)
                    tempBkgWeight = calculation_functions.calculateWeight(num_events=numBkgEvents, effective_area=cross_section, lumi=luminescence)

                    background_weight_list.append(tempBkgWeight)
                    
    return background_weight_list


# DEPRECATED
def get_weights(signal_dir,type_signal,signal_files, background_folders):

    '''This function calculates the weights for signal and background events. 
    It takes the directory of the signal files and the type of signal (signal or background) as input.
    It returns a list of weights for the events in the specified type of signal.

    example:
    signal_weight_list = get_weights(signal_dir, type_signal="signal")
    background_weight_list = get_weights(background_dir, type_signal="background")

    output:
    signal_weight_list = [0.0038, 0.0086, 0.0018, 0.00026]
    background_weight_list = [47744.85, 8818.65]
    
    
    
    '''

    if type_signal=="signal":
        signal_weight_list = []

        # Iterate over each signal file
        for signal_file in sorted(os.listdir(signal_dir)):  #sorted so that it's the same order each time
            
            if signal_file.endswith(".root"):
                file_path = os.path.join(signal_dir, signal_file)
                i=0
               # print("step number",i)
                i+=1
                
                # Check if the root file is in the signal_files dictionary
                if signal_file in signal_files:
                    print(file_path)
                    signal_df = openTree(file_path)
                    signal_temp = signal_df["MissingET.MET"].values
                    #print(signal_df.head(3))

                    numSigEvents = len(signal_temp)
                    tempSigWeight = calculation_functions.calculateWeight(num_events=numSigEvents,effective_area= signal_files[signal_file],lumi=luminescence)
                    signal_weight_list.append(tempSigWeight)
        

        return signal_weight_list
            

    elif type_signal=="background":
        background_weight_list = []
        print("signal dir: ", signal_dir)
        for folder_name in sorted(os.listdir(signal_dir)):
            print("curr folder name:" , folder_name)
            folder_path = os.path.join(signal_dir, folder_name)
        
            # Check if the current item is a directory and if its name is in the background_folders dictionary
            
            if os.path.isdir(folder_path) and (folder_name in background_folders):
                cross_section = background_folders[folder_name]
                
                for root_file in os.listdir(folder_path):
                    if root_file.endswith(".root"):
                        file_path = os.path.join(folder_path, root_file)
                        background_df = openTree(file_path) 
                        background_temp = background_df["MissingET.MET"].values #arbitraty branch 

                        numBkgEvents = len(background_temp)
                        tempBkgWeight = calculation_functions.calculateWeight(num_events=numBkgEvents, effective_area=cross_section, lumi=luminescence)

                        background_weight_list.append(tempBkgWeight)
                        
        return background_weight_list


def getJetsData(dataname, masklist,signal_directory=signal_dir,background_directory=background_dir,background_folders=background_folders): 
    #used when we are working with J0,J1 stuff (which is often)

    #Signal Processing
    signal_j0_list = []
    signal_j1_list = []

    for signal_file in sorted(os.listdir(signal_directory)):
        if signal_file.endswith(".root"):
            file_path = os.path.join(signal_directory, signal_file)
            
            # Check if the root file is in the signal_files dictionary
            if signal_file in signal_files:
                signal_df = openTree(file_path)

                if(masklist is not None):
                    signal_df = applyMultipleCuts(signal_df,masklist)


                 # Extract and filter jet data
                signal_jets = signal_df["Jet." + dataname].values
                signal_filtered_jets = [entry for entry in signal_jets if len(entry) >= 2]  # At least two entries (j0,j1)

                # Extract J0 and J1
                signal_j0 = np.array([entry[0] for entry in signal_filtered_jets])
                signal_j1 = np.array([entry[1] for entry in signal_filtered_jets])

                signal_j0_list.append(signal_j0)
                signal_j1_list.append(signal_j1)

    

    # Background processing
    background_j0_list = []
    background_j1_list = []


    for folder_name in sorted(os.listdir(background_directory)):
        folder_path = os.path.join(background_directory, folder_name)

        # Check if the current item is a directory and if its name is in the background_folders dictionary
        if os.path.isdir(folder_path) and folder_name in background_folders:

            for root_file in sorted(os.listdir(folder_path)):
                if root_file.endswith(".root"):
                    file_path = os.path.join(folder_path, root_file)
                    background_df = openTree(file_path)  

                    if(masklist is not None):
                        background_df = applyMultipleCuts(background_df, masklist)

                    # Extract and filter jet data
                    background_jets = background_df["Jet." + dataname].values
                    background_filtered_jets = [entry for entry in background_jets if len(entry) >= 2]  # At least two entries

                    # Extract J0 and J1
                    background_j0 = np.array([entry[0] for entry in background_filtered_jets])
                    background_j1 = np.array([entry[1] for entry in background_filtered_jets])

                  
                    background_j0_list.append(background_j0)
                    background_j1_list.append(background_j1)
                    

    return signal_j0_list, signal_j1_list, background_j0_list, background_j1_list
        

# # # # # # # End of Gathering Data Functions # # # # # #


#TODO move functions below to more appropriate files


def applySingleCut(df, mask_function):
    mask = mask_function(df)
    return df[mask]

def applyMultipleCuts(df, mask_functions):
    
    for mask in mask_functions:
        df = applySingleCut(df, mask)

    return df



def GetSignificances(sighist,bkghist):
    '''Calculate and return an array of significances from a given
        signal data histogram, and background data histogram.
    '''
    signal_counts = sighist.values()
    bkg_counts = bkghist.values()

    significance = []
    for n_s, n_b in zip(signal_counts, bkg_counts):
        if n_b + n_s > 0:
            S = calculation_functions.calculateSignificance(numSig=n_s, numBkg=n_b)
            significance.append(S)
        else:
            significance.append(0)  # If both are zero, significance is zero.

    return np.array(significance)

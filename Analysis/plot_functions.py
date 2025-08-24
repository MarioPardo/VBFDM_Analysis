'''
#       This file stores functions for making different plots
          
#
#
'''


import matplotlib.pyplot as plt
import awkward as ak
import pandas as pd
import numpy as np
import uproot
from hist import Hist, axis
import os
import calculation_functions
import files_functions 



# Variables
# Variables for current analysis context 
from setup_variables import binning,  current_dir , signal_dir, signal_files ,background_dir, background_folders

# Plot the "dataname" graph for J0, J1
def PlotJets(binname,dataname,masklist,signal_weight_list,background_weight_list,savefilename):
    '''Plot graph for Jets J0,J1 for a given dataname, susing signal weights and background weights given

        Extraction of data is handled by getJetsData 

        For example, PlotJets(dataname = "PT") will plot the "JT" plot for J0, J1

    '''

    #TODO remove binname as its the same as dataname


    j0_hist_background = Hist(
            axis.Regular(binning[binname]["bins"], *binning[binname]["range"], name="thedata", label=dataname+"J0")
        )
    j0_hist_signal = Hist(
        axis.Regular(binning[binname]["bins"], *binning[binname]["range"], name="thedata", label=dataname+"J0")
    )

    #J1
    j1_hist_background = Hist(
        axis.Regular(binning[binname]["bins"], *binning[binname]["range"], name="thedata", label=dataname+"J1")
    )
    j1_hist_signal = Hist(
        axis.Regular(binning[binname]["bins"], *binning[binname]["range"], name="thedata", label=dataname+"J1")
    )

    j0siglist, j1siglist, bkgWlist, bkgZlist,  = files_functions.getJetsData(dataname,masklist)

    # Fill Signal
    for i in range(len(j0siglist)):
        j0_hist_signal.fill(thedata=j0siglist[i], weight=signal_weight_list[i])
        j1_hist_signal.fill(thedata=j1siglist[i], weight=signal_weight_list[i])

    # Fill Background
    bkgW_weights = background_weight_list[0]
    bkgZ_weights = background_weight_list[1]

    for i in range(len(bkgW_weights)): 
        j0_hist_background.fill(thedata=bkgWlist[0][i], weight=bkgW_weights[i])
        j1_hist_background.fill(thedata=bkgWlist[1][i], weight=bkgW_weights[i])

    #J1
    for i in range(len(bkgZ_weights)): 
        j0_hist_background.fill(thedata=bkgZlist[0][i], weight=bkgZ_weights[i])
        j1_hist_background.fill(thedata=bkgZlist[1][i], weight=bkgZ_weights[i])

    
    
    # Create a figure and a set of subplots (2 columns, 1 row)More actions
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot J0
    axs[0].stairs(
        j0_hist_background.values(),
        j0_hist_background.axes[0].edges,
        color='blue',
        label='Background',
        linewidth=2
    )
    axs[0].stairs(
        j0_hist_signal.values(),
        j0_hist_signal.axes[0].edges,
        color='red',
        label='Signal',
        linewidth=3
    )
    axs[0].set_xlabel(dataname+'(j0)')
    axs[0].set_ylabel('Counts')
    axs[0].set_yscale('log')
    axs[0].set_title(dataname+'(j0) Distributions')
    axs[0].legend()
    axs[0].grid(True)

    # Plot J1
    axs[1].stairs(
        j1_hist_background.values(),
        j1_hist_background.axes[0].edges,
        color='blue',
        label='Background',
        linewidth=2
    )
    axs[1].stairs(
        j1_hist_signal.values(),
        j1_hist_signal.axes[0].edges,
        color='red',
        label='Signal',
        linewidth=3
    )
    axs[1].set_xlabel(dataname+'(j1)')
    axs[1].set_ylabel('Counts')
    axs[1].set_yscale('log')
    axs[1].set_title(dataname+'(j1) Distributions')
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout
    plt.tight_layout()
    
    if masklist==None:
        number_of_cuts=0
    else:
        number_of_cuts=str(len(masklist))
    
    # Show the plot
    if savefilename is not None:
        plt.savefig(savefilename+".png")


    #signal and background events to for the 'cut chart'
    numSigEvents = j0_hist_signal.sum() + j1_hist_signal.sum()
    numBkgEvents = j0_hist_background.sum() + j1_hist_background.sum()

    return numSigEvents, numBkgEvents, [j0_hist_signal, j0_hist_background], [j1_hist_signal, j1_hist_background]

#TODO adapt to new handling of background (W,Z)
# Plots a certain graph for a certain jet, J0 or J1
def PlotSingleJet(binname, dataname,signal_weight_list,background_weight_list, masklist, kind,savefilename): 
    '''
    Plots a certain graph for J0, J1, depending on "kind", being either "j1" or "j0"

    '''


    # Create histograms for J0
    j0_hist_background = Hist(
        axis.Regular(binning[binname]["bins"], *binning[binname]["range"], name="thedata", label=dataname+"J0")
    )
    j0_hist_signal = Hist(
        axis.Regular(binning[binname]["bins"], *binning[binname]["range"], name="thedata", label=dataname+"J0")
    )

    # Create histograms for J1
    j1_hist_background = Hist(
        axis.Regular(binning[binname]["bins"], *binning[binname]["range"], name="thedata", label=dataname+"J1")
    )
    j1_hist_signal = Hist(
        axis.Regular(binning[binname]["bins"], *binning[binname]["range"], name="thedata", label=dataname+"J1")
    )

    # Retrieve data
    j0siglist, j1siglist, bkgj0list, bkgj1list = files_functions.getJetsData(dataname, masklist)

    # Fill histograms
    for i in range(len(j0siglist)):
        j0_hist_signal.fill(thedata=j0siglist[i], weight=signal_weight_list[i])
        j1_hist_signal.fill(thedata=j1siglist[i], weight=signal_weight_list[i])

    for i in range(len(background_weight_list)):
        j0_hist_background.fill(thedata=bkgj0list[i], weight=background_weight_list[i])
        j1_hist_background.fill(thedata=bkgj1list[i], weight=background_weight_list[i])

    # Create a figure and a set of subplots based on the parameter
    if kind=="j1":
        # Plot J1 only
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))

        axs.stairs(
            j1_hist_background.values(),
            j1_hist_background.axes[0].edges,
            color='blue',
            label='Background',
            linewidth=2
        )
        axs.stairs(
            j1_hist_signal.values(),
            j1_hist_signal.axes[0].edges,
            color='red',
            label='Signal',
            linewidth=3
        )
        axs.set_xlabel(dataname+'(j1)')
        axs.set_ylabel('Counts')
        axs.set_yscale('log')
        axs.set_title(dataname+'(j1) Distributions')
        axs.legend()
        axs.grid(True)
    elif kind=="j0":
        # Plot J0 only
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))

        axs.stairs(
            j0_hist_background.values(),
            j0_hist_background.axes[0].edges,
            color='blue',
            label='Background',
            linewidth=2
        )
        axs.stairs(
            j0_hist_signal.values(),
            j0_hist_signal.axes[0].edges,
            color='red',
            label='Signal',
            linewidth=3
        )
        axs.set_xlabel(dataname+'(j0)')
        axs.set_ylabel('Counts')
        axs.set_yscale('log')
        axs.set_title(dataname+'(j0) Distributions')
        axs.legend()
        axs.grid(True)


    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.savefig(savefilename+".png")


    if kind=="j1":
        return j1_hist_signal, j1_hist_background
    else:
        return j0_hist_signal, j0_hist_background


# Plot Missing Energy
def PlotMET(masklist, signal_weight_list,background_weight_list,savefilename,signal_directory=signal_dir,signal_files=signal_files, background_directory = background_dir, background_folders=background_folders):
    
    '''Plot graph for Jets J0,J1 for a given dataname, susing signal weights and background weights given

        Extraction of ddata is handled by this method

        For example, PlotJets(dataname = "PT") will plot the "JT" plot for J0, J1

    '''

    met_hist_background = Hist(
        axis.Regular(binning["Transverse"]["bins"], *binning["Transverse"]["range"], name="MET", label="MET")
    )
    z_background_hist=Hist(
        axis.Regular(binning["Transverse"]["bins"], *binning["Transverse"]["range"], name="MET", label="MET")
    )
    w_background_hist=Hist(
        axis.Regular(binning["Transverse"]["bins"], *binning["Transverse"]["range"], name="MET", label="MET")
    )

    met_hist_signal = Hist(
        axis.Regular(binning["Transverse"]["bins"], *binning["Transverse"]["range"], name="MET", label="MET")
    )

    #Signal Data
    signalWeightIndex = 0 #which weight we use 
    for signal_file in sorted(os.listdir(signal_directory)):
        if signal_file.endswith(".root"):
            file_path = os.path.join(signal_directory, signal_file)
            
            # Check if the root file is in the signal_files dictionary
            if signal_file in signal_files:
                signal_df = files_functions.openTree(file_path)
                if signal_df is None:
                    print("Ignoring root file: " + file_path)
                    continue
    
                if(masklist is not None):
                    signal_df = files_functions.applyMultipleCuts(signal_df,masklist)
    
                signal_met = signal_df["MissingET.MET"].values
                signal_met = ak.flatten(signal_met).to_numpy()
    
                met_hist_signal.fill(MET=signal_met,weight = signal_weight_list[signalWeightIndex])
                signalWeightIndex += 1


    #Background Data
    bkgWeightsW  = background_weight_list[0]
    bkgWeightsZ  = background_weight_list[1]
    currBkgWeightList = None


    for folder_name in sorted(os.listdir(background_directory)):
        folder_path = os.path.join(background_directory, folder_name)
            
        # Check if the current item is a directory and if its name is in the background_folders dictionary
        if os.path.isdir(folder_path) and folder_name in background_folders:

            bkgWeightIndex = 0
            for root_file in sorted(os.listdir(folder_path)):
                if root_file.endswith(".root"):
                    file_path = os.path.join(folder_path, root_file)
                    background_df = files_functions.openTree(file_path)
                    if background_df is None:
                        print("Ignoring root file: " + file_path)
                        continue

                    if(masklist is not None):
                        background_df = files_functions.applyMultipleCuts(background_df, masklist)

                    background_met = background_df["MissingET.MET"].values
                    background_met = ak.flatten(background_met).to_numpy()
                   
                    if folder_name.startswith("W"):
                        currBkgWeightList = bkgWeightsW
                        w_background_hist.fill(MET=background_met,weight =currBkgWeightList[bkgWeightIndex])
                    elif folder_name.startswith("Z"):
                        currBkgWeightList = bkgWeightsZ
                        z_background_hist.fill(MET=background_met,weight =currBkgWeightList[bkgWeightIndex])
                    

                    met_hist_background.fill(MET=background_met,weight =currBkgWeightList[bkgWeightIndex])
                    bkgWeightIndex+=1
                
                    

    #Set up histogram
    plt.figure(figsize=(10, 6))

    # Background histogram
    plt.stairs(
        met_hist_background.values(),
        met_hist_background.axes[0].edges,
        color='blue',
        label='Background',
        linewidth=2
    )

    # Signal histogram
    plt.stairs(
        met_hist_signal.values(),
        met_hist_signal.axes[0].edges,
        color='red',
        label='Signal',
        linewidth=3
    )

    # Add labels and legend
    plt.xlabel('MET')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title('MET Distribution')
    plt.legend()
    plt.grid(True)
    
    
    # Save the plot
    plt.savefig(savefilename+".png")

    #signal and background events to for the 'cut chart'
    numSigEvents = met_hist_signal.sum() 
    numBkgEvents = met_hist_background.sum()
    numWBkgEvents = w_background_hist.sum()
    numZBkgEvents = z_background_hist.sum()

    return numSigEvents, numBkgEvents,numWBkgEvents, numZBkgEvents

##TODO These two (PlotMET, PlotPhiMet) can likely be refactored into a single function

# Plot Phi(Met)
def PlotPhiMet(masklist, signal_weight_list,background_weight_list,savefilename,signal_directory=signal_dir,signal_files=signal_files, background_directory = background_dir, background_folders=background_folders):


    met_phi_hist_background = Hist(
        axis.Regular(binning["Phi"]["bins"], *binning["Phi"]["range"], name="MET_Phi", label="MET Phi (rad)")
    )
    z_background_hist=Hist(
        axis.Regular(binning["Phi"]["bins"], *binning["Phi"]["range"], name="MET_Phi", label="MET Phi (rad)")
    )
    w_background_hist=Hist(
        axis.Regular(binning["Phi"]["bins"], *binning["Phi"]["range"], name="MET_Phi", label="MET Phi (rad)")
    )
    met_phi_hist_signal = Hist(
        axis.Regular(binning["Phi"]["bins"], *binning["Phi"]["range"], name="MET_Phi", label="MET Phi (rad)")
    )


    #Signal Data
    sigWeightIndex = 0 #which weight we use 
    for signal_file in sorted(os.listdir(signal_directory)):
        if signal_file.endswith(".root"):
            file_path = os.path.join(signal_directory, signal_file)
            
            # Check if the root file is in the signal_files dictionary
            if signal_file in signal_files:
                signal_df = files_functions.openTree(file_path)
                if signal_df is None:
                    print("Ignoring root file: " + file_path)
                    continue
    
                if(masklist is not None):
                    signal_df = files_functions.applyMultipleCuts(signal_df,masklist)
    
                signal_met_phi = signal_df["MissingET.Phi"].values
                signal_met_phi = ak.flatten(signal_met_phi).to_numpy()
    
                met_phi_hist_signal.fill(MET_Phi=signal_met_phi,weight = signal_weight_list[sigWeightIndex])
                sigWeightIndex += 1

    #Background
    bkgWeightsW  = background_weight_list[0]
    bkgWeightsZ  = background_weight_list[1]
    currBkgWeightList = None


  
    for folder_name in sorted(os.listdir(background_directory)):
        folder_path = os.path.join(background_directory, folder_name)
            
        # Check if the current item is a directory and if its name is in the background_folders dictionary
        if os.path.isdir(folder_path) and folder_name in background_folders:
            
            bkgWeightIndex = 0
            for root_file in sorted(os.listdir(folder_path)):
                if root_file.endswith(".root"):
                    file_path = os.path.join(folder_path, root_file)
                    background_df = files_functions.openTree(file_path)
                    if background_df is None:
                        print("Ignoring root file: " + file_path)
                        continue

                    if(masklist is not None):
                        background_df = files_functions.applyMultipleCuts(background_df, masklist)

                    background_met_phi = background_df["MissingET.Phi"].values
                    background_met_phi = ak.flatten(background_met_phi).to_numpy()
                    if folder_name.startswith("W"):
                        currBkgWeightList = bkgWeightsW
                        w_background_hist.fill(MET_Phi=background_met_phi,weight =currBkgWeightList[bkgWeightIndex])
                    elif folder_name.startswith("Z"):
                        currBkgWeightList = bkgWeightsZ
                        z_background_hist.fill(MET_Phi=background_met_phi,weight=currBkgWeightList[bkgWeightIndex])
                    
                    met_phi_hist_background.fill(MET_Phi=background_met_phi,weight =currBkgWeightList[bkgWeightIndex])
                    bkgWeightIndex += 1
                    

    #Set up histogram
    plt.figure(figsize=(10, 6))

    # Background histogram
    plt.stairs(
        met_phi_hist_background.values(),
        met_phi_hist_background.axes[0].edges,
        color='blue',
        label='Background',
        linewidth=2
    )

    # Signal histogram
    plt.stairs(
        met_phi_hist_signal.values(),
        met_phi_hist_signal.axes[0].edges,
        color='red',
        label='Signal',
        linewidth=3
    )

    # Add labels and legend
    plt.xlabel('MET Phi (rad)')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title('MET Phi Distribution')
    plt.legend()
    plt.grid(True)
        
   
    # save the plot
    plt.savefig(savefilename+".png")


    #signal and background events to for the 'cut chart'
    numSigEvents = met_phi_hist_signal.sum() 
    numBkgEvents = met_phi_hist_background.sum()
    numWBkgEvents = w_background_hist.sum()
    numZBkgEvents = z_background_hist.sum()

    return numSigEvents, numBkgEvents,numWBkgEvents, numZBkgEvents





# Plot EtaJ0 * EtaJ1
def PlotEtaEta(masklist,signal_weight_list,background_weight_list,savefilename):
    '''


    '''

    #TODO refactor to take in same parameters as above functions

    etaeta_hist_background = Hist(
        axis.Regular(binning["Eta*Eta"]["bins"], *binning["Eta*Eta"]["range"], name="Eta*Eta", label="Eta*Eta")
    )

    etaeta_hist_signal = Hist(
        axis.Regular(binning["Eta*Eta"]["bins"], *binning["Eta*Eta"]["range"], name="Eta*Eta", label="Eta*Eta")
    )

    j0siglist, j1siglist, bkgWlist, bkgZlist, = files_functions.getJetsData("Eta",masklist)

    # Fill Signal
    for i in range(len(j0siglist)):
        etaeta_hist_signal.fill(j0siglist[i]*j1siglist[i], weight=signal_weight_list[i])

    # Fill Background
    bkgW_weights = background_weight_list[0]
    bkgZ_weights = background_weight_list[1]

    for i in range(len(bkgW_weights)): 
        etaeta_hist_background.fill(bkgWlist[0][i] * bkgWlist[1][i], weight=bkgW_weights[i])

    for i in range(len(bkgZ_weights)): 
        etaeta_hist_background.fill(bkgZlist[0][i] * bkgZlist[1][i], weight=bkgZ_weights[i])


    #Set up histogram
    plt.figure(figsize=(10, 6))

    # Background histogram
    plt.stairs(
        etaeta_hist_background.values(),
        etaeta_hist_background.axes[0].edges,
        color='blue',
        label='Background',
        linewidth=2
    )

    # Signal histogram
    plt.stairs(
        etaeta_hist_signal.values(),
        etaeta_hist_signal.axes[0].edges,
        color='red',
        label='Signal',
        linewidth=3
    )

    # Add labels and legend
    plt.xlabel('Eta(j0) * Eta(j1)')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title('Eta * Eta Distribution')
    plt.legend()
    plt.grid(True)
        
    if masklist==None:
        number_of_cuts=0
    else:
        number_of_cuts=str(len(masklist))

    # ave the plot
    plt.savefig(savefilename+".png")

    #returns hist itself, for use in the Significance function
    return etaeta_hist_signal, etaeta_hist_background



def PlotDeltaJets(masklist,signal_weight_list,background_weight_list,savefilename):

    deltaeta_hist_background = Hist(
        axis.Regular(binning["Delta_Eta"]["bins"], *binning["Delta_Eta"]["range"], name="DeltaEta", label="DeltaEta")
    )

    deltaeta_hist_signal = Hist(
        axis.Regular(binning["Delta_Eta"]["bins"], *binning["Delta_Eta"]["range"], name="DeltaEta", label="DeltaEta")
    )


    j0siglist, j1siglist, bkgWlist, bkgZlist = files_functions.getJetsData("Eta",masklist)

    # Fill signal
    for i in range(len(j0siglist)):
        deltaeta_hist_signal.fill(np.abs(j0siglist[i]-j1siglist[i]), weight=signal_weight_list[i])

     # Fill Background
    bkgW_weights = background_weight_list[0]
    bkgZ_weights = background_weight_list[1]

    for i in range(len(bkgW_weights)): 
        deltaeta_hist_background.fill(np.abs(bkgWlist[0][i] - bkgWlist[1][i]), weight=bkgW_weights[i])

    for i in range(len(bkgZ_weights)): 
        deltaeta_hist_background.fill(np.abs(bkgZlist[0][i] - bkgZlist[1][i]), weight=bkgZ_weights[i])


    #Set up histogram
    plt.figure(figsize=(10, 6))

    # Background histogram
    plt.stairs(
        deltaeta_hist_background.values(),
        deltaeta_hist_background.axes[0].edges,
        color='blue',
        label='Background',
        linewidth=2
    )

    # Signal histogram
    plt.stairs(
        deltaeta_hist_signal.values(),
        deltaeta_hist_signal.axes[0].edges,
        color='red',
        label='Signal',
        linewidth=3
    )

    # Add labels and legend
    plt.xlabel('Abs(Delta Eta) (j0, j1)')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title('Delta Eta Distribution')
    plt.legend()
    plt.grid(True)


    # Save
    plt.savefig(savefilename+".png")

    #returns hist itself, for use in the Significance function
    return deltaeta_hist_signal, deltaeta_hist_background

def PlotInvariantMass(masklist,signal_weight_list,background_weight_list,savefilename):

    print("Plotting Invariant Mass")

    invariant_hist_background = Hist(
        axis.Regular(binning["Invariant"]["bins"], *binning["Invariant"]["range"], name="Invariant", label="Invariant")
    )

    #for internal number event counts only
    bkg_W_hist = Hist(
        axis.Regular(binning["Invariant"]["bins"], *binning["Invariant"]["range"], name="Invariant", label="Invariant")
    )
    bkg_Z_hist = Hist(
        axis.Regular(binning["Invariant"]["bins"], *binning["Invariant"]["range"], name="Invariant", label="Invariant")
    )
    ####

    invariant_hist_signal = Hist(
        axis.Regular(binning["Invariant"]["bins"], *binning["Invariant"]["range"], name="Invariant", label="Invariant")
    )

    #Data needed: pt0, eta0, phi0, m0, pt1, eta1, phi1, m1
    background_weights_W = background_weight_list[0]
    background_weights_Z = background_weight_list[1]


    ptj0siglist, ptj1siglist, ptbkgWlists, ptbkgZlists = files_functions.getJetsData("PT",masklist)
    etaj0siglist, etaj1siglist, etabkgWlists, etabkgZlists = files_functions.getJetsData("Eta",masklist)
    phij0siglist, phij1siglist, phibkgWlists, phibkgZlists = files_functions.getJetsData("Phi",masklist)
    massj0siglist, massj1siglist, massbkgWlists, massbkgZlists = files_functions.getJetsData("Mass",masklist)

    for i in range(len(ptj0siglist)):
        
        invariant_hist_signal.fill(
            calculation_functions.calc_invariant_mass(pt0=ptj0siglist[i], eta0=etaj0siglist[i], phi0 = phij0siglist[i],m0=massj0siglist[i],
                        pt1=ptj1siglist[i], eta1=etaj1siglist[i], phi1 = phij1siglist[i],m1=massj1siglist[i]),
            weight = signal_weight_list[i])
    
    for i in range(len(background_weights_W)):
        #W Boson
        invariant_hist_background.fill(
            calculation_functions.calc_invariant_mass(
                    pt0=ptbkgWlists[0][i], eta0=etabkgWlists[0][i], phi0 = phibkgWlists[0][i],m0=massbkgWlists[0][i],
                    pt1=ptbkgWlists[1][i], eta1=etabkgWlists[1][i], phi1 = phibkgWlists[1][i],m1=massbkgWlists[1][i]),
                    weight = background_weights_W[i])
        
        bkg_W_hist.fill(
            calculation_functions.calc_invariant_mass(
                    pt0=ptbkgWlists[0][i], eta0=etabkgWlists[0][i], phi0 = phibkgWlists[0][i],m0=massbkgWlists[0][i],
                    pt1=ptbkgWlists[1][i], eta1=etabkgWlists[1][i], phi1 = phibkgWlists[1][i],m1=massbkgWlists[1][i]),
                    weight = background_weights_W[i])

    for i in range(len(background_weights_Z)):
        #Z Boson
        invariant_hist_background.fill(
            calculation_functions.calc_invariant_mass(
                    pt0=ptbkgZlists[0][i], eta0=etabkgZlists[0][i], phi0 = phibkgZlists[0][i],m0=massbkgZlists[0][i],
                    pt1=ptbkgZlists[1][i], eta1=etabkgZlists[1][i], phi1 = phibkgZlists[1][i],m1=massbkgZlists[1][i]),
                    weight = background_weights_Z[i])
        
        bkg_Z_hist.fill(
            calculation_functions.calc_invariant_mass(
                    pt0=ptbkgZlists[0][i], eta0=etabkgZlists[0][i], phi0 = phibkgZlists[0][i],m0=massbkgZlists[0][i],
                    pt1=ptbkgZlists[1][i], eta1=etabkgZlists[1][i], phi1 = phibkgZlists[1][i],m1=massbkgZlists[1][i]),
                    weight = background_weights_Z[i])



    #Set up histogram
    plt.figure(figsize=(10, 6))

    # Background histogram
    plt.stairs(
        invariant_hist_background.values(),
        invariant_hist_background.axes[0].edges,
        color='blue',
        label='Background',
        linewidth=2
    )

    # Signal histogram
    plt.stairs(
        invariant_hist_signal.values(),
        invariant_hist_signal.axes[0].edges,
        color='red',
        label='Signal',
        linewidth=3
    )

    # Add labels and legend
    plt.xlabel('Invariant Mass j0 j1')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title('Invariant Mass Distribution')
    plt.legend()
    plt.grid(True)

        
    if masklist==None:
        number_of_cuts=0
    else:
        number_of_cuts=str(len(masklist))

    # Save the plot
    plt.savefig(savefilename+".png")

    #signal and background events to for the 'cut chart'
    numSigEvents = invariant_hist_signal.sum()
    numBkgEvents = invariant_hist_background.sum()
    numBkgWEvents = bkg_W_hist.sum()
    numBkgZEvents = bkg_Z_hist.sum()

    return numSigEvents, numBkgEvents, numBkgWEvents, numBkgZEvents


#TODO refactor to take in variables and not use if statements
def significance_plot(lims,signal_hist,background_hist,kind):

    if kind=="eta*eta":
        significance = GetSignificances(signal_hist,background_hist)

        # Get bin edges from the histogram
        bin_edges = signal_hist.axes[0].edges

        # Plot significance
        plt.figure(figsize=(10, 6))
        plt.step(bin_edges[:-1], significance, where='mid', label='Significance', color='purple', linewidth=2)

        # Add labels and title
        plt.xlabel('Eta(j0) * Eta(j1)')
        plt.xlim(lims[0], lims[1])  #from visual inspection
        plt.ylabel('Significance')
        plt.yscale('log')
        plt.title('Significance Plot')
        plt.grid(True)
        plt.legend()

        # Show the plot
        plt.savefig("Significance_plot_eta*eta.png")
    
    elif kind=="Delta(eta)":

        significance = files_functions.GetSignificances(signal_hist,background_hist)

        # Get bin edges from the histogram
        bin_edges = signal_hist.axes[0].edges

        # Plot significance
        plt.figure(figsize=(10, 6))
        plt.step(bin_edges[:-1], significance, where='mid', label='Significance', color='purple', linewidth=2)

        # Add labels and title
        plt.xlabel('Delta(Eta(J0,J1))')
        plt.xlim(lims[0], lims[1])  #from visual inspection
        plt.ylabel('Significance')
        plt.yscale('log')
        plt.title('Significance Plot')
        plt.grid(True)
        plt.legend()

        plt.savefig("Significance_plot_Delta(eta).png")

    elif kind=="j0":
        significance = files_functions.GetSignificances(signal_hist, background_hist)

        # Get bin edges from the histogram
        bin_edges = signal_hist.axes[0].edges

        # Plot significance
        plt.figure(figsize=(20, 6))  # <-- Make the figure wider by increasing the first value
        plt.step(bin_edges[:-1], significance, where='mid', label='Significance', color='purple', linewidth=2)

        # Add labels and title
        plt.xlabel('PT(j0)')
        # Set x-axis ticks every 20 units
        plt.xticks(np.arange(bin_edges[0], bin_edges[-1]+1, 20))
        #plt.xlim(0, 5)  #from visual inspection
        plt.ylabel('Significance')
        plt.yscale('log')
        plt.title('Significance Plot')
        plt.grid(True)
        plt.legend()

        # Show the plot
        plt.savefig("Significance_Plot_PT(j0).png")

        
    elif kind=="j1":
        significance = files_functions.GetSignificances(signal_hist, background_hist)

        # Get bin edges from the histogram
        bin_edges = signal_hist.axes[0].edges

        # Plot significance
        plt.figure(figsize=(10, 6))
        plt.step(bin_edges[:-1], significance, where='mid', label='Significance', color='purple', linewidth=2)

        # Add labels and title
        plt.xlabel('PT(j1)')
        #plt.xlim(0, 5)  #from visual inspection
        plt.ylabel('Significance')
        plt.yscale('log')
        plt.title('Significance Plot')
        plt.grid(True)
        plt.legend()

        # Show the plot
        plt.savefig("Significance_Plot_PT(j1).png")

    #Top 5 points
    # Find the indices of the top 5 significance values
    top_indices = np.argsort(significance)[-5:]  # Get indices of the 5 largest values

    # Reverse to get them in descending order
    top_indices = top_indices[::-1]

    # Retrieve the top 5 significance values and their corresponding bin edges
    top_significance_values = significance[top_indices]
    top_bin_edges = bin_edges[top_indices]

    # Print the top 5 significance values and their corresponding bins
    print("Top 5 Significance Values and Their Corresponding Bins:")
    for i, idx in enumerate(top_indices):
        print(f"Rank {i + 1}: Significance = {top_significance_values[i]:.3f}, Bin = {top_bin_edges[i]:.3f}")


    
def Get_Table(number_of_sig_events, number_of_bkg_events, numWBkgEvents, numZBkgEvents, significances, cuts):
    # Data for the table
    data = [
        [
            i,
            cuts[i],
            number_of_sig_events[i],
            number_of_bkg_events[i],
            numWBkgEvents[i],
            numZBkgEvents[i],
            significances[i]
        ]
        for i in range(len(cuts))
    ]

    columns = ["", "Cut", "Signal Events", "Background Events", "W Bkg Events", "Z Bkg Events", "Significance"]

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Hide the axes
    ax.axis("tight")
    ax.axis("off")

    # Add and customize the table
    table = ax.table(
        cellText=data,
        colLabels=columns,
        loc="center",
        cellLoc="center"  # Align text to the center of each cell
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(columns))))  # Auto-adjust column widths

    # Display the plot
    plt.savefig("Final_Table.png")
        


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


def PlotSignificance(signal_hist, bkg_hist, title, xlabel, xlim=None):
    # Calculate significance bin by bin
    significance = GetSignificances(signal_hist, bkg_hist)

    # Get bin edges
    bin_edges = signal_hist.axes[0].edges

    # Plot
    plt.figure(figsize=(10, 6))
    plt.step(bin_edges[:-1], significance, where='mid',
             label='Significance', color='purple', linewidth=2)

    # Labels and title
    plt.xlabel(xlabel)
    if xlim is not None:
        plt.xlim(xlim)   # expects tuple like (-17, 17)
    plt.ylabel('Significance')
    plt.yscale('log')
    plt.title(title)
    plt.grid(True)
    plt.legend()

    plt.show()
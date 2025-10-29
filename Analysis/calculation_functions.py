import matplotlib.pyplot as plt
import awkward as ak
import pandas as pd
import numpy as np
import uproot
from hist import Hist, axis
import os


def calculateWeight(num_events, effective_area, lumi):
    return effective_area * lumi / num_events

def calculateSignificance(numSig,numBkg):
    return numSig/np.sqrt(numSig + numBkg)

def calc_invariant_mass(pt0, eta0, phi0, m0, pt1, eta1, phi1, m1):
    # Energy and momentum components for Jet0
    pz0 = pt0 * np.sinh(eta0)
    e0 = np.sqrt(pt0**2 + pz0**2 + m0**2)
    px0 = pt0 * np.cos(phi0)
    py0 = pt0 * np.sin(phi0)

    # Energy and momentum components for Jet1
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(pt1**2 + pz1**2 + m1**2)
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)

    # Invariant mass calculation
    e_total = e0 + e1
    px_total = px0 + px1
    py_total = py0 + py1
    pz_total = pz0 + pz1

    m2 = e_total**2 - (px_total**2 + py_total**2 + pz_total**2)
    m2 = np.maximum(m2, 0)  # This will replace negative values with 0
    return np.sqrt(m2)

def calc_invariant_mass_lorentz_vectors(v1_pt, v1_eta, v1_phi, v1_mass, v2_pt, v2_eta, v2_phi, v2_mass):
    # Create Lorentz vectors for both particles
    v1 = ak.zip({
        "pt": v1_pt,
        "eta": v1_eta,
        "phi": v1_phi,
        "mass": v1_mass
    }, with_name="LorentzVector")

    v2 = ak.zip({
        "pt": v2_pt,
        "eta": v2_eta,
        "phi": v2_phi,
        "mass": v2_mass
    }, with_name="LorentzVector")

    # Sum the Lorentz vectors
    v_total = v1 + v2

    # Return the invariant mass
    return v_total.mass

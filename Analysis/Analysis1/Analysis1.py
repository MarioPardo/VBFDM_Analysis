




#### CUTS:


#Cut for eta(j0)*eta(j1) >=0
def mask_etaeta_condition(df):
    # Extract Eta for j0 and j1
    eta_j0 = df["Jet.Eta"].apply(lambda x: x[0] if len(x) > 0 else None)
    eta_j1 = df["Jet.Eta"].apply(lambda x: x[1] if len(x) > 1 else None)

    # Create mask where Eta(j0) * Eta(j1) >= 0
    return (eta_j0 * eta_j1) >= 0

## We choose PT(j0)>510
def mask_pt_j0_condition(df):
    # Extract PT for j0
    pt_j0 = df["Jet.PT"].apply(lambda x: x[0] if len(x) > 0 else None)

    # Create mask where PT(j0) > 510
    return pt_j0 > 440



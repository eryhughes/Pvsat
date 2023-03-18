# melt-gas.py

import pandas as pd
import numpy as np
import gmpy2 as gp
import math
import densityx as dx

################
### Contents ###
################

# Solubility constants
# Solid/liquid saturation
# Equilibrium constants
# Fugacity coefficients
# Isotope fractionation factors
# Fugacity, mole fraction, partial pressure
# Speciation 
# Converting gas and melt compositions
# Volume and density

#################################################################################################################################
##################################################### SOLUBILITY CONSTANTS ######################################################
#################################################################################################################################

###################################
### Solubility constant for H2O ###
###################################
def C_H2O(PT,models):
    
    if models.loc["Hspeciation","option"] == "none": ### C_H2O = (xmH2O)^2/fH2O ### (mole fraction)
        model = models.loc["water","option"]
        if model == "AllisonDataComp": # fitted to experimental data compilation from Allison et al. (2022) for H2O < 6 wt%
            C = 4.6114e-6
    
    else: ### C_H2O = xmH2Omol/fH2O ### (mole fraction)
        P = PT['P']
        model = models.loc["water","option"]
        P0 = 1.0 # bar
        R = 83.15 # cm3 etc.
        T0 = 1473.15 # K
        if model == "Dixon95": # Dixon et al. (1995) JPet, 36(6):1607-1631
            DV = 12.
            A = 3.28e-5
            C = A*gp.exp((-DV*(P-P0))/(R*T0))
        elif model == "alkali basalt": # Lesne et al. (2011) 162:133-151 eqn 31 with added RT term otherwise it will not work
            A = 5.71e-5 # XmH2Om0
            DV = 26.9 # VH2Om0 in cm3/mol
            C = A*gp.exp((-DV*(P-P0))/(R*T0)) 
    return C

        
        
#########################################
### Solubility constant for carbonate ###
#########################################
def C_CO3(run,PT,melt_wf,setup,species,models): ### C_CO3 = xmCO2/fCO2 ### mole fraction
    model = models.loc["carbonate","option"]
    
    P = PT['P']
    T_K = PT['T']+273.15
        
    # Calculate cation proportions with no volatiles but correct Fe speciation if available (a la Dixon 1997)
    tot = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"]) + ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"]) + ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"]) + ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"]) + ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"]) + ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"]) + ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"]) + ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"]) + ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"]) + ((species.loc["P2O5","no_cat"]*setup.loc[run,"P2O5"])/species.loc["P2O5","M"])
    Si = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"])/tot
    Ti = ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"])/tot
    Al = ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"])/tot
    #Fe2 = ((species.loc["FeO","no_cat"]*Wm_FeO(Fe3FeT,species))/species.loc["FeO","M"])/tot
    Fe2 = ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"])/tot
    Mg = ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"])/tot
    Ca = ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"])/tot
    Na = ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"])/tot
    K = ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"])/tot

    R = 83.15
    T0 = 1473.15 # K
    PI = -6.5*(Si+Al) + 20.17*(Ca+0.8*K+0.7*Na+0.4*Mg+0.4*Fe2) # Dixon (1997) Am. Min. 82:368-378
    DH = -13.1 # kJ/mol # Lesne et al. (2011) CMP 162:153-168 from basanite of Holloway & Blank (1994)
 
    if model == "Dixon95": # Dixon et al. (1995) JPet 36(6):1607-1631
        DV = 23 # cm3/mol
        P0 = 1.0 # bar
        A = 3.8e-7
        B = (-DV*(P-P0))/(R*T0)
        C = A*gp.exp(B)
    elif model == "Dixon97": # Compositional dependence from Dixon (1997) Am. Min. 82:368-378 as shown by Witham et al. (2012) [assumes PI-SiO2 relationship in caption of figre 2 is 10.19 instead of 10.9 - if 10.9 is assumed you get negative C_CO3]
        DV = 23 # cm3/mol
        P0 = 1.0 # bar
        A = (7.94e-7)*(PI+0.762)
        B = (-DV*(P-P0))/(R*T0)
        C = A*gp.exp(B)
    elif model == "Lesne11": # Lesne et al. (2011) CMP 162:153–168
        DV = 23 # cm3/mol
        P0 = 1.0 # bar
        A = 7.94e-7*((((871*PI)+93.0)/1000.0)+0.762)
        B = (-DV*(P-P0))/(R*T0)
        C = A*gp.exp(B)
    elif model == "VES-9": # Lesne et al. (2011) CMP 162:153-168
        DV = 31.0 # cm3/mol
        P0 = 1000.0 # bar
        A = gp.exp(-14.10)
        B = -((DV/(R*T_K))*(P-P0)) + (DH/R)*((1.0/T0) - (1.0/T_K))
        C = A*gp.exp(B)
    elif model == "ETN-1": # Lesne et al. (2011) CMP 162:153-168
        DV = 23.0 # cm3/mol
        P0 = 1000.0 # bar
        A = gp.exp(-14.55)
        B = -((DV/(R*T_K))*(P-P0)) + (DH/R)*((1.0/T0) - (1.0/T_K))    
        C = A*gp.exp(B)
    elif model == "PST-9": # Lesne et al. (2011) CMP 162:153-168
        DV = 6.0 # cm3/mol
        P0 = 1000.0 # bar
        A = gp.exp(-14.74)
        B = -((DV/(R*T_K))*(P-P0)) + (DH/R)*((1.0/T0) - (1.0/T_K))
        C = A*gp.exp(B)
    elif model == "Sunset Crater": # Sunset Crater from Allison et al. (2022) CMP 177:40
        R_ = 83.144621 # cm3 bar K−1 mol−1
        DV = 16.40 # cm3/mol
        P0 = 1000.0 # bar
        A = gp.exp(-14.67)
        B = -((DV/(R_*T_K))*(P-P0))
        C = A*gp.exp(B)
    elif model == "SFVF": # SFVF from Allison et al. (2022) CMP 177:40
        R_ = 83.144621 # cm3 bar K−1 mol−1
        DV = 15.02 # cm3/mol
        P0 = 1000.0 # bar
        A = gp.exp(-14.87)
        B = -((DV/(R_*T_K))*(P-P0))
        C = A*gp.exp(B)
    elif model == "Erebus": # Erebus from Allison et al. (2022) CMP 177:40
        R_ = 83.144621 # cm3 bar K−1 mol−1
        DV = -14.65 # cm3/mol
        P0 = 1000.0 # bar
        A = gp.exp(-14.65)
        B = -((DV/(R_*T_K))*(P-P0))
        C = A*gp.exp(B)
    elif model == "Vesuvius": # Vesuvius from Allison et al. (2022) CMP 177:40
        R_ = 83.144621 # cm3 bar K−1 mol−1
        DV = 24.42 # cm3/mol
        P0 = 1000.0 # bar
        A = gp.exp(-14.04)
        B = -((DV/(R_*T_K))*(P-P0))
        C = A*gp.exp(B)
    elif model == "Etna": # Etna from Allison et al. (2022) CMP 177:40
        R_ = 83.144621 # cm3 bar K−1 mol−1
        DV = 21.59 # cm3/mol
        P0 = 1000.0 # bar
        A = gp.exp(-14.28)
        B = -((DV/(R_*T_K))*(P-P0))
        C = A*gp.exp(B)
    elif model == "Stromboli": # Stromboli from Allison et al. (2022) CMP 177:40
        R_ = 83.144621 # cm3 bar K−1 mol−1
        DV = 14.93 # cm3/mol
        P0 = 1000.0 # bar
        A = gp.exp(-14.68)
        B = -((DV/(R_*T_K))*(P-P0))
        C = A*gp.exp(B)
    elif model == "Basanite": # Basanite composition from Holloway and Blank (1994), data from Allison et al. (2022) CMP 177:40
        R_ = 83.144621 # cm3 bar K−1 mol−1
        DV = 21.72 # cm3/mol
        P0 = 1000.0 # bar
        A = gp.exp(-14.32)
        B = -((DV/(R_*T_K))*(P-P0))
        C = A*gp.exp(B)
    elif model == "Leucite": # Leucite composition from Thibault and Holloway (1994), data from Allison et al. (2022) CMP 177:40
        R_ = 83.144621 # cm3 bar K−1 mol−1
        DV = 21.53 # cm3/mol
        P0 = 1000.0 # bar
        A = gp.exp(-13.36)
        B = -((DV/(R_*T_K))*(P-P0))
        C = A*gp.exp(B)
    elif model == "AH3 phonotephrite": # AH3 Phonotephrite composition from Vetere et al. (2014), data from Allison et al. (2022) CMP 177:40
        R_ = 83.144621 # cm3 bar K−1 mol−1
        DV = 30.45 # cm3/mol
        P0 = 1000.0 # bar
        A = gp.exp(-13.26)
        B = -((DV/(R_*T_K))*(P-P0))
        C = A*gp.exp(B)
    elif model == "N72 Basalt": # N72 basalt composition from Shishkina et al. (2010), data from Allison et al. (2022) CMP 177:40
        R_ = 83.144621 # cm3 bar K−1 mol−1
        DV = 19.05 # cm3/mol
        P0 = 1000.0 # bar
        A = gp.exp(-14.86)
        B = -((DV/(R_*T_K))*(P-P0))
        C = A*gp.exp(B)
    return C


########################################
### solubility constant for sulphide ###
########################################
def C_S(run,PT,melt_wf,setup,species,models): ### C_S = wmS2-*(fO2/fS2)^0.5 ### (weight ppm)
    model = models.loc["sulphide","option"]
    
    T = PT['T'] + 273.15 # T in K
    
    if model == "ONeill21": # O'Neill (2021) Magma Redox Geochemistry (AGU Monograph Series) 
               
        # Mole fractions in the melt on cationic lattice (all Fe as FeO) no volatiles
        tot = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"]) + ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"]) + ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"]) + ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"]) + ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"]) + ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"]) + ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"]) + ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"]) + ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"]) + ((species.loc["P2O5","no_cat"]*setup.loc[run,"P2O5"])/species.loc["P2O5","M"]) 
        Si = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"])/tot
        Ti = ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"])/tot
        Al = ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"])/tot
        Fe = ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"])/tot
        Mn = ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"])/tot
        Mg = ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"])/tot
        Ca = ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"])/tot
        Na = ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"])/tot
        K = ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"])/tot
      
        lnC = (8.77 - (23590.0/T) + (1673.0/T)*(6.7*(Na+K) + 4.9*Mg + 8.1*Ca + 8.9*(Fe+Mn) + 5.0*Ti + 1.8*Al - 22.2*Ti*(Fe+Mn) + 7.2*Fe*Si) - 2.06*math.erf(-7.2*(Fe+Mn)))
     
    elif model == "ONeill21hyd": # # O'Neill (2021) Magma Redox Geochemistry (AGU Monograph Series); includes dilution effect and additional H term (eqn 49)
        H2O = melt_wf["H2OT"]*100.
        
        # Mole fractions in the melt on cationic lattice (all Fe as FeO) adjusted for H2O content
        Wm_tot = setup.loc[run,"SiO2"] + setup.loc[run,"TiO2"] + setup.loc[run,"Al2O3"] + Wm_FeOT(run,setup,species) + setup.loc[run,"MnO"] + setup.loc[run,"MgO"] + setup.loc[run,"MnO"] + setup.loc[run,"CaO"] + setup.loc[run,"Na2O"] + setup.loc[run,"K2O"]
        tot = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"]) + ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"]) + ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"]) + ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"]) + ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"]) + ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"]) + ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"]) + ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"]) + ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"]) + ((species.loc["P2O5","no_cat"]*setup.loc[run,"P2O5"])/species.loc["P2O5","M"]) 
        HO = ((species.loc["H2O","no_cat"]*H2O)/species.loc["H2O","M"])/((((100.0-H2O)*tot)/Wm_tot)+((species.loc["H2O","no_cat"]*H2O)/species.loc["H2O","M"]))
        Si = (1.0-HO)*((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"])/tot
        Ti = (1.0-HO)*((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"])/tot
        Al = (1.0-HO)*((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"])/tot
        Fe = (1.0-HO)*((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"])/tot
        Mn = (1.0-HO)*((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"])/tot
        Mg = (1.0-HO)*((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"])/tot
        Ca = (1.0-HO)*((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"])/tot
        Na = (1.0-HO)*((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"])/tot
        K = (1.0-HO)*((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"])/tot
        
        lnCH = (HO*(6.4 + 12.4*HO - 20.3*Si + 73.0*(Na+K)))
        lnCdil = (8.77 - (23590.0/T) + (1673.0/T)*(6.7*(Na+K) + 4.9*Mg + 8.1*Ca + 8.9*(Fe+Mn) + 5.0*Ti + 1.8*Al - 22.2*Ti*(Fe+Mn) + 7.2*Fe*Si) - 2.06*math.erf(-7.2*(Fe+Mn)))
        lnC = lnCdil+lnCH

    elif model == "ONeill21dil": # O'Neill (2021) Magma Redox Geochemistry (AGU Monograph Series); includes dilution effect
        H2O = melt_wf["H2OT"]*100.
        
        # Mole fractions in the melt on cationic lattice (all Fe as FeO) adjusted for H2O content
        Wm_tot = setup.loc[run,"SiO2"] + setup.loc[run,"TiO2"] + setup.loc[run,"Al2O3"] + Wm_FeOT(run,setup,species) + setup.loc[run,"MnO"] + setup.loc[run,"MgO"] + setup.loc[run,"MnO"] + setup.loc[run,"CaO"] + setup.loc[run,"Na2O"] + setup.loc[run,"K2O"]
        tot = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"]) + ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"]) + ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"]) + ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"]) + ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"]) + ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"]) + ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"]) + ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"]) + ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"]) + ((species.loc["P2O5","no_cat"]*setup.loc[run,"P2O5"])/species.loc["P2O5","M"]) 
        HO = ((species.loc["H2O","no_cat"]*H2O)/species.loc["H2O","M"])/((((100.0-H2O)*tot)/Wm_tot)+((species.loc["H2O","no_cat"]*H2O)/species.loc["H2O","M"]))
        Si = (1.0-HO)*((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"])/tot
        Ti = (1.0-HO)*((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"])/tot
        Al = (1.0-HO)*((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"])/tot
        Fe = (1.0-HO)*((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"])/tot
        Mn = (1.0-HO)*((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"])/tot
        Mg = (1.0-HO)*((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"])/tot
        Ca = (1.0-HO)*((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"])/tot
        Na = (1.0-HO)*((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"])/tot
        K = (1.0-HO)*((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"])/tot
        
        lnC = (8.77 - (23590.0/T) + (1673.0/T)*(6.7*(Na+K) + 4.9*Mg + 8.1*Ca + 8.9*(Fe+Mn) + 5.0*Ti + 1.8*Al - 22.2*Ti*(Fe+Mn) + 7.2*Fe*Si) - 2.06*math.erf(-7.2*(Fe+Mn)))

    C = math.exp(lnC)    
    return C



########################################
### solubility constant for sulphate ###
########################################
def C_SO4(run,PT,melt_wf,setup,species,models): ### C_SO4 = wmS6+*(fS2*fO2^3)^-0.5 ### (weight ppm)
    model = models.loc["sulphate","option"]
    T = PT['T'] + 273.15 # T in Kelvin
    P = PT['P'] # P in bars
    if model == "Nash19": # Nash et al. (2019) EPSL 507:187-198
        S = 1. # S6+/S2- ratio of S6+/S2- of 0.5
        Csulphide = C_S(run,PT,melt_wf,setup,species,models)
        A = PT_KCterm(run,PT,setup,species,models) # P, T, compositional term from Kress & Carmicheal (1991)
        B = (8743600/T**2) - (27703/T) + 20.273 # temperature dependence from Nash et al. (2019)
        a = 0.196 # alnfO2 from Kress & Carmicheal (1991)
        F = 10**(((math.log10(S))-B)/8.)
        fO2 = math.exp(((math.log(0.5*F))-A)/a)
        Csulphate = (S*Csulphide)/(fO2**2)
    elif model == "S6ST":
        Csulphide = C_S(run,PT,melt_wf,setup,species,models)
        fO2 = f_O2(run,PT,melt_wf,setup,species,models)
        S6ST_ = melt_wf["S6ST"]
        S = overtotal2ratio(S6ST_)
        Csulphate = (S*Csulphide)/(fO2**2)
    elif model == "Boulliung22nP": # Boullioung & Wood (2022) GCA 336:150-164 [eq5] - corrected!
        tot = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"]) + ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"]) + ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"]) + ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"]) + ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"]) + ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"]) + ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"]) + ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"]) + ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"]) + ((species.loc["P2O5","no_cat"]*setup.loc[run,"P2O5"])/species.loc["P2O5","M"]) 
        Si = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"])/tot
        Ti = ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"])/tot
        Al = ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"])/tot
        Fe = ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"])/tot
        Mn = ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"])/tot
        Mg = ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"])/tot
        Ca = ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"])/tot
        Na = ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"])/tot
        K = ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"])/tot
        logCS6 = -12.948 + ((15602.*Ca + 28649.*Na - 9596.*Mg + 4194.*Al +16016.*Mn + 29244.)/T) # wt% S
        Csulphate = (10.**logCS6)*10000. # ppm S
    elif model == "Boulliung22wP": # Boullioung & Wood (2022) GCA 336:150-164 [eq13]
        tot = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"]) + ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"]) + ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"]) + ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"]) + ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"]) + ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"]) + ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"]) + ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"]) + ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"]) + ((species.loc["P2O5","no_cat"]*setup.loc[run,"P2O5"])/species.loc["P2O5","M"]) 
        Si = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"])/tot
        Ti = ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"])/tot
        Al = ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"])/tot
        Fe = ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"])/tot
        Mn = ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"])/tot
        Mg = ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"])/tot
        Ca = ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"])/tot
        Na = ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"])/tot
        K = ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"])/tot
        logCS6 = -12.659 + ((3692.*Ca - 7592.*Si - 13736.*Ti + 3762.*Al + 34483)/T) - (0.1*P*1.5237)/T # wt% S
        Csulphate = (10.**logCS6)*10000. # ppm S
    
    elif model == "ONeill22": # O'Neill & Mavrogenes (2022) GCA 334:368-382 eq[12a]
        # Mole fractions in the melt on cationic lattice (all Fe as FeO) no volatiles
        tot = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"]) + ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"]) + ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"]) + ((species.loc["FeO","no_cat"]*Wm_FeO(run,melt_wf,setup,species))/species.loc["FeO","M"]) + ((species.loc["Fe2O3","no_cat"]*Wm_Fe2O3(run,melt_wf,setup,species))/species.loc["Fe2O3","M"]) + ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"]) + ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"]) + ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"]) + ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"]) + ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"]) + ((species.loc["P2O5","no_cat"]*setup.loc[run,"P2O5"])/species.loc["P2O5","M"]) 
        Si = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"])/tot
        Ti = ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"])/tot
        Al = ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"])/tot
        Fe2 = ((species.loc["FeO","no_cat"]*Wm_FeO(run,melt_wf,setup,species))/species.loc["FeO","M"])/tot
        Mn = ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"])/tot
        Mg = ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"])/tot
        Ca = ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"])/tot
        Na = ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"])/tot
        K = ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"])/tot
        lnC = -8.02 + ((21100. + 44000.*Na + 18700.*Mg + 4300.*Al + 44200.*K + 35600.*Ca + 12600.*Mn + 16500.*Fe2)/T) #CS6+ = [S6+, ppm]/fSO3 
        Csulphate = gp.exp(lnC)*KOSg2(PT) # ppm S
    
    return Csulphate



###################################
### solubility constant for H2S ###
###################################
def C_H2S(run,PT,melt_wf,setup,species,models): # C_H2S = wmH2S/fH2S (ppm H2S, fH2S bar)
    model = models.loc["hydrogen sulfide","option"]
    if model == "basalt":
        K = 10.23 # fitted to basalt data from Moune+ 2009 CMP 157:691–707 and Lesne+ 2015 ChemGeol 418:104–116
    elif model == "basaltic andesite":
        K = 6.82 # fitted to basaltic andesite data from Moune+ 2009 CMP 157:691–707 and Lesne+ 2015 ChemGeol 418:104–116 
    return K



########################################
### solubility constant for hydrogen ###
########################################
def C_H2(PT,models): # C_H2 = wmH2/fH2 (wtppm)
    # Hirchmann et al. (2012) EPSL 345-348:38-48
    model = models.loc["hydrogen","option"] 
    R = 83.144598 # bar cm3 /mol /K
    P = PT['P'] # pressure in bars
    T = PT['T'] + 273.15 # T in Kelvin SHOULD BE T0
    P0 = 100.*0.01 # kPa to bars
    if model == "basalt":
        #lnK0 = -11.4 # T0 = 1400 'C, P0 = 100 kPa for mole fraction H2
        lnK0 = -0.9624 # for ppm H2 (fitted in excel)
        DV = 10.6 # cm3/mol
    elif model == "andesite":
        #lnK0 = -10.6 # T0 = 1400 'C, P0 = 100 kPa for mole fraction H2
        lnK0 = -0.1296 # for ppm H2 (fitted in excel)
        DV = 11.3 # cm3/mol
    lnK = lnK0 - (DV*(P-P0))/(R*T) # = ln(XH2/fH2) in ppm/bar
    C = gp.exp(lnK) 
    return C



######################################
### solubility constant for methane ##
######################################
def C_CH4(PT,models): # C_CH4 = wmCH4/fCH4 (ppm)
    model = models.loc["methane","option"]
    if model == "Ardia13": # Ardia et al. (2013) GCA 114:52-71
        R = 83.144598 # bar cm3 /mol /K 
        P = PT['P'] # pressure in bars
        T = PT['T'] + 273.15 # T in Kelvin SHOULD BE T0
        P0 = 100.*0.01 # kPa to bars
        lnK0 = 4.93 # ppm CH4 
        #lnK0 = -7.63 # mole fraction CH4
        DV = 26.85 # cm3/mol
        lnK = lnK0 - (DV*(P-P0))/(R*T) 
        K_ = gp.exp(lnK) # for fCH4 in GPa
        K = 0.0001*K_ # for fCH4 in bars 
    return K



#################################
### solubility constant for CO ##
#################################
def C_CO(PT,models): # C_CO = wmCO/fCO (ppm)
    model = models.loc["carbon monoxide","option"]
    if model == "basalt": # from fitting Armstrong et al. (2015) GCA 171:283-302; Stanley+2014, and Wetzel+13 thermodynamically
        R = 83.144598 # bar cm3 /mol /K 
        P = PT['P'] # pressure in bars
        T = PT['T'] +273.15 # T in Kelvin
        P0 = 1. # in bars
        lnK0 = -2.11 # ppm CO
        DV = 15.20 # cm3/mol
        lnK = lnK0 - (DV*(P-P0))/(R*T) 
        K = gp.exp(lnK) # CO(ppm)/fCO(bars)
    return K


#################################################################################################################################
################################################ solid/liquid volatile saturation ###############################################
#################################################################################################################################

################################################
### sulphate content at anhydrite saturation ###
################################################
def SCAS(run,PT,melt_wf,setup,species,models): 
    model = models.loc["SCAS","option"]
    T = PT['T'] +273.15
    wm_H2OT = 100*melt_wf['H2OT']
    
    Wm_tot = setup.loc[run,"SiO2"] + setup.loc[run,"TiO2"] + setup.loc[run,"Al2O3"] + Wm_FeOT(run,setup,species) + setup.loc[run,"MnO"] + setup.loc[run,"MgO"] + setup.loc[run,"MnO"] + setup.loc[run,"CaO"] + setup.loc[run,"Na2O"] + setup.loc[run,"K2O"]
    mol_tot = ((100.0-wm_H2OT)*(((setup.loc[run,"SiO2"]/species.loc["SiO2","M"]) + (setup.loc[run,"TiO2"]/species.loc["TiO2","M"]) + (setup.loc[run,"Al2O3"]/species.loc["Al2O3","M"]) + (Wm_FeOT(run,setup,species)/species.loc["FeO","M"]) + (setup.loc[run,"MnO"]/species.loc["MnO","M"]) + (setup.loc[run,"MgO"]/species.loc["MgO","M"]) + (setup.loc[run,"CaO"]/species.loc["CaO","M"]) + (setup.loc[run,"Na2O"]/species.loc["Na2O","M"]) + (setup.loc[run,"K2O"]/species.loc["K2O","M"]))/Wm_tot)) + wm_H2OT/species.loc["H2O","M"]
    Si = (((100.0-wm_H2OT)*(setup.loc[run,"SiO2"]/Wm_tot))/species.loc["SiO2","M"])/mol_tot
    Ca = (((100.0-wm_H2OT)*(setup.loc[run,"CaO"]/Wm_tot))/species.loc["CaO","M"])/mol_tot
    Mg = (((100.0-wm_H2OT)*(setup.loc[run,"MgO"]/Wm_tot))/species.loc["MgO","M"])/mol_tot                                                                                            
    Fe = (((100.0-wm_H2OT)*(Wm_FeOT(run,setup,species)/Wm_tot))/species.loc["FeO","M"])/mol_tot
    Al = (((100.0-wm_H2OT)*(setup.loc[run,"Al2O3"]/Wm_tot))/species.loc["Al2O3","M"])/mol_tot
    Na = (((100.0-wm_H2OT)*(setup.loc[run,"Na2O"]/Wm_tot))/species.loc["Na2O","M"])/mol_tot
    K = (((100.0-wm_H2OT)*(setup.loc[run,"K2O"]/Wm_tot))/species.loc["K2O","M"])/mol_tot
    H = (((100.0-wm_H2OT)*(wm_H2OT/Wm_tot))/species.loc["H2O","M"])/mol_tot
    
    if model == "Chowdhury18": # sulphate content (ppm) at anhydrite saturation from Chowdhury & Dasgupta (2018) Chem. Geo. 522:162-174 [T in K]
        a = -13.23
        b = -0.50
        dSi = 3.02
        dCa = 36.70
        dMg = 2.84
        dFe = 10.14
        dAl = 44.28
        dNa = 26.27
        dK = -25.77
        e = 0.09
        f = 0.54
        dX = dSi*Si + dCa*Ca + dMg*Mg + dFe*Fe + dAl*Al + dNa*Na + dK*K
        lnxm_SO4 = a + b*((10.0**4.0)/T) + dX + e*wm_H2OT - f*gp.log(Ca)                                                                                  
        xm_SO4 = gp.exp(lnxm_SO4) 
        Xm_SO4 = xm_SO4*(xm_SO4 + mol_tot)
        S6CAS = Xm_SO4*species.loc["S","M"]*10000.0
                           
    elif model == "Zajacz19": # Zajacz & Tsay (2019) GCA 261:288-304
        if Na + K + Ca >= Al:
            P_Rhyo = 3.11*(Na+K+Ca-Al)
        else:
            P_Rhyo = 1.54*(Al-(Na+K+Ca))
        NBOT = (2.*Na+2.*K+2.*(Ca+Mg+Fe)-Al*2.)/(Si+2.*Al) # according to spreadsheet not paper
        P_T = gp.exp(-7890./T)
        P_H2O = H*(2.09 - 1.65*NBOT) + 0.42*NBOT + 0.23
        P_C = ((P_Rhyo + 251.*Ca**2. + 57.*Mg**2. + 154.*Fe**2.)/(2.*Al + Si))/(1. + 4.8*NBOT)
        Ksm_SPAnh = gp.exp(1.226*gp.log(P_C*P_T*P_H2O) + 0.079)                         
        Xsm_S = Ksm_SPAnh/Ca  
        S6CAS = Xsm_S*mol_tot*32.07*10000.
    return S6CAS

###############################################
### sulphide content at sulphide saturation ###
###############################################
def SCSS(run,PT,melt_wf,setup,species,models): # sulphide content (ppm) at sulphide saturation from O'Neill (2021) Magma Redox Geochemistry (AGU Monograph Series) [P in bar,T in K]
    model = models.loc["sulphide","option"]
    P_bar = PT['P']
    T = PT['T'] + 273.15
    Fe3FeT = melt_wf["Fe3FeT"]
    
    # Mole fractions in the melt on cationic lattice (all Fe as FeO) no volatiles
    tot = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"]) + ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"]) + ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"]) + ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"]) + ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"]) + ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"]) + ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"]) + ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"]) + ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"]) + ((species.loc["P2O5","no_cat"]*setup.loc[run,"P2O5"])/species.loc["P2O5","M"]) 
    Si = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"])/tot
    Ti = ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"])/tot
    Al = ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"])/tot
    Fe = ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"])/tot
    Mn = ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"])/tot
    Mg = ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"])/tot
    Ca = ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"])/tot
    Na = ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"])/tot
    K = ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"])/tot
    Fe2 = Fe*(1-Fe3FeT)
    
    R = 8.31441
    P = (1.0e-4)*P_bar # pressure in GPa
    D = (137778.0 - 91.66*T + 8.474*T*gp.log(T)) # J/mol
    sulphide_comp = 1.0 # assumes the sulphide is pure FeS (no Ni, Cu, etc.)
        
    if model == "ONeill20hyd": # includes dilution effect and additional H term (eqn 49)
        Wm_tot = setup.loc[run,"SiO2"] + setup.loc[run,"TiO2"] + setup.loc[run,"Al2O3"] + Wm_FeOT(run,setup,species) + setup.loc[run,"MnO"] + setup.loc[run,"MgO"] + setup.loc[run,"MnO"] + setup.loc[run,"CaO"] + setup.loc[run,"Na2O"] + setup.loc[run,"K2O"]
        HO = ((species.loc["H2O","no_cat"]*setup.loc[run,"H2O"])/species.loc["H2O","M"])/((((100.0-setup.loc[run,"H2O"])*tot)/Wm_tot)+((species.loc["H2O","no_cat"]*setup.loc[run,"H2O"])/species.loc["H2O","M"]))
        Si = Si*(1.0-HO)
        Ti = Ti*(1.0-HO)
        Al = Al*(1.0-HO)
        Fe = Fe*(1.0-HO)
        Mn = Mn*(1.0-HO)
        Mg = Mg*(1.0-HO)
        Ca = Ca*(1.0-HO)
        Na = Na*(1.0-HO)
        K = K*(1.0-HO)
        Fe2 = Fe2*(1.0-HO)
        lnaFeS = gp.log((1.0 - Fe2)*sulphide_comp)
        lnyFe2 = (((1.0-Fe2)**2.0)*(28870.0 - 14710.0*Mg + 1960.0*Ca + 43300.0*Na + 95380.0*K - 76880.0*Ti) + (1.0-Fe2)*(-62190.0*Si + 31520.0*Si**2.0))/(R*T)
    else: # model == "ONeill20": # no effect of H2O included
        lnaFeS = gp.log((1.0 - Fe2)*sulphide_comp)
        lnyFe2 = (((1.0-Fe2)**2.0)*(28870.0 - 14710.0*Mg + 1960.0*Ca + 43300.0*Na + 95380.0*K - 76880.0*Ti) + (1.0-Fe2)*(-62190.0*Si + 31520.0*Si**2.0))/(R*T)
    lnS = D/(R*T) + gp.log(C_S(run,PT,melt_wf,setup,species,models)) - gp.log(Fe2) - lnyFe2 + lnaFeS + (-291.0*P + 351.0*gp.erf(P))/T
    return gp.exp(lnS)




#################################################################################################################################
#################################################### EQUILIBRIUM CONSTANTS ######################################################
#################################################################################################################################

# H2 + 0.5O2 = H2O
# K = fH2O/(fH2*(fO2)^0.5)
def KHOg(PT): # Kerrick & Ohmoto (1997) Am. J. Sci. 277:1013–1044
    T_K = PT['T']+273.15
    return pow(10,((12510.0/T_K)-0.979*(gp.log10(T_K))+0.483))

# H2O + 0.5S2 = H2S + 0.5O2
# K = (fH2S*(fO2)^0.5)/((fS2^0.5)*fH2O)
def KHOSg(PT): # Kerrick & Ohmoto (1997) Am. J. Sci. 277:1013–1044
    T_K = PT['T']+273.15
    return pow(10,((-8117.0/T_K)+0.188*gp.log10(T_K)-0.352))

# 0.5S2 + O2 = SO2
# K = fSO2/((fS2^0.5)*fO2)
def KOSg(PT): # Kerrick & Ohmoto (1997) Am. J. Sci. 277:1013–1044
    T_K = PT['T']+273.15
    return 10.**((18929.0/T_K)-3.783)

# 0.5S2 + 1.5O2 = SO3
# K = fSO3/((fS2^0.5)*(fO2^1.5)
def KOSg2(PT):
    T_K = PT['T']+273.15
    lnK = (55921./T_K) - 25.07 + 0.6465*gp.log(T_K) #O'Neill & Mavrogenes (2022) GCA 334:368-382 from JANF 
    return gp.exp(lnK)

# CO + 0.5O = CO2
# K = fCO2/(fCO*(fO2^0.5))
def KCOg(PT): # Kerrick & Ohmoto (1997) Am. J. Sci. 277:1013–1044
    T_K = PT['T']+273.15
    return pow(10,((14751.0/T_K)-4.535))

# CH4 + 2O2 = CO2 + 2H2O
# K = (fCO2*(fH2O^2))/(fCH4*(fO2^2))
def KCOHg(PT): # Kerrick & Ohmoto (1997) Am. J. Sci. 277:1013–1044
    T_K = PT['T']+273.15
    return pow(10,((41997.0/T_K)+0.719*gp.log10(T_K)-2.404))

def KOCSg(PT,models): # OCS - depends on system
    T = PT['T']+273.15
    OCSmodel = models.loc["carbonylsulphide","option"]
    if OCSmodel == "COHS": # from EVo
    # OCS + H2O = CO2 + H2S
    # K = (fCO2*fH2S)/(fOCS*fH2O)
        return gp.exp(0.482 + (16.166e-2/T) + 0.081e-3*T - (5.715e-3/T**2) - 2.224e-1*gp.log(T))
    else:
    # 2CO2 + OCS = 3CO + SO2 - Moussallam et al. (2019) EPSL 520:260-267
    # K = (fCO^3*fSO2)/(fCO2^2*fOCS)
        return 10.0**(9.24403 - (15386.45/T)) # P and f in bars, T in K ***NOT GP PRECISION***

# H2Omol + O = 2OH
# K = xOH*2/(xH2Omol*xO)
def KHOm(PT,models):
    Hspeccomp = models.loc["Hspeccomp","option"]
    T_K = PT['T']+273.15
    
    if Hspeccomp == "rhyolite": # Zhang (1999) Reviews in Geophysics 37(4):493-516
        a = -3120.0
        b = 1.89
        K = gp.exp((a/T_K) + b)
    elif Hspeccomp == "alkali basalt": # average of eqn-15-17 from Lesne et al. (2010) CMP 162:133-151
        a = -8348.0 # VES-9 = -8033.0, ETN-1 = -8300.0, and PST-9 = -8710.0
        b = 7.8108 # VES-9 = 7.4222, ETN-1 = 7.4859, and PEST-9 = 8.5244
        K = gp.exp((a/T_K) + b)
    elif Hspeccomp == "PST-9": # Lesne et al. (2010) CMP 162:133-151 eqn 15
        a = -8710.0
        b = 8.5244
        K = gp.exp((a/T_K) + b)
    elif Hspeccomp == "VES-9": # Lesne et al. (2010) CMP 162:133-151 eqn 16
        a = -8033.0
        b = 7.4222
        K = gp.exp((a/T_K) + b)
    elif Hspeccomp == "ETN-1": # Lesne et al. (2010) CMP 162:133-151 eqn 17
        a = -8300.0
        b = 7.4859
        K = gp.exp((a/T_K) + b)
    elif Hspeccomp == "andesite": # eqn-7 from Botcharnikov et al. (2006) Chem. Geol. 229(1-3)125-143
        a = -3650.0
        b = 2.99
        K = gp.exp((a/T_K) + b)
    elif Hspeccomp == "MORB": # fit to Dixon et al. (1995) data digitised from Lesne et al. (2010) CMP 162:133-151
        a = -2204.99
        b = 1.2600
        K = gp.exp((a/T_K) + b)
    elif Hspeccomp == "albite": # Silver & Stolper (1989) J.Pet 30(3)667-709
        K = 0.17
    return K

# CO2 + O = CO3
def KCOm(PT,models):
    T_K = PT['T']+273.15
    Cspeccomp = models.loc["Cspeccomp","option"]
    if Cspeccomp == "andesite": # eqn-8 from Botcharnikov et al. (2006) Chem. Geol. 229(1-3)125-143
        a = 8665.0
        b = -5.11
    elif Cspeccomp == "dacite": # Botcharnikov et al. (2006) Chem. Geol. 229(1-3)125-143
        a = 9787.0
        b = -7.69
    return gp.exp((a/T_K) + b)  

# Cgraphite + O2 = CO2
def KCOs(PT,models): # Holloway et al. (1992) Eur J. Mineral. 4:105-114 equation (3) KI
    T_K = PT['T']+273.15
    P = PT['P']
    a = 40.07639
    b = -2.5392e-2
    c = 5.27096e-6
    d = 0.0267
    log10K = a + (b*T_K) + (c*T_K**2) + ((P-1)/T_K)
    return 10.**log10K 



################################################################################################################################# 
##################################################### FUGACITY COEFFICIENTS #####################################################
#################################################################################################################################

# all fugacity coefficients are assumed to equal 1 below 1 bar.

###################
### H2O and CO2 ###
###################

def CORK(PT,p0,a,b,c,d): # Holland & Powell (1991) CMP 1092(109):265–273
    P = PT['P']
    T_K = PT['T']+273.15
    def MRK(P_kb,VMRK,R,T_K,a,b): # MRK volume equation rearranged to equal 0
        return P_kb*pow(VMRK,3.0) - R*T_K*pow(VMRK,2.0) - (b*R*T_K + pow(b,2.0)*P_kb - a*pow(T_K,-0.5))*VMRK - (a*b)*pow(T_K,-0.5)

    def dMRK(P_kb,VMRK,R,T_K,a,b): # derivative of above
        return 3.0*P_kb*pow(VMRK,2.0) - 2.0*R*T_K*VMRK - (b*R*T_K + pow(b,2.0)*P_kb - a*pow(T_K,-0.5))

    def dVMRK(MRK,P_kb,VMRK,R,T_K,a,b):
        return abs(0-MRK(P_kb,VMRK,R,T_K,a,b))

    def NR_VMRK(MRK, dMRK, VMRK0, e1, P_kb,R,T_K,a,b):
        delta1 = dVMRK(MRK,P_kb,VMRK0,R,T_K,a,b)
        while delta1 > e1:
            VMRK0 = VMRK0 - MRK(P_kb,VMRK0,R,T_K,a,b)/dMRK(P_kb,VMRK0,R,T_K,a,b)
            delta1 = dVMRK(MRK,P_kb,VMRK0,R,T_K,a,b)
        return VMRK0
    
    R = 8.314e-3 # in kJ/mol/K
    P_kb = P/1000.0
    
    Vi = ((R*T_K)/P_kb) + b
        
    VMRK = NR_VMRK(MRK, dMRK, Vi, 1E20, P_kb,R,T_K,a,b)
        
    if P_kb > p0:
        V = VMRK + c*pow((P_kb-p0),0.5) + d*(P_kb-p0)
        ln_y_virial = (1/(R*T_K))*((2./3.)*c*pow((P_kb-p0),1.5) + (d/2.0)*pow((P_kb-p0),2.0))
    else:
        V = VMRK
        ln_y_virial = 0.0
        
    z = (P_kb*V)/(R*T_K)
    A = a/(b*R*pow(T_K,1.5))
    B = (b*P_kb)/(R*T_K)
        
    ln_y = z - 1.0 - gp.log(z-B) - A*gp.log(1.0 + (B/z)) + ln_y_virial
    return gp.exp(ln_y)

def y_H2O(PT,models):
    P = PT['P']
    T_K = PT['T']+273.15
    ideal_gas = models.loc["ideal_gas","option"]
    if ideal_gas == "yes":
        return 1.
    elif P < 1.: # ideal gas below 1 bar
        return 1.
    else: # (T > 673 K only) # Holland & Powell (1991) CMP 1092(109):265–273
        p0 = 2.00 # in kb
        a = 1113.4 + -0.22291*(T_K - 673.0) + -3.8022e-4*pow((T_K-673.0),2.0) + 1.7791e-7*pow((T_K-673.0),3.0)
        b = 1.465
        c = -3.025650e-2 + -5.343144e-6*T_K
        d = -3.2297554e-3 + 2.2215221e-6*T_K
        y = CORK(PT,p0,a,b,c,d)
        return y

def y_CO2(PT,species,models):
    P = PT['P']
    T_K = PT['T']+273.15
    ideal_gas = models.loc["ideal_gas","option"]
    if ideal_gas == "yes":
        return 1.0
    elif P < 1.: # ideal gas below 1 bar
        return 1.
    else:
        if models.loc["y_CO2","option"] == "HP91": # Holland & Powell (1991) CMP 1092(109):265–273
            p0 = 5.00 # in kb
            a = 741.2 + -0.10891*(T_K) + -3.4203e-4*pow(T_K,2.0)
            b = 3.057
            c = -2.26924e-1 + -7.73793e-5*T_K
            d = 1.33790e-2 + -1.1740e-5*T_K
            y = CORK(PT,p0,a,b,c,d)
        elif models.loc["y_CO2","option"] == "SS92": # Shi & Saxena (1992) Am. Min. 77:1038–1049
            gas_species = "CO2"
            y = y_SS(gas_species,PT,species,models)
        return y
    
    
##########
### H2 ###
##########

def y_H2(PT,models):
    P = PT['P']
    T_K = PT['T']+273.15
    ideal_gas = models.loc["ideal_gas","option"]
    if ideal_gas == "yes":
        return 1.0
    elif P < 1.: # ideal gas below 1 bar
        return 1.
    elif models.loc["y_H2","option"] == "SW64": # Shaw & Wones (1964) Am. J. Sci. 262:918–929
        SW1 = gp.exp(-3.8402*pow(T_K,0.125)+0.5410)
        SW2 = gp.exp(-0.1263*pow(T_K,0.5)-15.980)
        SW3 = 300*gp.exp((-0.011901*T_K)-5.941) # NB used a value of -0.011901 instead of -0.11901 as reported to match data in Table 2
        P_atm = 0.986923*P
        ln_y = SW1*P_atm - SW2*pow(P_atm,2.0) + SW3*gp.exp((-P_atm/300.0)-1.0)
        return gp.exp(ln_y)
    
###################################################################################
### O2, CO2, CO, CH4, S2, and OCS from hi & Saxena (1992) Am. Min. 77:1038–1049 ###
###################################################################################

def lny_SS(PT,Pcr,Tcr):
    P = PT['P']
    T_K = PT['T']+273.15
    Tr = T_K/Tcr
    A, B, C, D, P0, integral0 = Q_SS(PT,Tr,Pcr)
    Pr = P/Pcr
    P0r = P0/Pcr
    integral = A*gp.log(Pr/P0r) + B*(Pr - P0r) + (C/2.0)*(pow(Pr,2.0) - pow(P0r,2.0)) + (D/3.0)*(pow(Pr,3.0) - pow(P0r,3.0))
    integral_total = integral + integral0
    return integral_total

def Q_SS(PT,Tr,Pcr):
    P = PT['P']
    def Q1000(Pcr):
        Pr_ = 1000.0/Pcr
        P0r_ = 1.0/Pcr
        A0 = 1.0
        B0 = 0.9827e-1*pow(Tr,-1.0) + -0.2709*pow(Tr,-3.0)
        C0 = -0.1030e-2*pow(Tr,-1.5) + 0.1427e-1*pow(Tr,-4.0)
        D0 = 0.0
        return A0*gp.log(Pr_/P0r_) + B0*(Pr_ - P0r_) + (C0/2.0)*(pow(Pr_,2.0) - pow(P0r_,2.0)) + (D0/3.0)*(pow(Pr_,3.0) - pow(P0r_,3.0))
    def Q5000(Pcr):
        Pr_ = 5000.0/Pcr
        P0r_ = 1000.0/Pcr
        A0 = 1.0 + -5.917e-1*pow(Tr,-2.0)
        B0 = 9.122e-2*pow(Tr,-1.0)
        C0 = -1.416e-4*pow(Tr,-2.0) + -2.835e-6*gp.log(Tr)
        D0 = 0.0
        return A0*gp.log(Pr_/P0r_) + B0*(Pr_ - P0r_) + (C0/2.0)*(pow(Pr_,2.0) - pow(P0r_,2.0)) + (D0/3.0)*(pow(Pr_,3.0) - pow(P0r_,3.0))
    if P > 5000.0:
        A = 2.0614 + -2.235*pow(Tr,-2.0) + -3.941e-1*gp.log(Tr)
        B = 5.513e-2*pow(Tr,-1.0) + 3.934e-2*pow(Tr,-2.0)
        C = -1.894e-6*pow(Tr,-1.0) + -1.109e-5*pow(Tr,-2.0) + -2.189e-5*pow(Tr,-3.0)
        D = 5.053e-11*pow(Tr,-1.0) + -6.303e-21*pow(Tr,3.0)
        P0 = 5000.0
        integral0 = Q1000(Pcr) + Q5000(Pcr)
        return A, B, C, D, P0, integral0
    elif P == 5000.0:
        A = 0
        B = 0
        C = 0
        D = 0
        P0 = 5000.0
        integral0 = Q1000(Pcr) + Q5000(Pcr)
        return A, B, C, D, P0, integral0
    elif P > 1000.0 and P < 5000.0:
        A = 1.0 + -5.917e-1*pow(Tr,-2.0)
        B = 9.122e-2*pow(Tr,-1.0)
        C = -1.416e-4*pow(Tr,-2.0) + -2.835e-6*gp.log(Tr)
        D = 0.0
        P0 = 1000.0
        integral0 = Q1000(Pcr)
        return A, B, C, D, P0, integral0
    elif P == 1000.0:
        A = 0
        B = 0
        C = 0
        D = 0.0
        P0 = 1000.0
        integral0 = Q1000(Pcr)
        return A, B, C, D, P0, integral0
    else:
        A = 1.0
        B = 0.9827e-1*pow(Tr,-1.0) + -0.2709*pow(Tr,-3.0)
        C = -0.1030e-2*pow(Tr,-1.5) + 0.1427e-1*pow(Tr,-4.0)
        D = 0.0
        P0 = 1.0
        integral0 = 0.0
        return A, B, C, D, P0, integral0
    
def y_SS(gas_species,PT,species,models):
    P = PT['P']
    T_K = PT['T']+273.15
    ideal_gas = models.loc["ideal_gas","option"]
    if ideal_gas == "yes":
        return 1.0
    elif P < 1.: # ideal gas below 1 bar
        return 1.
    else:    
        Tcr = species.loc[gas_species,"Tcr"]
        Pcr = species.loc[gas_species,"Pcr"]
        return gp.exp(lny_SS(PT,Pcr,Tcr))/P    

def y_O2(PT,species,models):
    gas_species = "O2"
    y = y_SS(gas_species,PT,species,models)
    return y
    
def y_S2(PT,species,models):
    gas_species = "S2"
    y = y_SS(gas_species,PT,species,models)
    return y

def y_CO(PT,species,models):
    gas_species = "CO"
    y = y_SS(gas_species,PT,species,models)
    return y
    
def y_CH4(PT,species,models):
    gas_species = "CH4"
    y = y_SS(gas_species,PT,species,models)
    return y
    
def y_OCS(PT,species,models):
    gas_species = "OCS"
    y = y_SS(gas_species,PT,species,models)
    return y

#################################################################################
### SO2 and H2S from Shi & Saxena (1992) with option to modify below 500 bars ###
#################################################################################

def y_SO2(PT,species,models):
    P = PT['P']
    T_K = PT['T']+273.15
    ideal_gas = models.loc["ideal_gas","option"]
    gas_species = "SO2"
    if ideal_gas == "yes":
        return 1.
    elif P < 1.: # ideal gas below 1 bar
        return 1.
    else: # 1-10000 bar
        Tcr = species.loc[gas_species,"Tcr"] # critical temperature in K
        Pcr = species.loc[gas_species,"Pcr"] # critical temperature in bar
        P0 = 1.0
        P0r = P0/Pcr
        Tr = T_K/Tcr
        Q1_A, Q2_A, Q3_A, Q4_A, Q5_A, Q6_A, Q7_A, Q8_A  = 0.92854, 0.43269e-1, -0.24671, 0., 0.24999, 0., -0.53182, -0.16461e-1
        Q1_B, Q2_B, Q3_B, Q4_B, Q5_B, Q6_B, Q7_B, Q8_B  = 0.84866e-3, -0.18379e-2, 0.66787e-1, 0., -0.29427e-1, 0., 0.29003e-1, 0.54808e-2
        Q1_C, Q2_C, Q3_C, Q4_C, Q5_C, Q6_C, Q7_C, Q8_C  = -0.35456e-3, 0.23316e-4, 0.94159e-3, 0., -0.81653e-3, 0., 0.23154e-3, 0.55542e-4
        A = Q1_A + Q2_A*Tr + Q3_A*Tr**(-1.) + Q4_A*Tr**2. + Q5_A*Tr**(-2.) + Q6_A*Tr**3. + Q7_A*Tr**(-3.0) + Q8_A*gp.log(Tr)
        B = Q1_B + Q2_B*Tr + Q3_B*Tr**(-1.) + Q4_B*Tr**2. + Q5_B*Tr**(-2.) + Q6_B*Tr**3. + Q7_B*Tr**(-3.0) + Q8_B*gp.log(Tr)
        C = Q1_C + Q2_C*Tr + Q3_C*Tr**(-1.) + Q4_C*Tr**2. + Q5_C*Tr**(-2.) + Q6_C*Tr**3. + Q7_C*Tr**(-3.0) + Q8_C*gp.log(Tr)
        D = 0.0
        if P >= 500.: # above 500 bar using Shi and Saxena (1992) as is
            Pr = P/Pcr
            integral = A*gp.log(Pr/P0r) + B*(Pr - P0r) + (C/2.0)*(pow(Pr,2.0) - pow(P0r,2.0)) + (D/3.0)*(pow(Pr,3.0) - pow(P0r,3.0))
            y = (gp.exp(integral))/P
        elif models.loc["y_SO2","option"] == "SS92": # as is Shi and Saxena (1992)
            Pr = P/Pcr
            integral = A*gp.log(Pr/P0r) + B*(Pr - P0r) + (C/2.0)*(pow(Pr,2.0) - pow(P0r,2.0)) + (D/3.0)*(pow(Pr,3.0) - pow(P0r,3.0))
            y = (gp.exp(integral))/P
        elif models.loc["y_SO2","option"] == "SS92_modified": # below 500 bar linear fit between the value at 500 bar and y = 1 at 1 bar to avoid weird behaviour...
            Pr = 500./Pcr # calculate y at 500 bar
            integral = A*gp.log(Pr/P0r) + B*(Pr - P0r) + (C/2.0)*(pow(Pr,2.0) - pow(P0r,2.0)) + (D/3.0)*(pow(Pr,3.0) - pow(P0r,3.0))
            y_500 = (gp.exp(integral))/500.
            y = ((y_500 - 1.)*(P/500.)) + 1. # linear extrapolation to P of interest
        return y       
            
def y_H2S(PT,species,models):
    P = PT['P']
    T_K = PT['T']+273.15
    ideal_gas = models.loc["ideal_gas","option"]
    gas_species = "H2S"
    if ideal_gas == "yes":
        return 1.0
    elif ideal_gas == "no":
        Tcr = species.loc[gas_species,"Tcr"] # critical temperature in K 
        Pcr = species.loc[gas_species,"Pcr"] # critical temperature in bar
        Tr = T_K/Tcr
        # Q for 1-500 bar
        Q1_A_LP, Q2_A_LP, Q3_A_LP, Q4_A_LP, Q5_A_LP, Q6_A_LP, Q7_A_LP, Q8_A_LP = 0.14721e1, 0.11177e1, 0.39657e1, 0., -0.10028e2, 0., 0.45484e1, -0.382e1
        Q1_B_LP, Q2_B_LP, Q3_B_LP, Q4_B_LP, Q5_B_LP, Q6_B_LP, Q7_B_LP, Q8_B_LP = 0.16066, 0.10887, 0.29014, 0., -0.99593, 0., -0.18627, -0.45515
        Q1_C_LP, Q2_C_LP, Q3_C_LP, Q4_C_LP, Q5_C_LP, Q6_C_LP, Q7_C_LP, Q8_C_LP = -0.28933, -0.70522e-1, 0.39828, 0., -0.50533e-1, 0., 0.1176, 0.33972
        # Q for 500-10000 bar
        Q1_A_HP, Q2_A_HP, Q3_A_HP, Q4_A_HP, Q5_A_HP, Q6_A_HP, Q7_A_HP, Q8_A_HP = 0.59941, -0.1557e-2, 0.4525e-1, 0., 0.36687, 0., -0.79248, 0.26058
        Q1_B_HP, Q2_B_HP, Q3_B_HP, Q4_B_HP, Q5_B_HP, Q6_B_HP, Q7_B_HP, Q8_B_HP = 0.22545e-1, 0.17473e-2, 0.48253e-1, 0., -0.1989e-1, 0., 0.32794e-1, -0.10985e-1
        Q1_C_HP, Q2_C_HP, Q3_C_HP, Q4_C_HP, Q5_C_HP, Q6_C_HP, Q7_C_HP, Q8_C_HP = 0.57375e-3, -0.20944e-5, -0.11894e-2, 0., 0.14661e-2, 0., -0.75605e-3, -0.27985e-3
        if P < 1.:
            return 1. # ideal gas below 1 bar
        elif P < 500.:
            if models.loc["y_H2S","option"] == "SS92": # as is Shi and Saxena (1992) 
                A = Q1_A_LP + Q2_A_LP*Tr + Q3_A_LP*Tr**(-1.) + Q4_A_LP*Tr**2. + Q5_A_LP*Tr**(-2.) + Q6_A_LP*Tr**3. + Q7_A_LP*Tr**(-3.0) + Q8_A_LP*gp.log(Tr)
                B = Q1_B_LP + Q2_B_LP*Tr + Q3_B_LP*Tr**(-1.) + Q4_B_LP*Tr**2. + Q5_B_LP*Tr**(-2.) + Q6_B_LP*Tr**3. + Q7_B_LP*Tr**(-3.0) + Q8_B_LP*gp.log(Tr)
                C = Q1_C_LP + Q2_C_LP*Tr + Q3_C_LP*Tr**(-1.) + Q4_C_LP*Tr**2. + Q5_C_LP*Tr**(-2.) + Q6_C_LP*Tr**3. + Q7_C_LP*Tr**(-3.0) + Q8_C_LP*gp.log(Tr)
                D = 0.0
                P0 = 1.0
                integral0 = 0.
            elif models.loc["y_SO2","option"] == "SS92_modified": # below 500 bar linear fit between the value at 500 bar and y = 1 at 1 bar to avoid weird behaviour... 
                P0 = 500.0 # calculate y at 500 bars
                Pr_ = 500.0/Pcr
                P0r_ = 1.0/Pcr
                A0 = Q1_A_LP + Q2_A_LP*Tr + Q3_A_LP*Tr**(-1.) + Q4_A_LP*Tr**2. + Q5_A_LP*Tr**(-2.) + Q6_A_LP*Tr**3. + Q7_A_LP*Tr**(-3.0) + Q8_A_LP*gp.log(Tr)
                B0 = Q1_B_LP + Q2_B_LP*Tr + Q3_B_LP*Tr**(-1.) + Q4_B_LP*Tr**2. + Q5_B_LP*Tr**(-2.) + Q6_B_LP*Tr**3. + Q7_B_LP*Tr**(-3.0) + Q8_B_LP*gp.log(Tr)
                C0 = Q1_C_LP + Q2_C_LP*Tr + Q3_C_LP*Tr**(-1.) + Q4_C_LP*Tr**2. + Q5_C_LP*Tr**(-2.) + Q6_C_LP*Tr**3. + Q7_C_LP*Tr**(-3.0) + Q8_C_LP*gp.log(Tr)
                D0 = 0.0
                integral0 = A0*gp.log(Pr_/P0r_) + B0*(Pr_ - P0r_) + (C0/2.0)*(pow(Pr_,2.0) - pow(P0r_,2.0)) + (D0/3.0)*(pow(Pr_,3.0) - pow(P0r_,3.0))            
                y_500 = gp.exp(integral0)/500.
                y = ((y_500 - 1.)*(P/500.)) + 1. # linear extrapolation to P of interest
                return y
        elif P == 500.:
            A = 0.0
            B = 0.0
            C = 0.0
            D = 0.0
            P0 = 500.0
            Pr_ = 500.0/Pcr
            P0r_ = 1.0/Pcr
            A0 = Q1_A_LP + Q2_A_LP*Tr + Q3_A_LP*Tr**(-1.) + Q4_A_LP*Tr**2. + Q5_A_LP*Tr**(-2.) + Q6_A_LP*Tr**3. + Q7_A_LP*Tr**(-3.0) + Q8_A_LP*gp.log(Tr)
            B0 = Q1_B_LP + Q2_B_LP*Tr + Q3_B_LP*Tr**(-1.) + Q4_B_LP*Tr**2. + Q5_B_LP*Tr**(-2.) + Q6_B_LP*Tr**3. + Q7_B_LP*Tr**(-3.0) + Q8_B_LP*gp.log(Tr)
            C0 = Q1_C_LP + Q2_C_LP*Tr + Q3_C_LP*Tr**(-1.) + Q4_C_LP*Tr**2. + Q5_C_LP*Tr**(-2.) + Q6_C_LP*Tr**3. + Q7_C_LP*Tr**(-3.0) + Q8_C_LP*gp.log(Tr)
            D0 = 0.0
            integral0 = A0*gp.log(Pr_/P0r_) + B0*(Pr_ - P0r_) + (C0/2.0)*(pow(Pr_,2.0) - pow(P0r_,2.0)) + (D0/3.0)*(pow(Pr_,3.0) - pow(P0r_,3.0))            
        elif P > 500.:
            A = Q1_A_HP + Q2_A_HP*Tr + Q3_A_HP*Tr**(-1.) + Q4_A_HP*Tr**2. + Q5_A_HP*Tr**(-2.) + Q6_A_HP*Tr**3. + Q7_A_HP*Tr**(-3.0) + Q8_A_HP*gp.log(Tr)
            B = Q1_B_HP + Q2_B_HP*Tr + Q3_B_HP*Tr**(-1.) + Q4_B_HP*Tr**2. + Q5_B_HP*Tr**(-2.) + Q6_B_HP*Tr**3. + Q7_B_HP*Tr**(-3.0) + Q8_B_HP*gp.log(Tr)
            C = Q1_C_HP + Q2_C_HP*Tr + Q3_C_HP*Tr**(-1.) + Q4_C_HP*Tr**2. + Q5_C_HP*Tr**(-2.) + Q6_C_HP*Tr**3. + Q7_C_HP*Tr**(-3.0) + Q8_C_HP*gp.log(Tr)
            D = 0.0
            P0 = 500.0
            Pr_ = 500.0/Pcr
            P0r_ = 1.0/Pcr
            A0 = Q1_A_LP + Q2_A_LP*Tr + Q3_A_LP*Tr**(-1.) + Q4_A_LP*Tr**2. + Q5_A_LP*Tr**(-2.) + Q6_A_LP*Tr**3. + Q7_A_LP*Tr**(-3.0) + Q8_A_LP*gp.log(Tr)
            B0 = Q1_B_LP + Q2_B_LP*Tr + Q3_B_LP*Tr**(-1.) + Q4_B_LP*Tr**2. + Q5_B_LP*Tr**(-2.) + Q6_B_LP*Tr**3. + Q7_B_LP*Tr**(-3.0) + Q8_B_LP*gp.log(Tr)
            C0 = Q1_C_LP + Q2_C_LP*Tr + Q3_C_LP*Tr**(-1.) + Q4_C_LP*Tr**2. + Q5_C_LP*Tr**(-2.) + Q6_C_LP*Tr**3. + Q7_C_LP*Tr**(-3.0) + Q8_C_LP*gp.log(Tr)
            D0 = 0.0
            integral0 = A0*gp.log(Pr_/P0r_) + B0*(Pr_ - P0r_) + (C0/2.0)*(pow(Pr_,2.0) - pow(P0r_,2.0)) + (D0/3.0)*(pow(Pr_,3.0) - pow(P0r_,3.0))
        P0r = P0/Pcr
        Pr = P/Pcr
        integral = A*gp.log(Pr/P0r) + B*(Pr - P0r) + (C/2.0)*(pow(Pr,2.0) - pow(P0r,2.0)) + (D/3.0)*(pow(Pr,3.0) - pow(P0r,3.0))
        return gp.exp(integral + integral0)/P
    
    

#################################################################################################################################
########################################### FUGACITY, MOLE FRACTION, PARTIAL PRESSURE ###########################################
#################################################################################################################################

def f_H2O(run,PT,melt_wf,setup,species,models):
    Hspeciation = models.loc["Hspeciation","option"]
    if Hspeciation == "none": # fH2O = xmH2OT^2/CH2O
        value = ((xm_H2OT_so(run,melt_wf,setup,species))**2.0)/C_H2O(PT,models)
        return value
    else: # regular or ideal: fH2O = xmH2Omol/CH2O
        value = xm_H2Omol_so(run,PT,melt_wf,setup,species,models)/C_H2O(PT,models)
        return value
    
def f_CO2(run,PT,melt_wf,setup,species,models):
    f = xm_CO2_so(run,melt_wf,setup,species)/C_CO3(run,PT,melt_wf,setup,species,models)
    return f

def f_S2(run,PT,melt_wf,setup,species,models): # wtppm S2- NOT mole fraction due to parameterisation by O'Neill (2020)
    K = C_S(run,PT,melt_wf,setup,species,models)/1000000.
    fS2 = ((melt_wf["S2-"]/K)**2.)*f_O2(run,PT,melt_wf,setup,species,models)
    return fS2
    
def f_H2(run,PT,melt_wf,setup,species,models):
    K = KHOg(PT)
    return f_H2O(run,PT,melt_wf,setup,species,models)/(K*pow(f_O2(run,PT,melt_wf,setup,species,models),0.5))

def f_CO(run,PT,melt_wf,setup,species,models):
    K = KCOg(PT)
    return f_CO2(run,PT,melt_wf,setup,species,models)/(K*pow(f_O2(run,PT,melt_wf,setup,species,models),0.5))

def f_H2S(run,PT,melt_wf,setup,species,models):
    K = KHOSg(PT)
    return (K*pow(f_S2(run,PT,melt_wf,setup,species,models),0.5)*f_H2O(run,PT,melt_wf,setup,species,models))/pow(f_O2(run,PT,melt_wf,setup,species,models),0.5)

def f_SO2(run,PT,melt_wf,setup,species,models):
    K = KOSg(PT)
    return K*f_O2(run,PT,melt_wf,setup,species,models)*f_S2(run,PT,melt_wf,setup,species,models)**0.5

def f_SO3(run,PT,melt_wf,setup,species,models):
    K = KOSg2(PT)
    return K*(f_O2(run,PT,melt_wf,setup,species,models))**1.5*(f_S2(run,PT,melt_wf,setup,species,models))**0.5

def f_CH4(run,PT,melt_wf,setup,species,models):
    K = KCOHg(PT)
    return (f_CO2(run,PT,melt_wf,setup,species,models)*pow(f_H2O(run,PT,melt_wf,setup,species,models),2.0))/(K*pow(f_O2(run,PT,melt_wf,setup,species,models),2.0))

def f_OCS(run,PT,melt_wf,setup,species,models):
    OCSmodel = models.loc["carbonylsulphide","option"]
    K = KOCSg(PT,models)
    if OCSmodel == "COHS":
        if f_H2O(run,PT,melt_wf,setup,species,models) > 0.0:
            return (f_CO2(run,PT,melt_wf,setup,species,models)*f_H2S(run,PT,melt_wf,setup,species,models))/(f_H2O(run,PT,melt_wf,setup,species,models)*K)
        else:
            return 0.0
    else:
        if f_CO2(run,PT,melt_wf,setup,species,models) > 0.0:
            return ((f_CO(run,PT,melt_wf,setup,species,models)**3.0)*f_SO2(run,PT,melt_wf,setup,species,models))/((f_CO2(run,PT,melt_wf,setup,species,models)**2.0)*K)
        else:
            return 0.0


#######################
### oxygen fugacity ###
#######################

# buffers
def NNO(PT):
    P = PT['P']
    T_K = PT["T"]+273.15
    return (-24930/T_K + 9.36 + 0.046*(P-1.0)/T_K) # Frost (1991) RiMG 25:1–9
def FMQ(PT):
    P = PT['P']
    T_K = PT["T"]+273.15
    return (-25096.3/T_K + 8.735 + 0.11*(P-1.0)/T_K) # Frost (1991) RiMG 25:1–9
def fO22Dbuffer(PT,fO2,buffer):
    if buffer == "NNO":
        return math.log10(fO2) - NNO(PT)
    elif buffer == "FMQ":
        return math.log10(fO2) - FMQ(PT)
def Dbuffer2fO2(PT,D,buffer):
    if buffer == "NNO":
        return 10.0**(D + NNO(PT))
    elif buffer == "FMQ":
        return 10.0**(D + FMQ(PT))
    
### Compositional parameter from Kress & Carmichael (1991) CMP 108:82–92 Appensix A ###
def KC_mf(run,setup,species): # requires mole frations in the melt based on oxide components (all Fe as FeO) with no volatiles    
    tot = (setup.loc[run,"SiO2"]/species.loc["SiO2","M"]) + (setup.loc[run,"TiO2"]/species.loc["TiO2","M"]) + (setup.loc[run,"Al2O3"]/species.loc["Al2O3","M"]) + (Wm_FeOT(run,setup,species)/species.loc["FeO","M"]) + (setup.loc[run,"MnO"]/species.loc["MnO","M"]) + (setup.loc[run,"MgO"]/species.loc["MgO","M"]) + (setup.loc[run,"CaO"]/species.loc["CaO","M"]) + (setup.loc[run,"Na2O"]/species.loc["Na2O","M"]) + (setup.loc[run,"K2O"]/species.loc["K2O","M"])  + (setup.loc[run,"P2O5"]/species.loc["P2O5","M"])
    Al = (setup.loc[run,"Al2O3"]/species.loc["Al2O3","M"])/tot
    Fe = (Wm_FeOT(run,setup,species)/species.loc["FeO","M"])/tot
    Ca = (setup.loc[run,"CaO"]/species.loc["CaO","M"])/tot
    Na = (setup.loc[run,"Na2O"]/species.loc["Na2O","M"])/tot
    K = (setup.loc[run,"K2O"]/species.loc["K2O","M"])/tot
    return Al, Fe, Ca, Na, K

def d4X_KC(run,setup,species,models):
    Al, Fe, Ca, Na, K = KC_mf(run,setup,species)
    DAl = -2.243
    DFe = -1.828
    DCa = 3.201
    DNa = 5.854
    DK = 6.215
    return DAl*Al + DFe*Fe + DCa*Ca + DNa*Na + DK*K

def d4X_KCA(run,setup,species,models):
    Al, Fe, Ca, Na, K = KC_mf(run,setup,species)
    DWAl = 39.86e3             #J
    DWCa = -62.52e3            #J
    DWNa = -102.0e3            #J
    DWK = -119.0e3             #J
    return DWAl*Al+DWCa*Ca+DWNa*Na+DWK*K

def PT_KCterm(run,PT,setup,species,models):
    P = PT['P']
    T_K = PT['T']+273.15
    b = 1.1492e4 # K
    c = -6.675
    D4X = d4X_KC(run,setup,species,models)
    e = -3.36
    f = -7.01e-7 # K/Pa
    g = -1.54e-10 # /Pa
    h = 3.85e-17 # K/Pa2
    T0 = 1673.0 # K
    P_Pa = P*1.0e5 # converts bars to pascals
    value = (b/T_K) + c + D4X + e*(1.0 - (T0/T_K) - math.log(T_K/T0)) + f*(P_Pa/T_K) + g*(((T_K-T0)*P_Pa)/T_K) + h*((P_Pa**2.0)/T_K)
    return value

def KC91(run,PT,melt_wf,setup,species,models):    
    F = 0.5*Fe3Fe2(melt_wf) # XFe2O3/XFeO
    a = 0.196
    PTterm = PT_KCterm(run,PT,setup,species,models)
    alnfO2 = math.log(F) - PTterm
    return math.exp(alnfO2/a)

def KD1(run,PT,setup,species,models): #K&C91 appendix A 
    T_K = PT['T']+273.15
    P = PT['P']
    DH = -106.2e3               #J
    DS = -55.10                 #J/K
    DCp = 31.86                 #J/K
    DV = 7.42e-6                #m3
    DVdot = 1.63e-9             #m3/K
    DVdash = -8.16e-16          #m3/Pa
    D4X = d4X_KCA(run,setup,species,models)
    T0 = 1673.0                 # K
    P0 = 1.0e5                  # Pa 
    R = 8.3144598               # J/K/mol
    P_Pa = P*1.0e5
    return math.exp((-DH/(R*T_K)) + (DS/R) - (DCp/R)*(1.0 - (T0/T_K) - gp.log(T_K/T0)) - (1.0/(R*T_K))*D4X - ((DV*(P_Pa-P0))/(R*T_K)) - ((DVdot*(T_K-T0)*(P_Pa-P0))/(R*T_K)) - (DVdash/(2.0*R*T_K))*pow((P_Pa-P0),2.0))

def f_O2(run,PT,melt_wf,setup,species,models): # Kress91 is equation/table 7 and Kress91A is equations A5-6 and table A1
    model = models.loc["fO2","option"]
    
    if model == "yes":
        return 10.0**(setup.loc[run,"logfO2"]) 

    elif model == "Kress91":
        fO2 = KC91(run,PT,melt_wf,setup,species,models)
        return fO2
    
    elif model == "Kress91A": 
        F = Fe3Fe2(melt_wf) # XFeO1.5/XFeO
        D4X = d4X_KCA(run,setup,species,models)
        KD2 = 0.4
        y = 0.3
        kd1 = KD1(run,PT,setup,species,models)
            
        def f(y,F,KD2,kd1,x): # KC91A rearranged to equal 0
            f = ((2.0*y - F + 2.0*y*F)*KD2*kd1**(2.0*y)*x**(0.5*y) + kd1*x**0.25 - F)
            return f

        def df(y,F,KD2,kd1,x): # derivative of above
            df = (0.5*y)*(2.0*y - F +2.0*y*F)*KD2*kd1**(2.0*y)*x**((0.5*y)-1.0) + 0.25*kd1*x**-0.75
            return df

        def dx(x):
            diff = abs(0-f(y,F,KD2,kd1,x))
            return diff
 
        def nr(x0, e1):
            delta1 = dx(x0)
            while delta1 > e1:
                x0 = x0 - f(y,F,KD2,kd1,x0)/df(y,F,KD2,kd1,x0)
                delta1 = dx(x0)
            return x0
            
        x0 = KC91(run,PT,melt_wf,setup,species,models)
    
        fO2 = nr(x0, 1e-15)
        return fO2
        
    elif model == "ONeill18": # O'Neill et al. (2018) EPSL 504:152-162
        F = Fe3Fe2(melt_wf) # Fe3+/Fe2+
        # mole fractions on a single cation basis
        tot = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"]) + ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"]) + ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"]) + ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"]) + ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"]) + ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"]) + ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"]) + ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"]) + ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"]) + ((species.loc["P2O5","no_cat"]*setup.loc[run,"P2O5"])/species.loc["P2O5","M"])
        Ca = ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"])/tot
        Na = ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"])/tot
        K = ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"])/tot
        P = ((species.loc["P2O5","no_cat"]*setup.loc[run,"P2O5"])/species.loc["P2O5","M"])/tot
        DQFM = (math.log(F) + 1.36 - 2.4*Ca - 2.0*Na - 3.7*K + 2.4*P)/0.25
        logfO2 = DQFM + (8.58 - (25050/T_K)) # O'Neill (1987)
        return 10.0**logfO2
    
    elif model == "S6ST":
        S6T = melt_wf['S6ST']
        S62 = overtotal2ratio(S6T)
        fO2 = ((S62*C_S(run,PT,melt_wf,setup,species,models))/C_SO4(run,PT,melt_wf,setup,species,models))**0.5
        return fO2
    
def S6S2_2_fO2(S62,melt_wf,run,PT,setup,species,models):
    fO2 = ((S62*C_S(run,PT,melt_wf,setup,species,models))/C_SO4(run,PT,melt_wf,setup,species,models))**0.5
    return fO2
    
    
########################        
### partial pressure ###
########################

def p_H2(run,PT,melt_wf,setup,species,models):
    return f_H2(run,PT,melt_wf,setup,species,models)/y_H2(PT,models)
def p_H2O(run,PT,melt_wf,setup,species,models):
    return f_H2O(run,PT,melt_wf,setup,species,models)/y_H2O(PT,models)
def p_O2(run,PT,melt_wf,setup,species,models):
    return f_O2(run,PT,melt_wf,setup,species,models)/y_O2(PT,species,models)
def p_SO2(run,PT,melt_wf,setup,species,models):
    return f_SO2(run,PT,melt_wf,setup,species,models)/y_SO2(PT,species,models)
def p_S2(run,PT,melt_wf,setup,species,models):
    return f_S2(run,PT,melt_wf,setup,species,models)/y_S2(PT,species,models)
def p_H2S(run,PT,melt_wf,setup,species,models):
    return f_H2S(run,PT,melt_wf,setup,species,models)/y_H2S(PT,species,models)
def p_CO2(run,PT,melt_wf,setup,species,models):
    return f_CO2(run,PT,melt_wf,setup,species,models)/y_CO2(PT,species,models)
def p_CO(run,PT,melt_wf,setup,species,models):
    return f_CO(run,PT,melt_wf,setup,species,models)/y_CO(PT,species,models)
def p_CH4(run,PT,melt_wf,setup,species,models):
    return f_CH4(run,PT,melt_wf,setup,species,models)/y_CH4(PT,species,models)
def p_OCS(run,PT,melt_wf,setup,species,models):
    return f_OCS(run,PT,melt_wf,setup,species,models)/y_OCS(PT,species,models)

def p_tot(run,PT,melt_wf,setup,species,models):
    return p_H2(run,PT,melt_wf,setup,species,models) + p_H2O(run,PT,melt_wf,setup,species,models) + p_O2(run,PT,melt_wf,setup,species,models) + p_SO2(run,PT,melt_wf,setup,species,models) + p_S2(run,PT,melt_wf,setup,species,models) + p_H2S(run,PT,melt_wf,setup,species,models) + p_CO2(run,PT,melt_wf,setup,species,models) + p_CO(run,PT,melt_wf,setup,species,models) + p_CH4(run,PT,melt_wf,setup,species,models) + p_OCS(run,PT,melt_wf,setup,species,models)


############################       
### vapor molar fraction ###
############################

def xg_H2(run,PT,melt_wf,setup,species,models):
    P = PT['P']
    return p_H2(run,PT,melt_wf,setup,species,models)/P
def xg_H2O(run,PT,melt_wf,setup,species,models):
    P = PT['P']
    return p_H2O(run,PT,melt_wf,setup,species,models)/P
def xg_O2(run,PT,melt_wf,setup,species,models):
    P = PT['P']
    return p_O2(run,PT,melt_wf,setup,species,models)/P
def xg_SO2(run,PT,melt_wf,setup,species,models):
    P = PT['P']
    return p_SO2(run,PT,melt_wf,setup,species,models)/P
def xg_S2(run,PT,melt_wf,setup,species,models):
    P = PT['P']
    return p_S2(run,PT,melt_wf,setup,species,models)/P
def xg_H2S(run,PT,melt_wf,setup,species,models):
    P = PT['P']
    return p_H2S(run,PT,melt_wf,setup,species,models)/P
def xg_CO2(run,PT,melt_wf,setup,species,models):
    P = PT['P']
    return p_CO2(run,PT,melt_wf,setup,species,models)/P
def xg_CO(run,PT,melt_wf,setup,species,models):
    P = PT['P']
    return p_CO(run,PT,melt_wf,setup,species,models)/P
def xg_CH4(run,PT,melt_wf,setup,species,models):
    P = PT['P']
    return p_CH4(run,PT,melt_wf,setup,species,models)/P
def xg_OCS(run,PT,melt_wf,setup,species,models):
    P = PT['P']
    return p_OCS(run,PT,melt_wf,setup,species,models)/P
def Xg_tot(run,PT,melt_wf,setup,species,models):
    P = PT['P']
    Xg_t = xg_CO2(run,PT,melt_wf,setup,species,models)*species.loc["CO2","M"] + xg_CO(run,PT,melt_wf,setup,species,models)*species.loc["CO","M"] + xg_O2(run,PT,melt_wf,setup,species,models)*species.loc["O2","M"] + xg_H2O(run,PT,melt_wf,setup,species,models)*species.loc["H2O","M"] + xg_H2(run,PT,melt_wf,setup,species,models)*species.loc["H2","M"] + xg_CH4(run,PT,melt_wf,setup,species,models)*species.loc["CH4","M"] + xg_SO2(run,PT,melt_wf,setup,species,models)*species.loc["SO2","M"] + xg_S2(run,PT,melt_wf,setup,species,models)*species.loc["S2","M"] + xg_H2S(run,PT,melt_wf,setup,species,models)*species.loc["H2S","M"] + xg_OCS(run,PT,melt_wf,setup,species,models)*species.loc["OCS","M"]
    return Xg_t


##########################
### melt mole fraction ###
##########################

# totals
def wm_vol(melt_wf): # wt% total volatiles in the melt
    wm_H2OT = 100.*melt_wf["H2OT"]
    wm_CO2 = 100.*melt_wf["CO2"]
    return wm_H2OT + wm_CO2 #+ wm_S(wm_ST) + wm_SO3(wm_ST,species)
def wm_nvol(melt_wf): # wt% total of non-volatiles in the melt
    return 100.0 - wm_vol(melt_wf)

# molecular mass on a singular oxygen basis
def M_m_SO(run,setup,species): # no volatiles
    Wm_tot = setup.loc[run,"SiO2"] + setup.loc[run,"TiO2"] + setup.loc[run,"Al2O3"] + setup.loc[run,"MnO"] + setup.loc[run,"MgO"] + setup.loc[run,"MnO"] + setup.loc[run,"CaO"] + setup.loc[run,"Na2O"] + setup.loc[run,"K2O"] + setup.loc[run,"P2O5"] + Wm_FeOT(run,setup,species)
    Xm_tot = (setup.loc[run,"SiO2"]/(species.loc["SiO2","M"]/species.loc["SiO2","no_O"])) + (setup.loc[run,"TiO2"]/(species.loc["TiO2","M"]/species.loc["TiO2","no_O"])) + (setup.loc[run,"Al2O3"]/(species.loc["Al2O3","M"]/species.loc["Al2O3","no_O"])) + (setup.loc[run,"MnO"]/(species.loc["MnO","M"]/species.loc["MnO","no_O"])) + (setup.loc[run,"MgO"]/(species.loc["MgO","M"]/species.loc["MgO","no_O"])) + (setup.loc[run,"CaO"]/(species.loc["CaO","M"]/species.loc["CaO","no_O"])) + (setup.loc[run,"Na2O"]/(species.loc["Na2O","M"]/species.loc["Na2O","no_O"])) + (setup.loc[run,"K2O"]/(species.loc["K2O","M"]/species.loc["K2O","no_O"])) + (setup.loc[run,"P2O5"]/(species.loc["P2O5","M"]/species.loc["P2O5","no_O"])) + (Wm_FeOT(run,setup,species)/(species.loc["FeO","M"]/species.loc["FeO","no_O"]))   
    result = Wm_tot/Xm_tot
    return result

# molecular mass on a oxide basis
def M_m_ox(run,setup,species): # no volatiles
    Wm_tot = setup.loc[run,"SiO2"] + setup.loc[run,"TiO2"] + setup.loc[run,"Al2O3"] + setup.loc[run,"MnO"] + setup.loc[run,"MgO"] + setup.loc[run,"MnO"] + setup.loc[run,"CaO"] + setup.loc[run,"Na2O"] + setup.loc[run,"K2O"] + setup.loc[run,"P2O5"] + Wm_FeOT(run,setup,species)
    Xm_tot = (setup.loc[run,"SiO2"]/species.loc["SiO2","M"]) + (setup.loc[run,"TiO2"]/species.loc["TiO2","M"]) + (setup.loc[run,"Al2O3"]/species.loc["Al2O3","M"]) + (setup.loc[run,"MnO"]/species.loc["MnO","M"]) + (setup.loc[run,"MgO"]/species.loc["MgO","M"]) + (setup.loc[run,"CaO"]/species.loc["CaO","M"]) + (setup.loc[run,"Na2O"]/species.loc["Na2O","M"]) + (setup.loc[run,"K2O"]/species.loc["K2O","M"]) + (setup.loc[run,"P2O5"]/species.loc["P2O5","M"]) + (Wm_FeOT(run,setup,species)/species.loc["FeO","M"])   
    result = Wm_tot/Xm_tot
    return result
    
# Number of moles in the melt
def Xm_H2OT(melt_wf,species):
    wm_H2OT = 100.*melt_wf['H2OT']
    return wm_H2OT/species.loc["H2O","M"]
def Xm_CO2(melt_wf,species):
    wm_CO2 = 100.*melt_wf['CO2']
    return wm_CO2/species.loc["CO2","M"]
def Xm_ST(melt_wf,species):
    wm_ST = 100.*melt_wf['ST']
    return wm_ST/species.loc["S","M"]
def Xm_S(melt_wf,species):
    return wm_S(melt_wf)/species.loc["S","M"]
def Xm_SO3(melt_wf,species):
    return wm_SO3(melt_wf,species)/species.loc["SO3","M"]
def Xm_H2(melt_wf,species):
    wm_H2 = 100.*melt_wf['H2']
    return wm_H2/species.loc["H2","M"]

# Mole fraction in the melt based on mixing between volatile-free melt on a singular oxygen basis and volatiles
def Xm_m_so(run,melt_wf,setup,species): # singular oxygen basis
    return wm_nvol(melt_wf)/M_m_SO(run,setup,species)    
def Xm_tot_so(run,melt_wf,setup,species):
    return Xm_H2OT(melt_wf,species) + Xm_CO2(melt_wf,species) + Xm_m_so(run,melt_wf,setup,species) #+ Xm_S(wm_ST,species) + 
def xm_H2OT_so(run,melt_wf,setup,species):
    return Xm_H2OT(melt_wf,species)/Xm_tot_so(run,melt_wf,setup,species)
def xm_CO2_so(run,melt_wf,setup,species):
    return Xm_CO2(melt_wf,species)/Xm_tot_so(run,melt_wf,setup,species)
def xm_ST_so(run,melt_wf,setup,species):
    return Xm_ST(melt_wf,species)/Xm_tot_so(run,melt_wf,setup,species)
def xm_S_so(run,melt_wf,setup,species):
    return Xm_S(melt_wf,species)/Xm_tot_so(run,melt_wf,setup,species)
def xm_SO3_so(run,melt_wf,setup,species):
    return Xm_SO3(melt_wf,species)/Xm_tot_so(run,melt_wf,setup,species)
def xm_melt_so(run,melt_wf,setup,species):
    return Xm_m_so(run,melt_wf,setup,species)/Xm_tot_so(run,melt_wf,setup,species)
def Xm_t_so(run,melt_wf,setup,species):
    return xm_H2OT_so(run,melt_wf,setup,species)*species.loc["H2O","M"] + xm_CO2_so(run,melt_wf,setup,species)*species.loc["CO2","M"] + xm_melt_so(run,melt_wf,setup,species)*M_m_SO(run,setup,species)

# Mole fraction in the melt based on mixing between volatile-free melt on an oxide basis, H2O, and H2
def Xm_m_ox(run,melt_wf,setup,species): # singular oxygen basis
    return wm_nvol(melt_wf)/M_m_ox(run,setup,species)    
def Xm_tot_ox(run,melt_wf,setup,species):
    if melt_wf['H2'] > 0.:
        result = Xm_H2OT(melt_wf,species) + Xm_H2(melt_wf,species) + Xm_m_ox(run,melt_wf,setup,species) 
    else:
        result = Xm_H2OT(melt_wf,species) + Xm_m_ox(run,melt_wf,setup,species) 
    return result
def xm_H2OT_ox(run,melt_wf,setup,species):
    return Xm_H2OT(melt_wf,species)/Xm_tot_ox(run,melt_wf,setup,species)
def xm_H2_ox(run,melt_wf,setup,species):
    return Xm_H2(melt_wf,species)/Xm_tot_ox(run,melt_wf,setup,species)
def xm_melt_ox(run,melt_wf,setup,species):
    return Xm_m_ox(run,melt_wf,setup,species)/Xm_tot_ox(run,melt_wf,setup,species)
def Xm_t_ox(run,melt_wf,setup,species):
    if melt_wf["H2"] > 0.:
        result = xm_H2OT_ox(run,melt_wf,setup,species)*species.loc["H2O","M"] + xm_H2_ox(run,melt_wf,setup,species)*species.loc["H2","M"] + xm_melt_ox(run,melt_wf,setup,species)*M_m_ox(run,setup,species)
    else: result = xm_H2OT_ox(run,melt_wf,setup,species)*species.loc["H2O","M"] + xm_melt_ox(run,melt_wf,setup,species)*M_m_ox(run,setup,species)
    return result



################################################################################################################################
########################################################## SPECIATION ##########################################################
################################################################################################################################

def ratio2overtotal(x):
    return x/x+1.

def overtotal2ratio(x):
    return x/(1.-x)

###########################
### hydrogen speciation ###
###########################

def xm_OH_so(run,PT,melt_wf,setup,species,models):
    Hspeciation = models.loc["Hspeciation","option"]
    Hspeccomp = models.loc["Hspeccomp","option"]
    T_K = PT['T']+273.15
    wm_H2OT = 100.*melt_wf['H2OT']
    
    Z = xm_H2OT_so(run,melt_wf,setup,species)
    
    def f(A, B, C, Z, x): # regular solution model rearranged to equal 0 to solve for xm_OH
        return (A + B*x + C*(Z - 0.5*x) + gp.log((x**2.0)/((Z - 0.5*x)*(1.0 - Z - 0.5*x))))

    def df(B, C, Z, x): # derivative of above
        return (B - 0.5*C - (2.0/x) + (0.5/(Z-0.5*x) + (0.5/(1.0-Z-0.5*x))))

    def dx(A,B,C,x):
        return (abs(0-f(A, B, C, Z, x)))
 
    def nr(A,B,C,x0,e1):
        delta1 = dx(A,B,C,x0)
        while delta1 > e1:
            x0 = x0 - f(A, B, C, Z, x0)/df(B, C, Z, x0)
            delta1 = dx(A,B,C,x0)
        return x0
    
    if Z == 0.0: 
        return 0.0
    elif Hspeciation == "ideal": # ideal mixture from Silver & Stolper (1985) J. Geol. 93(2) 161-177
        K = KHOm(PT,models)
        return (0.5 - (0.25 - ((K - 4.0)/K)*(Z - Z**2.0))**0.5)/((K - 4.0)/(2.0*K)) 
    elif Hspeciation == "none": # all OH-
        return 0.0
    
    elif Hspeciation == "regular": # else use regular solution model from Silver & Stolper (1989) J.Pet. 30(3)667-709
        tol = 0.000001 #tolerance
        K = KHOm(PT,models)
        x0 = (0.5 - (0.25 - ((K - 4.0)/K)*(Z - Z**2.0))**0.5)/((K - 4.0)/(2.0*K))
        R = 83.15 #cm3.bar/mol.K
        if Hspeccomp == "MORB": # Dixon et al. (1995)
            A = 0.403
            B = 15.333
            C = 10.894
        elif Hspeccomp == "alkali basalt XT": # No T-dependence, hence its the speciation frozen in the glass. Eqn 7-10 from Lesne et al. (2010) CMP 162:133-151 (eqn 7 is wrong)
            Wm_tot = setup.loc[run,"SiO2"] + setup.loc[run,"TiO2"] + setup.loc[run,"Al2O3"] + Wm_FeOT(run,melt_wf,setup,species) + setup.loc[run,"MnO"] + setup.loc[run,"MgO"] + setup.loc[run,"MnO"] + setup.loc[run,"CaO"] + setup.loc[run,"Na2O"] + setup.loc[run,"K2O"]
            tot = 100.0 - wm_H2OT
            Na = (setup.loc[run,"Na2O"]/Wm_tot)*tot # Na2O wt% inc. H2O in total
            K = (setup.loc[run,"K2O"]/Wm_tot)*tot # K2O wt% inc. H2O in total
            A = 0.5761*(Na+K) - 0.2884 # eqn-8
            B = -8.9589*(Na+K) + 24.65 # eqn-9
            C = 1.7013*(Na+K) + 9.6481 # eqn-1
        elif Hspeccomp == "alkali basalt": # Includes T-dependence, hence speciation in the melt. Eqn 24-27 from Lesne et al. (2010) CMP 162:133-151
            lnK = ((-2704.4/T_K) + 0.641)
            A = lnK + 49016.0/(R*T_K)
            B = -2153326.51/(R*T_K)
            C = 1.965495217/(R*T_K)
        elif Hspeccomp == "VES-9": # Lesne et al. (2011) 162:133-151 Table 5
            A = 3.139
            B = -29.555
            C = 20.535
        elif Hspeccomp == "ETN-1": # Lesne et al. (2011) 162:133-151 Table 5
            A = 4.128
            B = -45.905
            C = 21.311
        elif Hspeccomp == "PST-9": # Lesne et al. (2011) 162:133-151 Table 5
            A = 2.6
            B = -22.476
            C = 22.295
        elif Hspeccomp == "albite": # Silver & Stolper (1989) J.Pet 30(3)667-709
            A = 0.403
            B = 15.333
            C = 10.894
        return nr(A,B,C,x0,tol)

def xm_H2Omol_so(run,PT,melt_wf,setup,species,models):
    Z = xm_H2OT_so(run,melt_wf,setup,species)
    return Z - 0.5*xm_OH_so(run,PT,melt_wf,setup,species,models)

def wm_H2Omol_OH(run,PT,melt_wf,setup,species,models):
    H2Omol = xm_H2Omol_so(run,PT,melt_wf,setup,species,models)*species.loc["H2O","M"]
    OH = xm_OH_so(run,PT,melt_wf,setup,species,models)*species.loc["OH","M"]
    melt = xm_melt_so(run,melt_wf,setup,species)*M_m_SO(run,setup,species)
    wm_H2Omol = 100.*H2Omol/(H2Omol + OH + melt)
    wm_OH = 100.*OH/(H2Omol + OH + melt)
    return wm_H2Omol, wm_OH # wt% 


##########################
### sulphur speciation ###
##########################

def S6S2(run,PT,melt_wf,setup,species,models):
    T_K = PT['T']+273.15
    model = models.loc["sulphate","option"]
    if model == "Nash19":
        return pow(10.0,(8.0*math.log10(Fe3Fe2(melt_wf)) + ((8.7436e6)/pow(T_K,2.0)) - (27703.0/T_K) + 20.273))
    else:
        return (C_SO4(run,PT,melt_wf,setup,species,models)/C_S(run,PT,melt_wf,setup,species,models))*pow(f_O2(run,PT,melt_wf,setup,species,models),2.0)
    
def S6ST(run,PT,melt_wf,setup,species,models):
    S6S2_ = S6S2(run,PT,melt_wf,setup,species,models)
    return S6S2_/(S6S2_+1.0)

def wm_S(run,PT,melt_wf,setup,species,models):
    wm_ST = 100.*melt_wf['ST']
    S6ST_ = S6ST(run,PT,melt_wf,setup,species,models)
    return wm_ST*(1.0-S6ST_)

def wm_SO3(run,PT,melt_wf,setup,species,models):
    wm_ST = 100.*melt_wf['ST']
    S6ST_ = S6ST(run,PT,melt_wf,setup,species,models)    
    return ((wm_ST*S6ST_)/species.loc["S","M"])*species.loc["SO3","M"]



#######################
### iron speciation ###
#######################

def Fe3Fe2(melt_wf):
    Fe3FeT = melt_wf['Fe3FeT']
    return Fe3FeT/(1.0 - Fe3FeT)

def Wm_FeT(run,setup,species):
    if setup.loc[run,"FeOT"] > 0.0:
        return (setup.loc[run,"FeOT"]/species.loc["FeO","M"])*species.loc["Fe","M"]
    elif setup.loc[run,"Fe2O3T"] > 0.0:
        return (setup.loc[run,"Fe2O3T"]/species.loc["Fe2O3","M"])*species.loc["Fe","M"]
    else:
        return ((setup.loc[run,"FeO"]/species.loc["FeO","M"]) + (setup.loc[run,"Fe2O3"]/species.loc["Fe2O3","M"]))*species.loc["Fe","M"]

def Wm_FeO(run,melt_wf,setup,species):
    Fe3FeT = melt_wf['Fe3FeT']
    return (Wm_FeT(run,setup,species)/species.loc["Fe","M"])*(1.0-Fe3FeT)*species.loc["FeO","M"]

def Wm_Fe2O3(run,melt_wf,setup,species):
    Fe3FeT = melt_wf['Fe3FeT']
    return (Wm_FeT(run,setup,species)/species.loc["Fe","M"])*Fe3FeT*species.loc["Fe2O3","M"]

def Wm_FeOT(run,setup,species):
    return (Wm_FeT(run,setup,species)/species.loc["Fe","M"])*species.loc["FeO","M"]

def wm_Fe_nv(run,melt_wf,setup,species): # no volatiles
    Wm_tot = setup.loc[run,"SiO2"] + setup.loc[run,"TiO2"] + setup.loc[run,"Al2O3"] + setup.loc[run,"MnO"] + setup.loc[run,"MgO"] + setup.loc[run,"MnO"] + setup.loc[run,"CaO"] + setup.loc[run,"Na2O"] + setup.loc[run,"K2O"] + setup.loc[run,"P2O5"] + Wm_FeO(run,melt_wf,setup,species) + Wm_Fe2O3(run,melt_wf,setup,species)
    FeT = species.loc["Fe","M"]*((2.0*Wm_Fe2O3(run,melt_wf,setup,species)/species.loc["Fe2O3","M"]) + (Wm_FeO(run,melt_wf,setup,species)/species.loc["FeO","M"]))
    return 100.0*FeT/Wm_tot

def Fe3FeT_i(run,PT,setup,species,models):
    model = models.loc["fO2","option"]
    T_K = PT['T']+273.15
    
    if model == "buffered":
        fO2 = 10**(setup.loc[run,"logfO2"])
        return fO22Fe3FeT(fO2,run,PT,setup,species,models)
    else:
        if pd.isnull(setup.loc[run,"Fe3FeT"]) == False:
            return setup.loc[run,"Fe3FeT"]
        elif pd.isnull(setup.loc[run,"logfO2"]) == False:
            fO2 = 10.0**(setup.loc[run,"logfO2"])
            return fO22Fe3FeT(fO2,run,PT,setup,species,models)
        elif pd.isnull(setup.loc[run,"DNNO"]) == False:
            D = setup.loc[run,"DNNO"]
            fO2 = Dbuffer2fO2(PT,D,"NNO")
            return fO22Fe3FeT(fO2,run,PT,setup,species,models)
        elif pd.isnull(setup.loc[run,"DFMQ"]) == False:
            D = setup.loc[run,"DFMQ"]
            fO2 = Dbuffer2fO2(PT,D,"FMQ")
            return fO22Fe3FeT(fO2,run,PT,setup,species,models)
        else:
            return ((2.0*setup.loc[run,"Fe2O3"])/species.loc["Fe2O3","M"])/(((2.0*setup.loc[run,"Fe2O3"])/species.loc["Fe2O3","M"]) + (setup.loc[run,"FeO"]/species.loc["FeO","M"]))
        
        
def fO22Fe3FeT(fO2,run,PT,setup,species,models):
    model = models.loc["fO2","option"]
    T_K = PT['T']+273.15
    
    if model == "Kress91":
        a = 0.196
        PTterm = PT_KCterm(PT)
        lnXFe2O3XFeO = a*gp.log(fO2) + PTterm
        XFe2O3XFeO = gp.exp(lnXFe2O3XFeO)
        return (2.0*XFe2O3XFeO)/((2.0*XFe2O3XFeO)+1.0)
    
    elif model == "Kress91A": 
        KD2 = 0.4
        y = 0.3
        kd1 = KD1(run,PT,setup,species,models)
        XFeO15XFeO = ((kd1*fO2**0.25)+(2.0*y*KD2*(kd1**(2.0*y))*(fO2**(0.5*y))))/(1.0 + (1.0 - 2.0*y)*KD2*(kd1**(2.0*y))*(fO2**(0.5*y)))
        return XFeO15XFeO/(XFeO15XFeO+1.0)  
    
    elif model == "ONeill18": # O'Neill et al. (2018) EPSL 504:152-162
        # mole fractions on a single cation basis
        tot = ((species.loc["SiO2","no_cat"]*setup.loc[run,"SiO2"])/species.loc["SiO2","M"]) + ((species.loc["TiO2","no_cat"]*setup.loc[run,"TiO2"])/species.loc["TiO2","M"]) + ((species.loc["Al2O3","no_cat"]*setup.loc[run,"Al2O3"])/species.loc["Al2O3","M"]) + ((species.loc["FeO","no_cat"]*Wm_FeOT(run,setup,species))/species.loc["FeO","M"]) + ((species.loc["MgO","no_cat"]*setup.loc[run,"MgO"])/species.loc["MgO","M"]) + ((species.loc["MnO","no_cat"]*setup.loc[run,"MnO"])/species.loc["MnO","M"]) + ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"]) + ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"]) + ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"]) + ((species.loc["P2O5","no_cat"]*setup.loc[run,"P2O5"])/species.loc["P2O5","M"])
        Ca = ((species.loc["CaO","no_cat"]*setup.loc[run,"CaO"])/species.loc["CaO","M"])/tot
        Na = ((species.loc["Na2O","no_cat"]*setup.loc[run,"Na2O"])/species.loc["Na2O","M"])/tot
        K = ((species.loc["K2O","no_cat"]*setup.loc[run,"K2O"])/species.loc["K2O","M"])/tot
        P = ((species.loc["P2O5","no_cat"]*setup.loc[run,"P2O5"])/species.loc["P2O5","M"])/tot
        DQFM = gp.log10(fO2) - (8.58 - (25050/T_K)) # O'Neill (1987)
        lnFe3Fe2 = 0.25*DQFM - 1.36 + 2.4*Ca + 2.0*Na + 3.7*K - 2.4*P
        Fe3Fe2 =  gp.exp(lnFe3Fe2)
        return Fe3Fe2/(Fe3Fe2 + 1.0)
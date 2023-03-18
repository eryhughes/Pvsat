# calculations.py

import pandas as pd
from datetime import date
import gmpy2 as gp
import numpy as np

import melt_gas as mg
import equilibrium_equations as eq


###########################
### saturation pressure ###
###########################

# for a given melt composition, calcualte the saturation pressure
def P_sat(run,PT,melt_wf,setup,species,models,Ptol,nr_step,nr_tol):
    ST = melt_wf["ST"]
    melt_wf1 = melt_wf # to work out P_sat
    melt_wf2 = melt_wf # to work out sulphur saturation
    
    def Pdiff(guess,run,melt_wf,setup,species,models):
        PT["P"] = guess
        difference = abs(guess - mg.p_tot(run,PT,melt_wf,setup,species,models))
        return difference

    guess0 = 40000. # initial guess for pressure
    PT["P"] = guess0
    melt_wf1["Fe3FeT"] = mg.Fe3FeT_i(run,PT,setup,species,models)
    melt_wf2["Fe3FeT"] = mg.Fe3FeT_i(run,PT,setup,species,models)
    xm_H2O_, wm_H2O_, xm_CO2_, wm_CO2_, wm_H2_, wm_CO_, wm_CH4_, wm_H2S_, wm_S2m_, wm_S6p_, H2O_HT, H2_HT, CH4_HT, CO2_CT, CO_CT, CH4_CT, S6p_ST, S2m_ST, H2S_ST, H2S_HT = eq.melt_speciation(run,PT,melt_wf1,setup,species,models,nr_step,nr_tol)
    melt_wf1["H2OT"] = wm_H2O_
    melt_wf2["H2OT"] = wm_H2O_
    melt_wf1["CO2"] = wm_CO2_
    melt_wf2["CO2"] = wm_CO2_
    melt_wf1["S2-"] = wm_S2m_
    melt_wf2["S2-"] = wm_S2m_
    melt_wf1["ST"] = ST
    melt_wf2["ST"] = ST
    if models.loc["sulphur_saturation","option"] == "yes": # must incorporate H2S concentration into S2- for SCSS
        SCSS_,sulphide_sat,SCAS_,sulphate_sat, ss_ST = sulphur_saturation(run,PT,melt_wf2,setup,species,models)
        melt_wf1["ST"] = ss_ST/1000000.
    delta1 = Pdiff(guess0,run,melt_wf1,setup,species,models)
    while delta1 > Ptol :
        delta1 = Pdiff(guess0,run,melt_wf1,setup,species,models)
        guess0 = mg.p_tot(run,PT,melt_wf1,setup,species,models)
        guess0 = float(guess0)
        PT["P"] = guess0
        melt_wf1["Fe3FeT"] = mg.Fe3FeT_i(run,PT,setup,species,models)
        melt_wf2["Fe3FeT"] = mg.Fe3FeT_i(run,PT,setup,species,models)
        xm_H2O_, wm_H2O_, xm_CO2_, wm_CO2_, wm_H2_, wm_CO_, wm_CH4_, wm_H2S_, wm_S2m_, wm_S6p_, H2O_HT, H2_HT, CH4_HT, CO2_CT, CO_CT, CH4_CT, S6p_ST, S2m_ST, H2S_ST, H2S_HT = eq.melt_speciation(run,PT,melt_wf1,setup,species,models,nr_step,nr_tol)            
        melt_wf1["H2OT"] = wm_H2O_
        melt_wf2["H2OT"] = wm_H2O_
        melt_wf1["CO2"] = wm_CO2_
        melt_wf2["CO2"] = wm_CO2_
        melt_wf1["S2-"] = wm_S2m_
        melt_wf2["S2-"] = wm_S2m_
        if models.loc["sulphur_saturation","option"] == "yes":
            SCSS_,sulphide_sat,SCAS_,sulphate_sat,ss_ST = sulphur_saturation(run,PT,melt_wf2,setup,species,models)
            melt_wf1["ST"] = ss_ST/1000000.
    else:
        P_sat = guess0
        xm_H2O_, wm_H2O_, xm_CO2_, wm_CO2_, wm_H2_, wm_CO_, wm_CH4_, wm_H2S_, wm_S2m_, wm_S6p_, H2O_HT, H2_HT, CH4_HT, CO2_CT, CO_CT, CH4_CT, S6p_ST, S2m_ST, H2S_ST, H2S_HT = eq.melt_speciation(run,PT,melt_wf1,setup,species,models,nr_step,nr_tol)
        
    melt_wf["ST"] = ST
    return P_sat, wm_H2O_, wm_CO2_, wm_H2_, wm_CO_, wm_CH4_, wm_S2m_, wm_S6p_, wm_H2S_, H2O_HT, H2_HT, CH4_HT, H2S_HT, CO2_CT, CO_CT, CH4_CT, S6p_ST, S2m_ST, H2S_ST

# for a given melt composition, calcualte the saturation pressure
def P_sat_H2O_CO2(run,PT,melt_wf,setup,species,models,Ptol,nr_step,nr_tol): # Pvsat with just H2O and CO2 in vapour
    
    def p_tot_H2O_CO2(run,PT,melt_wf,setup,species,models):
        value = mg.p_H2O(run,PT,melt_wf,setup,species,models) + mg.p_CO2(run,PT,melt_wf,setup,species,models)  
        return value
    
    def Pdiff(guess,run,melt_wf,setup,species,models):
        PT["P"] = guess
        difference = abs(guess - p_tot_H2O_CO2(run,PT,melt_wf,setup,species,models))
        return difference

    guess0 = 40000. # initial guess for pressure
    PT["P"] = guess0
    delta1 = Pdiff(guess0,run,melt_wf,setup,species,models)
    while delta1 > Ptol :
        delta1 = Pdiff(guess0,run,melt_wf,setup,species,models)
        guess0 = p_tot_H2O_CO2(run,PT,melt_wf,setup,species,models)
        guess0 = float(guess0)
        PT["P"] = guess0
    else:
        P_sat = guess0
        xg_H2O_ = mg.xg_H2O(run,PT,melt_wf,setup,species,models)
        xg_CO2_ = mg.xg_CO2(run,PT,melt_wf,setup,species,models)
        p_H2O_ = mg.p_H2O(run,PT,melt_wf,setup,species,models)
        p_CO2_ = mg.p_CO2(run,PT,melt_wf,setup,species,models)
        f_H2O_ = mg.f_H2O(run,PT,melt_wf,setup,species,models)
        f_CO2_ = mg.f_CO2(run,PT,melt_wf,setup,species,models)
    return P_sat, xg_H2O_, xg_CO2_, f_H2O_, f_CO2_, p_H2O_, p_CO2_

# calculate the saturation pressure for multiple melt compositions in input file
def P_sat_output(first_row,last_row,p_tol,nr_step,nr_tol,setup,species,models):
    # set up results table
    results = pd.DataFrame([["oxygen fugacity","carbonate solubility","C speciation composition","water solubility","water speciation","water speciation composition","sulphide solubility","sulphate solubility","sulphide saturation","ideal gas","carbonylsulphide","insolubles","Saturation calculation","Date"]])
    results1 = pd.DataFrame([[models.loc["fO2","option"],models.loc["carbonate","option"],models.loc["Cspeccomp","option"],models.loc["water","option"],models.loc["Hspeciation","option"],models.loc["Hspeccomp","option"],models.loc["sulphide","option"],models.loc["sulphate","option"],models.loc["sulphur_saturation","option"],models.loc["ideal_gas","option"],models.loc["carbonylsulphide","option"],models.loc['insolubles','option'],models.loc['calc_sat','option'],date.today()]])
    results = results.append(results1, ignore_index=True)
    results1 = ([["Sample","Saturation pressure (bar)","T ('C)","fO2 (DNNO)","fO2 (DFMQ)",
                  "SiO2 (wt%)","TiO2 (wt%)","Al2O3 (wt%)","FeOT (wt%)","MnO (wt%)","MgO (wt%)","CaO (wt%)","Na2O (wt%)","K2O (wt%)","P2O5 (wt%)",
                "H2OT (wt%)","CO2 (ppm)","ST (ppm)","Fe3/FeT","CT input (ppm)","HT input (ppm)","H2OT (wt%)","OH (wt%)","H2Omol (wt%)","H2 (ppm)","CH4 (ppm)","CO2 (ppm)","CO (ppm)","S2- (ppm)","S6+ (ppm)","H2S (ppm)","C reduced calc. (ppm)","H reduced calc. (ppm)",
                  "H_H2O/HT", "H_H2/HT", "H_CH4/HT", "H_H2S/HT", "C_CO2/CT", "C_CO/CT", "C_CH4/CT", "S2-/ST", "S6+/ST", "H2S/ST",
                "SCSS (ppm)","sulphide saturated","SCAS (ppm)","anhydrite saturated","S melt (ppm)","graphite saturated",
                "fO2","fH2","fH2O","fS2","fSO2","fH2S","fCO2","fCO","fCH4","fOCS",
                #"yO2","yH2","yH2O","yS2","ySO2","yH2S","yCO2","yCO","yCH4","yOCS",
                "pO2","pH2","pH2O","pS2","pSO2","pH2S","pCO2","pCO","pCH4","pOCS",
                "xgO2","xgH2","xgH2O","xgS2","xgSO2","xgH2S","xgCO2","xgCO","xgCH4","xgOCS",
                 "Pvsat (H2O CO2 only)", "xg_H2O (H2O CO2 only)", "xg_CO2 (H2O CO2 only)","f_H2O (H2O CO2 only)", "f_CO2 (H2O CO2 only)","p_H2O (H2O CO2 only)", "p_CO2 (H2O CO2 only)", "Pvsat diff"]])
    results = results.append(results1, ignore_index=True)

    for n in range(first_row,last_row,1): # n is number of rows of data in conditions file
        run = n
        PT={"T":setup.loc[run,"T_C"]}
        melt_wf = {'CO2':setup.loc[run,"CO2ppm"]/1000000.,"H2OT":setup.loc[run,"H2O"]/100.}
        
        # calculate Pvsat assuming only H2O CO2 in vapour and melt
        if melt_wf['CO2'] == 0 and melt_wf['H2OT'] == 0:
            P_sat_H2O_CO2_only, xg_H2O_H2O_CO2_only, xg_CO2_H2O_CO2_only, f_H2O_H2O_CO2_only, f_CO2_H2O_CO2_only, p_H2O_H2O_CO2_only, p_CO2_H2O_CO2_only = 0., 0., 0., 0., 0., 0., 0.
        else:
            if setup.loc[run,"Fe3FeT"] > 0.:
                melt_wf['Fe3FeT'] = setup.loc[run,"Fe3FeT"]
            else:
                melt_wf['Fe3FeT'] = 0.
            P_sat_H2O_CO2_only, xg_H2O_H2O_CO2_only, xg_CO2_H2O_CO2_only, f_H2O_H2O_CO2_only, f_CO2_H2O_CO2_only, p_H2O_H2O_CO2_only, p_CO2_H2O_CO2_only  = P_sat_H2O_CO2(run,PT,melt_wf,setup,species,models,p_tol,nr_step,nr_tol)
        
        if models.loc["calc_sat","option"] == "fO2_fX":
            P_sat_, wm_ST, fSO2, wm_S2m = P_sat_fO2_fS2(run,PT,melt_wf,setup,species,models,p_tol)
            PT["P"] = P_sat_
        else:
            wm_ST = setup.loc[run,"STppm"]/1000000.
        melt_wf['ST'] = wm_ST
        melt_wf['CT'] = (melt_wf['CO2']/species.loc['CO2','M'])*species.loc['C','M']
        melt_wf['HT'] = (melt_wf['H2OT']/species.loc['H2O','M'])*(2.*species.loc['H','M'])
        if setup.loc[run,"S6ST"] > 0.:
            melt_wf["S6ST"] = setup.loc[run,"S6ST"]
        bulk_wf = {"H":(2.*species.loc["H","M"]*melt_wf["H2OT"])/species.loc["H2O","M"],"C":(species.loc["C","M"]*melt_wf["CO2"])/species.loc["CO2","M"],"S":wm_ST}
        P_sat_, wm_H2O_, wm_CO2_, wm_H2_, wm_CO_, wm_CH4_, wm_S2m_, wm_S6p_, wm_H2S_, H2O_HT, H2_HT, CH4_HT, H2S_HT, CO2_CT, CO_CT, CH4_CT, S6p_ST, S2m_ST, H2S_ST = P_sat(run,PT,melt_wf,setup,species,models,p_tol,nr_step,nr_tol)
        PT["P"] = P_sat_
        melt_wf["H2OT"] = wm_H2O_
        melt_wf["CO2"] = wm_CO2_
        melt_wf["S2-"] = wm_S2m_
        melt_wf["Fe3FeT"] = mg.Fe3FeT_i(run,PT,setup,species,models)
        SCSS_,sulphide_sat,SCAS_,sulphate_sat, ss_ST = sulphur_saturation(run,PT,melt_wf,setup,species,models)
        graphite_sat = graphite_saturation(run,PT,melt_wf,setup,species,models)
        gas_mf = {"O2":mg.xg_O2(run,PT,melt_wf,setup,species,models),"CO":mg.xg_CO(run,PT,melt_wf,setup,species,models),"CO2":mg.xg_CO2(run,PT,melt_wf,setup,species,models),"H2":mg.xg_H2(run,PT,melt_wf,setup,species,models),"H2O":mg.xg_H2O(run,PT,melt_wf,setup,species,models),"CH4":mg.xg_CH4(run,PT,melt_wf,setup,species,models),"S2":mg.xg_S2(run,PT,melt_wf,setup,species,models),"SO2":mg.xg_SO2(run,PT,melt_wf,setup,species,models),"H2S":mg.xg_H2S(run,PT,melt_wf,setup,species,models),"OCS":mg.xg_OCS(run,PT,melt_wf,setup,species,models),"Xg_t":mg.Xg_tot(run,PT,melt_wf,setup,species,models),"wt_g":0.}

        # forward calculate H, C, and S in the melt from reduced species
        if models.loc['insolubles','option'] == 'no':
            wm_H2_, wm_CH4_, wm_CO_, wm_H2S_, wm_S6p_, H2O_HT, H2_HT, CH4_HT, H2S_HT, CO2_CT, CO_CT, CH4_CT, S2m_ST, S6p_ST, H2S_ST, C_T, H_T, S_T = conc_insolubles(run,PT,melt_wf,setup,species,models)
        ppmCreduced = (species.loc["C","M"]*((wm_CH4_/species.loc["CH4","M"])+ (wm_CO_/species.loc["CO","M"])))*1000000.
        ppmHreduced = (species.loc["H","M"]*(((4.*wm_CH4_)/species.loc["CH4","M"]) + ((2.*wm_H2_)/species.loc["H2","M"])))*1000000.
        wm_H2Omol, wm_OH = mg.wm_H2Omol_OH(run,PT,melt_wf,setup,species,models) # wt% - not giving correct answer atm
        
        ### store results ###
        results2 = pd.DataFrame([[setup.loc[run,"Sample"],PT["P"],
                setup.loc[run,"T_C"],mg.fO22Dbuffer(PT,mg.f_O2(run,PT,melt_wf,setup,species,models),"NNO"),mg.fO22Dbuffer(PT,mg.f_O2(run,PT,melt_wf,setup,species,models),"FMQ"),setup.loc[run,"SiO2"],setup.loc[run,"TiO2"],setup.loc[run,"Al2O3"],mg.Wm_FeOT(run,setup,species),setup.loc[run,"MnO"],setup.loc[run,"MgO"],setup.loc[run,"CaO"],setup.loc[run,"Na2O"],setup.loc[run,"K2O"],setup.loc[run,"P2O5"],
                setup.loc[run,"H2O"],setup.loc[run,"CO2ppm"],wm_ST*1000000.,melt_wf["Fe3FeT"],melt_wf['CT']*1000000.,melt_wf['HT']*1000000.,wm_H2O_*100.,
wm_OH,wm_H2Omol,wm_H2_*1000000.,wm_CH4_*1000000.,wm_CO2_*1000000.,wm_CO_*1000000.,melt_wf['S2-']*1000000.,wm_S6p_*1000000.,wm_H2S_*1000000.,ppmCreduced,ppmHreduced, 
                H2O_HT, H2_HT, CH4_HT, H2S_HT, CO2_CT, CO_CT, CH4_CT, S2m_ST, S6p_ST, H2S_ST,
                SCSS_,sulphide_sat,SCAS_,sulphate_sat,ss_ST,graphite_sat,
                mg.f_O2(run,PT,melt_wf,setup,species,models),mg.f_H2(run,PT,melt_wf,setup,species,models),mg.f_H2O(run,PT,melt_wf,setup,species,models),mg.f_S2(run,PT,melt_wf,setup,species,models),mg.f_SO2(run,PT,melt_wf,setup,species,models),mg.f_H2S(run,PT,melt_wf,setup,species,models),mg.f_CO2(run,PT,melt_wf,setup,species,models),mg.f_CO(run,PT,melt_wf,setup,species,models),mg.f_CH4(run,PT,melt_wf,setup,species,models),mg.f_OCS(run,PT,melt_wf,setup,species,models),
                #mg.y_O2(PT,models),mg.y_H2(PT,models),mg.y_H2O(PT,models),mg.y_S2(PT,models),mg.y_SO2(PT,models),mg.y_H2S(PT,models),mg.y_CO2(PT,models),mg.y_CO(PT,models),mg.y_CH4(PT,models),mg.y_OCS(PT,models),
                mg.p_O2(run,PT,melt_wf,setup,species,models),mg.p_H2(run,PT,melt_wf,setup,species,models),mg.p_H2O(run,PT,melt_wf,setup,species,models),mg.p_S2(run,PT,melt_wf,setup,species,models),mg.p_SO2(run,PT,melt_wf,setup,species,models),mg.p_H2S(run,PT,melt_wf,setup,species,models),mg.p_CO2(run,PT,melt_wf,setup,species,models),mg.p_CO(run,PT,melt_wf,setup,species,models),mg.p_CH4(run,PT,melt_wf,setup,species,models),mg.p_OCS(run,PT,melt_wf,setup,species,models),
                mg.xg_O2(run,PT,melt_wf,setup,species,models),mg.xg_H2(run,PT,melt_wf,setup,species,models),mg.xg_H2O(run,PT,melt_wf,setup,species,models),mg.xg_S2(run,PT,melt_wf,setup,species,models),mg.xg_SO2(run,PT,melt_wf,setup,species,models),mg.xg_H2S(run,PT,melt_wf,setup,species,models),mg.xg_CO2(run,PT,melt_wf,setup,species,models),mg.xg_CO(run,PT,melt_wf,setup,species,models),mg.xg_CH4(run,PT,melt_wf,setup,species,models),mg.xg_OCS(run,PT,melt_wf,setup,species,models),
                 P_sat_H2O_CO2_only, xg_H2O_H2O_CO2_only, xg_CO2_H2O_CO2_only,f_H2O_H2O_CO2_only, f_CO2_H2O_CO2_only,p_H2O_H2O_CO2_only, p_CO2_H2O_CO2_only,P_sat_H2O_CO2_only-PT["P"]]])
                             
        results = results.append(results2, ignore_index=True)
        results.to_csv('saturation_pressures.csv', index=False, header=False)
        print(n, setup.loc[run,"Sample"],PT["P"])

        
        
#########################
### sulphur satuation ###
#########################

# check solid/immiscible liquid sulphur saturation
def sulphur_saturation(run,PT,melt_wf,setup,species,models): # melt weight fraction of ST and S6/ST
    wmST = melt_wf['ST']
    S6T = mg.S6ST(run,PT,melt_wf,setup,species,models)
    wmS2 = wmST*100.0*10000.0*(1.0-S6T)
    wmS6 = wmST*100.0*10000.0*S6T
    SCSS_ = mg.SCSS(run,PT,melt_wf,setup,species,models)
    SCAS_ = mg.SCAS(run,PT,melt_wf,setup,species,models)
    StCSS = SCSS_/(1.-S6T)
    StCAS = SCAS_/S6T
    if wmS2 < SCSS_ and wmS6 < SCAS_:
        sulphide_sat = "no"
        sulphate_sat = "no"
        ST = wmST*1000000.
    elif wmS2 >= SCSS_ and wmS6 >= SCAS_:
        sulphide_sat = "yes"
        sulphate_sat = "yes"
        ST = min(StCSS,StCAS)
    elif wmS2 >= SCSS_ and wmS6 < SCAS_:
        sulphide_sat = "yes"
        sulphate_sat = "no"
        ST = StCSS
    elif wmS2 < SCSS_ and wmS6 >= SCAS_:
        sulphide_sat = "no"
        sulphate_sat = "yes"
        ST = StCAS
    else:
        sulphide_sat = "nan"
        sulphate_sat = "nan"
        ST = wmST*1000000.
    
    return SCSS_, sulphide_sat, SCAS_, sulphate_sat, ST

##########################
### graphite satuation ###
##########################

# check graphite saturation
def graphite_saturation(run,PT,melt_wf,setup,species,models): # needs finishing
    K1 = mg.f_CO2(run,PT,melt_wf,setup,species,models)/mg.f_O2(run,PT,melt_wf,setup,species,models)
    K2 = mg.KCOs(PT,models) # K for graphite saturation
    if K1 < K2:
        graphite_sat = "no"
    else:
        graphite_sat = "yes"
    fCO2_ = K2*mg.f_O2(run,PT,melt_wf,setup,species,models)
    xmCO2 = fCO2_*mg.C_CO3(run,PT,melt_wf,setup,species,models)
    return graphite_sat
                             

#########################
### Sulphate capacity ###
#########################

# calculate the Csulphate for multiple melt compositions in input file
def Csulphate_output(first_row,last_row,setup,species,models):
    # set up results table
    results = pd.DataFrame([["oxygen fugacity","carbonate solubility","C speciation composition","water solubility","water speciation","water speciation composition","sulphide solubility","sulphate solubility","sulphide saturation","ideal gas","carbonylsulphide","mass_volume","Date"]])
    results1 = pd.DataFrame([[models.loc["fO2","option"],models.loc["carbonate","option"],models.loc["Cspeccomp","option"],models.loc["water","option"],models.loc["Hspeciation","option"],models.loc["Hspeccomp","option"],models.loc["sulphide","option"],models.loc["sulphate","option"],models.loc["sulphur_saturation","option"],models.loc["ideal_gas","option"],models.loc["carbonylsulphide","option"],models.loc["mass_volume","option"],date.today()]])
    results = results.append(results1, ignore_index=True)
    results1 = ([["Sample","Pressure (bar)","T ('C)","SiO2","TiO2","Al2O3","FeOT","MnO","MgO","CaO","Na2O","K2O","P2O5",
                "H2O","CO2 (ppm)","ST (ppm)","S6/ST","Fe3/FeT","ln[Csulphide]","ln[Csulphate]","fO2","DNNO","DFMQ"]])
    results = results.append(results1, ignore_index=True)

    for n in range(first_row,last_row,1): # n is number of rows of data in conditions file
        run = n
        PT={"T":setup.loc[run,"T_C"]}
        melt_wf = {'CO2':setup.loc[run,"CO2ppm"]/1000000.,"H2OT":setup.loc[run,"H2O"]/100.,"ST":setup.loc[run,"STppm"]/1000000.}
        PT["P"] = setup.loc[run,"P_bar"]
        melt_wf["Fe3FeT"] = mg.Fe3FeT_i(run,PT,setup,species,models)
        if setup.loc[run,"S6ST"] >= 0.:
            melt_wf["S6ST"] = setup.loc[run,"S6ST"]
        else:
            melt_wf["S6ST"] = ""
        Csulphate_ = mg.C_SO4(run,PT,melt_wf,setup,species,models)
                
        ### store results ###
        results2 = pd.DataFrame([[setup.loc[run,"Sample"],setup.loc[run,"P_bar"],setup.loc[run,"T_C"],setup.loc[run,"SiO2"],setup.loc[run,"TiO2"],setup.loc[run,"Al2O3"],mg.Wm_FeOT(run,setup,species),setup.loc[run,"MnO"],setup.loc[run,"MgO"],setup.loc[run,"CaO"],setup.loc[run,"Na2O"],setup.loc[run,"K2O"],setup.loc[run,"P2O5"],setup.loc[run,"H2O"],setup.loc[run,"CO2ppm"],setup.loc[run,"STppm"],melt_wf["S6ST"],melt_wf["Fe3FeT"],gp.log(mg.C_S(run,PT,melt_wf,setup,species,models)),gp.log(Csulphate_),mg.f_O2(run,PT,melt_wf,setup,species,models),mg.fO22Dbuffer(PT,mg.f_O2(run,PT,melt_wf,setup,species,models),"NNO"),mg.fO22Dbuffer(PT,mg.f_O2(run,PT,melt_wf,setup,species,models),"FMQ")]])
                             
        results = results.append(results2, ignore_index=True)
        results.to_csv('Csulphate.csv', index=False, header=False)
        print(n, setup.loc[run,"Sample"],gp.log(Csulphate_),gp.log(mg.C_S(run,PT,melt_wf,setup,species,models)))

##################
### Capacities ###
##################

# print capacities for multiple melt compositions in input file
def capacities_output(first_row,last_row,setup,species,models):
    # set up results table
    results = pd.DataFrame([["oxygen fugacity","carbonate solubility","C speciation composition","water solubility","water speciation","water speciation composition","sulphide solubility","sulphate solubility","sulphide saturation","ideal gas","carbonylsulphide","mass_volume","Date"]])
    results1 = pd.DataFrame([[models.loc["fO2","option"],models.loc["carbonate","option"],models.loc["Cspeccomp","option"],models.loc["water","option"],models.loc["Hspeciation","option"],models.loc["Hspeccomp","option"],models.loc["sulphide","option"],models.loc["sulphate","option"],models.loc["sulphur_saturation","option"],models.loc["ideal_gas","option"],models.loc["carbonylsulphide","option"],models.loc["mass_volume","option"],date.today()]])
    results = results.append(results1, ignore_index=True)
    results1 = ([["Sample","Pressure (bar)","T ('C)","SiO2","TiO2","Al2O3","FeOT","MnO","MgO","CaO","Na2O","K2O","P2O5",
                "H2O","CO2 (ppm)","ST (ppm)","fO2 DFMQ","ln[C_CO32-]","ln[C_H2OT]","ln[C_S2-]","ln[C_S6+]","ln[C_H2S]","ln[C_H2]","ln[C_CO]","ln[C_CH4]"]])
    results = results.append(results1, ignore_index=True)

    for n in range(first_row,last_row,1): # n is number of rows of data in conditions file
        run = n
        PT={"T":setup.loc[run,"T_C"]}
        melt_wf = {'CO2':setup.loc[run,"CO2ppm"]/1000000.,"H2OT":setup.loc[run,"H2O"]/100.,"ST":setup.loc[run,"STppm"]/1000000.}
        PT["P"] = setup.loc[run,"P_bar"]
        melt_wf["Fe3FeT"] = mg.Fe3FeT_i(run,PT,setup,species,models)
        C_CO32 = mg.C_CO3(run,PT,melt_wf,setup,species,models)
        C_H2OT = mg.C_H2O(PT,models)
        C_S2 = mg.C_S(run,PT,melt_wf,setup,species,models)
        C_S6 = mg.C_SO4(run,PT,melt_wf,setup,species,models)
        C_H2S = mg.C_H2S(run,PT,melt_wf,setup,species,models)
        C_H2 = mg.C_H2(PT,models)
        C_CO = mg.C_CO(PT,models)
        C_CH4 = mg.C_CH4(PT,models)
        fO2_ = mg.fO22Dbuffer(PT,mg.f_O2(run,PT,melt_wf,setup,species,models),"FMQ")
                
        ### store results ###
        results2 = pd.DataFrame([[setup.loc[run,"Sample"],setup.loc[run,"P_bar"],setup.loc[run,"T_C"],setup.loc[run,"SiO2"],setup.loc[run,"TiO2"],setup.loc[run,"Al2O3"],mg.Wm_FeOT(run,setup,species),setup.loc[run,"MnO"],setup.loc[run,"MgO"],setup.loc[run,"CaO"],setup.loc[run,"Na2O"],setup.loc[run,"K2O"],setup.loc[run,"P2O5"],setup.loc[run,"H2O"],setup.loc[run,"CO2ppm"],setup.loc[run,"STppm"],melt_wf["Fe3FeT"],fO2_,gp.log(C_CO32),gp.log(C_H2OT),gp.log(C_S2),gp.log(C_S6),gp.log(C_H2S),gp.log(C_H2),gp.log(C_CO),gp.log(C_CH4)]])
                             
        results = results.append(results2, ignore_index=True)
        results.to_csv('capacities.csv', index=False, header=False)
        print(n, setup.loc[run,"Sample"],gp.log(C_CO32),gp.log(C_H2OT),gp.log(C_S2),gp.log(C_S6),gp.log(C_H2S),gp.log(C_H2),gp.log(C_CO),gp.log(C_CH4))

        
##########################
### Fe3+/Fe2+ from fO2 ###        
##########################

def Fe3Fe2_output(first_row,last_row,setup,species,models):
    # set up results table
    results = pd.DataFrame([["oxygen fugacity","carbonate solubility","C speciation composition","water solubility","water speciation","water speciation composition","sulphide solubility","sulphate solubility","sulphide saturation","ideal gas","carbonylsulphide","mass_volume","insolubles","Date"]])
    results1 = pd.DataFrame([[models.loc["fO2","option"],models.loc["carbonate","option"],models.loc["Cspeccomp","option"],models.loc["water","option"],models.loc["Hspeciation","option"],models.loc["Hspeccomp","option"],models.loc["sulphide","option"],models.loc["sulphate","option"],models.loc["sulphur_saturation","option"],models.loc["ideal_gas","option"],models.loc["carbonylsulphide","option"],models.loc["mass_volume","option"],models.loc['insolubles','option'],date.today()]])
    results = results.append(results1, ignore_index=True)
    results1 = ([["Sample","Pressure (bar)","T ('C)","fO2 (DNNO)","fO2 (DFMQ)",
                  "SiO2 (wt%)","TiO2 (wt%)","Al2O3 (wt%)","FeOT (wt%)","MnO (wt%)","MgO (wt%)","CaO (wt%)","Na2O (wt%)","K2O (wt%)","P2O5 (wt%)","Fe3/FeT"]])
    results = results.append(results1, ignore_index=True)

    for n in range(first_row,last_row,1): # n is number of rows of data in conditions file
        run = n
        PT={"T":setup.loc[run,"T_C"]}
        PT["P"] = setup.loc[run,"P_bar"]
        melt_wf={'Fe3FeT':mg.Fe3FeT_i(run,PT,setup,species,models)}
      ### store results ###
        results2 = pd.DataFrame([[setup.loc[run,"Sample"],PT["P"],
                setup.loc[run,"T_C"],mg.fO22Dbuffer(PT,mg.f_O2(run,PT,melt_wf,setup,species,models),"NNO"),mg.fO22Dbuffer(PT,mg.f_O2(run,PT,melt_wf,setup,species,models),"FMQ"),setup.loc[run,"SiO2"],setup.loc[run,"TiO2"],setup.loc[run,"Al2O3"],mg.Wm_FeOT(run,setup,species),setup.loc[run,"MnO"],setup.loc[run,"MgO"],setup.loc[run,"CaO"],setup.loc[run,"Na2O"],setup.loc[run,"K2O"],setup.loc[run,"P2O5"],melt_wf['Fe3FeT']]])
                             
        results = results.append(results2, ignore_index=True)
        results.to_csv('Fe3FeT_outputs.csv', index=False, header=False)
        print(n, setup.loc[run,"Sample"],melt_wf['Fe3FeT'])
        

#############################        
### fugacity coefficients ###
#############################

def fugacity_coefficients(first_row,last_row,setup,species,models):
    # set up results table
    results = pd.DataFrame([["oxygen fugacity","carbonate solubility","C speciation composition","water solubility","water speciation","water speciation composition","sulphide solubility","sulphate solubility","sulphide saturation","ideal gas","carbonylsulphide","mass_volume","insolubles","Date"]])
    results1 = pd.DataFrame([[models.loc["fO2","option"],models.loc["carbonate","option"],models.loc["Cspeccomp","option"],models.loc["water","option"],models.loc["Hspeciation","option"],models.loc["Hspeccomp","option"],models.loc["sulphide","option"],models.loc["sulphate","option"],models.loc["sulphur_saturation","option"],models.loc["ideal_gas","option"],models.loc["carbonylsulphide","option"],models.loc["mass_volume","option"],models.loc['insolubles','option'],date.today()]])
    results = results.append(results1, ignore_index=True)
    results1 = ([["Sample","Pressure (bar)","T ('C)","yH2O","yCO2","yH2","yCO","yO2","yCH4","yS2","ySO2","yH2S","yOCS"]])
    results = results.append(results1, ignore_index=True)

    for n in range(first_row,last_row,1): # n is number of rows of data in conditions file
        run = n
        PT={"T":setup.loc[run,"T_C"]}
        PT["P"] = setup.loc[run,"P_bar"]
        
        ### store results ###
        results2 = pd.DataFrame([[setup.loc[run,"Sample"],PT["P"],setup.loc[run,"T_C"],mg.y_H2O(PT,models),mg.y_CO2(PT,species,models),mg.y_H2(PT,models),mg.y_CO(PT,species,models),mg.y_O2(PT,species,models),mg.y_CH4(PT,species,models),mg.y_S2(PT,species,models),mg.y_SO2(PT,species,models),mg.y_H2S(PT,species,models),mg.y_OCS(PT,species,models)]])
                             
        results = results.append(results2, ignore_index=True)
        results.to_csv('fugacity_coefficients_outputs.csv', index=False, header=False)
        print(n, setup.loc[run,"Sample"],mg.y_H2O(PT,models))
                
         
###################################
### concentration of insolubles ### 
###################################

def conc_insolubles(run,PT,melt_wf,setup,species,models):
    CO2 = melt_wf["CO2"] # weight fraction CO2
    C_CO2_ = (species.loc["C","M"]*CO2)/species.loc["CO2","M"]
    H2O = melt_wf["H2OT"] # weight fraction H2O
    H_H2O = (2.*species.loc["H","M"]*H2O)/species.loc["H2O","M"]
    H2 = (mg.C_H2(PT,models)*mg.f_H2(run,PT,melt_wf,setup,species,models))/1000000. # weight fraction H2
    H_H2 = (2.*species.loc["H","M"]*H2)/species.loc["H2","M"]
    CH4 = (mg.C_CH4(PT,models)*mg.f_CH4(run,PT,melt_wf,setup,species,models))/1000000. # weight fraction CH4
    H_CH4 = (4.*species.loc["H","M"]*CH4)/species.loc["CH4","M"]
    C_CH4_ = (species.loc["C","M"]*CH4)/species.loc["CH4","M"]
    CO = (mg.C_CO(PT,models)*mg.f_CO(run,PT,melt_wf,setup,species,models))/1000000. # weight fraction CO
    C_CO_ = (species.loc["C","M"]*CO)/species.loc["CO","M"]
    S2m = melt_wf["S2-"] # weight fraction of S2-
    S6p = (mg.C_SO4(run,PT,melt_wf,setup,species,models)*mg.f_O2(run,PT,melt_wf,setup,species,models)**2*S2m)/mg.C_S(run,PT,melt_wf,setup,species,models) # weight fraction S6+
    H2S = (mg.C_H2S(run,PT,melt_wf,setup,species,models)*mg.f_H2S(run,PT,melt_wf,setup,species,models))/1000000. # weight fraction H2S
    S_H2S = (species.loc["S","M"]*H2S)/species.loc["H2S","M"]
    H_H2S = (2.*species.loc["H","M"]*H2S)/species.loc["H2S","M"]
    C_T = C_CO_ + C_CH4_ + C_CO2_
    H_T = H_H2O + H_H2 + H_CH4 + H_H2S
    S_T = S_H2S + S2m + S6p
    H2O_HT = H_H2O/H_T
    H2_HT = H_H2/H_T
    CH4_HT = H_CH4/H_T
    H2S_HT = H_H2S/H_T
    CO2_CT = C_CO2_/C_T
    CO_CT = C_CO_/C_T
    CH4_CT = C_CH4_/C_T
    S2m_ST = S2m/S_T
    S6p_ST = S6p/S_T
    H2S_ST = S_H2S/S_T
    return H2, CH4, CO, H2S, S6p, H2O_HT, H2_HT, CH4_HT, H2S_HT, CO2_CT, CO_CT, CH4_CT, S2m_ST, S6p_ST, H2S_ST, C_T, H_T, S_T

########################################
### measured parameters within error ### 
########################################

def compositions_within_error(setup,run,iterations):
    
    # set up results table
    results = pd.DataFrame([["Sample","T_C",
                  "SiO2","TiO2","Al2O3","FeOT","MnO","MgO","CaO","Na2O","K2O","P2O5",
                "H2OT","CO2ppm","STppm","Fe3FeT"]])
    results1 = pd.DataFrame([[setup.loc[run,"Sample"],setup.loc[run,"T_C"],setup.loc[run,"SiO2"],setup.loc[run,"TiO2"],setup.loc[run,"Al2O3"],setup.loc[run,"FeOT"],setup.loc[run,"MnO"],setup.loc[run,"MgO"],setup.loc[run,"CaO"],setup.loc[run,"Na2O"],setup.loc[run,"K2O"],setup.loc[run,"P2O5"],
                setup.loc[run,"H2O"],setup.loc[run,"CO2ppm"],setup.loc[run,"STppm"], setup.loc[run,"Fe3FeT"]]])
                             
    results = results.append(results1, ignore_index=True)
    results1 = pd.DataFrame([["sds","",setup.loc[run,"SiO2_sd"],setup.loc[run,"TiO2_sd"],setup.loc[run,"Al2O3_sd"],setup.loc[run,"FeOT_sd"],setup.loc[run,"MnO_sd"],setup.loc[run,"MgO_sd"],setup.loc[run,"CaO_sd"],setup.loc[run,"Na2O_sd"],setup.loc[run,"K2O_sd"],setup.loc[run,"P2O5_sd"],
                setup.loc[run,"H2O_sd"],setup.loc[run,"CO2ppm_sd"],setup.loc[run,"STppm_sd"], setup.loc[run,"Fe3FeT_sd"]]])
                             
    results = results.append(results1, ignore_index=True)
    results1 = pd.DataFrame([["sd types","",setup.loc[run,"SiO2_sd_type"],setup.loc[run,"TiO2_sd_type"],setup.loc[run,"Al2O3_sd_type"],setup.loc[run,"FeOT_sd_type"],setup.loc[run,"MnO_sd_type"],setup.loc[run,"MgO_sd_type"],setup.loc[run,"CaO_sd_type"],setup.loc[run,"Na2O_sd_type"],setup.loc[run,"K2O_sd_type"],setup.loc[run,"P2O5_sd_type"],
                setup.loc[run,"H2O_sd_type"],setup.loc[run,"CO2ppm_sd_type"],setup.loc[run,"STppm_sd_type"], setup.loc[run,"Fe3FeT_sd_type"]]])
                             
    results = results.append(results1, ignore_index=True)
    for n in range(0,iterations,1): # n is number of rows of data in conditions file
        if setup.loc[run,"SiO2_sd_type"] == "A": # absolute
            SiO2_sd = setup.loc[run,"SiO2_sd"]
        else:
            SiO2_sd = setup.loc[run,"SiO2_sd"]*setup.loc[run,"SiO2"]
        SiO2 = float(np.random.normal(setup.loc[run,"SiO2"],SiO2_sd,1))
        
        if setup.loc[run,"TiO2_sd_type"] == "A": # absolute
            TiO2_sd = setup.loc[run,"TiO2_sd"]
        else:
            TiO2_sd = setup.loc[run,"TiO2_sd"]*setup.loc[run,"TiO2"]
        TiO2 = float(np.random.normal(setup.loc[run,"TiO2"],TiO2_sd,1))
        
        if setup.loc[run,"Al2O3_sd_type"] == "A": # absolute
            Al2O3_sd = setup.loc[run,"Al2O3_sd"]
        else:
            Al2O3_sd = setup.loc[run,"Al2O3_sd"]*setup.loc[run,"Al2O3"]
        Al2O3 = float(np.random.normal(setup.loc[run,"Al2O3"],Al2O3_sd,1))

        if setup.loc[run,"FeOT_sd_type"] == "A": # absolute
            FeOT_sd = setup.loc[run,"FeOT_sd"]
        else:
            FeOT_sd = setup.loc[run,"FeOT_sd"]*setup.loc[run,"FeOT"]
        FeOT = float(np.random.normal(setup.loc[run,"FeOT"],FeOT_sd,1))

        if setup.loc[run,"MnO_sd_type"] == "A": # absolute
            MnO_sd = setup.loc[run,"MnO_sd"]
        else:
            MnO_sd = setup.loc[run,"MnO_sd"]*setup.loc[run,"MnO"]
        MnO = float(np.random.normal(setup.loc[run,"MnO"],MnO_sd,1))

        if setup.loc[run,"MgO_sd_type"] == "A": # absolute
            MgO_sd = setup.loc[run,"MgO_sd"]
        else:
            MgO_sd = setup.loc[run,"MgO_sd"]*setup.loc[run,"MgO"]
        MgO = float(np.random.normal(setup.loc[run,"MgO"],MgO_sd,1))
        
        if setup.loc[run,"CaO_sd_type"] == "A": # absolute
            CaO_sd = setup.loc[run,"CaO_sd"]
        else:
            CaO_sd = setup.loc[run,"CaO_sd"]*setup.loc[run,"CaO"]
        CaO = float(np.random.normal(setup.loc[run,"CaO"],CaO_sd,1))

        if setup.loc[run,"Na2O_sd_type"] == "A": # absolute
            Na2O_sd = setup.loc[run,"Na2O_sd"]
        else:
            Na2O_sd = setup.loc[run,"Na2O_sd"]*setup.loc[run,"Na2O"]
        Na2O = float(np.random.normal(setup.loc[run,"Na2O"],Na2O_sd,1))
       
        if setup.loc[run,"K2O_sd_type"] == "A": # absolute
            K2O_sd = setup.loc[run,"K2O_sd"]
        else:
            K2O_sd = setup.loc[run,"K2O_sd"]*setup.loc[run,"K2O"]
        K2O = float(np.random.normal(setup.loc[run,"K2O"],K2O_sd,1))
        
        if setup.loc[run,"P2O5_sd_type"] == "A": # absolute
            P2O5_sd = setup.loc[run,"P2O5_sd"]
        else:
            P2O5_sd = setup.loc[run,"P2O5_sd"]*setup.loc[run,"P2O5"]
        P2O5 = float(np.random.normal(setup.loc[run,"P2O5"],P2O5_sd,1))
        
        if setup.loc[run,"H2O_sd_type"] == "A": # absolute
            H2O_sd = setup.loc[run,"H2O_sd"]
        else:
            H2O_sd = setup.loc[run,"H2O_sd"]*setup.loc[run,"H2O"]
        H2O = float(np.random.normal(setup.loc[run,"H2O"],H2O_sd,1))
        
        if setup.loc[run,"CO2ppm_sd_type"] == "A": # absolute
            CO2ppm_sd = setup.loc[run,"CO2ppm_sd"]
        else:
            CO2ppm_sd = setup.loc[run,"CO2ppm_sd"]*setup.loc[run,"CO2ppm"]
        CO2ppm = float(np.random.normal(setup.loc[run,"CO2ppm"],CO2ppm_sd,1))
        
        if setup.loc[run,"STppm_sd_type"] == "A": # absolute
            STppm_sd = setup.loc[run,"STppm_sd"]
        else:
            STppm_sd = setup.loc[run,"STppm_sd"]*setup.loc[run,"STppm"]
        STppm = float(np.random.normal(setup.loc[run,"STppm"],STppm_sd,1))
        
        if setup.loc[run,"Fe3FeT_sd_type"] == "A": # absolute
            Fe3FeT_sd = setup.loc[run,"Fe3FeT_sd"]
        else:
            Fe3FeT_sd = setup.loc[run,"Fe3FeT_sd"]*setup.loc[run,"Fe3FeT"]
        Fe3FeT = float(np.random.normal(setup.loc[run,"Fe3FeT"],Fe3FeT_sd,1))

        results1 = pd.DataFrame([[run,setup.loc[run,"T_C"],SiO2,TiO2,Al2O3,FeOT,MnO,MgO,CaO,Na2O,K2O,P2O5,
                H2O,CO2ppm,STppm, Fe3FeT]])
        results = results.append(results1, ignore_index=True)
        results.to_csv('random_compositions.csv', index=False, header=False)
        print(n, setup.loc[run,"Sample"],SiO2)
 

# calculations.py

import pandas as pd
from datetime import date
import gmpy2 as gp

import melt_gas as mg
import equilibrium_equations as eq


###########################
### saturation pressure ###
###########################

# for a given melt composition, calculate the saturation pressure
def P_sat(run,PT,melt_wf,setup,species,models,Ptol,nr_step,nr_tol):
    ST = melt_wf["ST"]
    melt_wf1 = melt_wf # to work out P_sat
    
    def Pdiff(guess,run,melt_wf,setup,species,models):
        PT["P"] = guess
        difference = abs(guess - mg.p_tot(run,PT,melt_wf,setup,species,models))
        return difference

    guess0 = 40000. # initial guess for pressure
    PT["P"] = guess0
    melt_wf1["Fe3FeT"] = mg.Fe3FeT_i(run,PT,setup,species,models)

    xm_H2O_, wm_H2O_, xm_CO2_, wm_CO2_, wm_H2_, wm_CO_, wm_CH4_, wm_H2S_, wm_S2m_, wm_S6p_, H2O_HT, H2_HT, CH4_HT, CO2_CT, CO_CT, CH4_CT, S6p_ST, S2m_ST, H2S_ST, H2S_HT = eq.melt_speciation(run,PT,melt_wf1,setup,species,models,nr_step,nr_tol)
    melt_wf1["H2OT"] = wm_H2O_
    melt_wf1["CO2"] = wm_CO2_
    melt_wf1["S2-"] = wm_S2m_
    melt_wf1["ST"] = ST
    delta1 = Pdiff(guess0,run,melt_wf1,setup,species,models)
    while delta1 > Ptol :
        delta1 = Pdiff(guess0,run,melt_wf1,setup,species,models)
        guess0 = mg.p_tot(run,PT,melt_wf1,setup,species,models)
        guess0 = float(guess0)
        PT["P"] = guess0
        melt_wf1["Fe3FeT"] = mg.Fe3FeT_i(run,PT,setup,species,models)
        xm_H2O_, wm_H2O_, xm_CO2_, wm_CO2_, wm_H2_, wm_CO_, wm_CH4_, wm_H2S_, wm_S2m_, wm_S6p_, H2O_HT, H2_HT, CH4_HT, CO2_CT, CO_CT, CH4_CT, S6p_ST, S2m_ST, H2S_ST, H2S_HT = eq.melt_speciation(run,PT,melt_wf1,setup,species,models,nr_step,nr_tol)         
        melt_wf1["H2OT"] = wm_H2O_
        melt_wf1["CO2"] = wm_CO2_
        melt_wf1["S2-"] = wm_S2m_
    else:
        P_sat = guess0
        xm_H2O_, wm_H2O_, xm_CO2_, wm_CO2_, wm_H2_, wm_CO_, wm_CH4_, wm_H2S_, wm_S2m_, wm_S6p_, H2O_HT, H2_HT, CH4_HT, CO2_CT, CO_CT, CH4_CT, S6p_ST, S2m_ST, H2S_ST, H2S_HT = eq.melt_speciation(run,PT,melt_wf1,setup,species,models,nr_step,nr_tol)
        
    melt_wf["ST"] = ST
    return P_sat, wm_H2O_, wm_CO2_, wm_H2_, wm_CO_, wm_CH4_, wm_S2m_, wm_S6p_, wm_H2S_, H2O_HT, H2_HT, CH4_HT, H2S_HT, CO2_CT, CO_CT, CH4_CT, S6p_ST, S2m_ST, H2S_ST

# Pvsat with just H2O and CO2 in vapour
def P_sat_H2O_CO2(run,PT,melt_wf,setup,species,models,Ptol,nr_step,nr_tol): 
    
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
    results = pd.DataFrame([["oxygen fugacity","carbonate solubility","water solubility","sulphide solubility","sulphate solubility","ideal gas","carbonylsulphide","insolubles","Date"]])
    results1 = pd.DataFrame([[models.loc["fO2","option"],models.loc["carbonate","option"],models.loc["water","option"],models.loc["sulphide","option"],models.loc["sulphate","option"],models.loc["ideal_gas","option"],models.loc["carbonylsulphide","option"],models.loc['insolubles','option'],date.today()]])
    results = results.append(results1, ignore_index=True)
    results1 = ([["Sample","Saturation pressure (bar)","T ('C)","fO2 (DNNO)","fO2 (DFMQ)",
                  "SiO2 (wt%)","TiO2 (wt%)","Al2O3 (wt%)","FeOT (wt%)","MnO (wt%)","MgO (wt%)","CaO (wt%)","Na2O (wt%)","K2O (wt%)","P2O5 (wt%)",
                "H2OT (wt%)","CO2 (ppm)","ST (ppm)","Fe3/FeT","CT input (ppm)","HT input (ppm)","H2OT (wt%)","H2 (ppm)","CH4 (ppm)","CO2 (ppm)","CO (ppm)","S2- (ppm)","S6+ (ppm)","H2S (ppm)","C reduced calc. (ppm)","H reduced calc. (ppm)",
                  "H_H2O/HT", "H_H2/HT", "H_CH4/HT", "H_H2S/HT", "C_CO2/CT", "C_CO/CT", "C_CH4/CT", "S2-/ST", "S6+/ST", "H2S/ST",
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
            P_sat_H2O_CO2_only, xg_H2O_H2O_CO2_only, xg_CO2_H2O_CO2_only, f_H2O_H2O_CO2_only, f_CO2_H2O_CO2_only, p_H2O_H2O_CO2_only, p_CO2_H2O_CO2_only  = P_sat_H2O_CO2(run,PT,melt_wf,setup,species,models,p_tol,nr_step,nr_tol)
        
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
        gas_mf = {"O2":mg.xg_O2(run,PT,melt_wf,setup,species,models),"CO":mg.xg_CO(run,PT,melt_wf,setup,species,models),"CO2":mg.xg_CO2(run,PT,melt_wf,setup,species,models),"H2":mg.xg_H2(run,PT,melt_wf,setup,species,models),"H2O":mg.xg_H2O(run,PT,melt_wf,setup,species,models),"CH4":mg.xg_CH4(run,PT,melt_wf,setup,species,models),"S2":mg.xg_S2(run,PT,melt_wf,setup,species,models),"SO2":mg.xg_SO2(run,PT,melt_wf,setup,species,models),"H2S":mg.xg_H2S(run,PT,melt_wf,setup,species,models),"OCS":mg.xg_OCS(run,PT,melt_wf,setup,species,models),"Xg_t":mg.Xg_tot(run,PT,melt_wf,setup,species,models),"wt_g":0.}

        # forward calculate H, C, and S in the melt from reduced species
        if models.loc['insolubles','option'] == 'no':
            wm_H2_, wm_CH4_, wm_CO_, wm_H2S_, wm_S6p_, H2O_HT, H2_HT, CH4_HT, H2S_HT, CO2_CT, CO_CT, CH4_CT, S2m_ST, S6p_ST, H2S_ST, C_T, H_T, S_T = mg.conc_insolubles(run,PT,melt_wf,setup,species,models)
        ppmCreduced = (species.loc["C","M"]*((wm_CH4_/species.loc["CH4","M"])+ (wm_CO_/species.loc["CO","M"])))*1000000.
        ppmHreduced = (species.loc["H","M"]*(((4.*wm_CH4_)/species.loc["CH4","M"]) + ((2.*wm_H2_)/species.loc["H2","M"])))*1000000.
       
        ### store results ###
        results2 = pd.DataFrame([[setup.loc[run,"Sample"],PT["P"],
                setup.loc[run,"T_C"],mg.fO22Dbuffer(PT,mg.f_O2(run,PT,melt_wf,setup,species,models),"NNO"),mg.fO22Dbuffer(PT,mg.f_O2(run,PT,melt_wf,setup,species,models),"FMQ"),setup.loc[run,"SiO2"],setup.loc[run,"TiO2"],setup.loc[run,"Al2O3"],mg.Wm_FeOT(run,setup,species),setup.loc[run,"MnO"],setup.loc[run,"MgO"],setup.loc[run,"CaO"],setup.loc[run,"Na2O"],setup.loc[run,"K2O"],setup.loc[run,"P2O5"],
                setup.loc[run,"H2O"],setup.loc[run,"CO2ppm"],wm_ST*1000000.,melt_wf["Fe3FeT"],melt_wf['CT']*1000000.,melt_wf['HT']*1000000.,wm_H2O_*100.,
wm_H2_*1000000.,wm_CH4_*1000000.,wm_CO2_*1000000.,wm_CO_*1000000.,melt_wf['S2-']*1000000.,wm_S6p_*1000000.,wm_H2S_*1000000.,ppmCreduced,ppmHreduced, 
                H2O_HT, H2_HT, CH4_HT, H2S_HT, CO2_CT, CO_CT, CH4_CT, S2m_ST, S6p_ST, H2S_ST,
                 mg.f_O2(run,PT,melt_wf,setup,species,models),mg.f_H2(run,PT,melt_wf,setup,species,models),mg.f_H2O(run,PT,melt_wf,setup,species,models),mg.f_S2(run,PT,melt_wf,setup,species,models),mg.f_SO2(run,PT,melt_wf,setup,species,models),mg.f_H2S(run,PT,melt_wf,setup,species,models),mg.f_CO2(run,PT,melt_wf,setup,species,models),mg.f_CO(run,PT,melt_wf,setup,species,models),mg.f_CH4(run,PT,melt_wf,setup,species,models),mg.f_OCS(run,PT,melt_wf,setup,species,models),
                #mg.y_O2(PT,models),mg.y_H2(PT,models),mg.y_H2O(PT,models),mg.y_S2(PT,models),mg.y_SO2(PT,models),mg.y_H2S(PT,models),mg.y_CO2(PT,models),mg.y_CO(PT,models),mg.y_CH4(PT,models),mg.y_OCS(PT,models),
                mg.p_O2(run,PT,melt_wf,setup,species,models),mg.p_H2(run,PT,melt_wf,setup,species,models),mg.p_H2O(run,PT,melt_wf,setup,species,models),mg.p_S2(run,PT,melt_wf,setup,species,models),mg.p_SO2(run,PT,melt_wf,setup,species,models),mg.p_H2S(run,PT,melt_wf,setup,species,models),mg.p_CO2(run,PT,melt_wf,setup,species,models),mg.p_CO(run,PT,melt_wf,setup,species,models),mg.p_CH4(run,PT,melt_wf,setup,species,models),mg.p_OCS(run,PT,melt_wf,setup,species,models),
                mg.xg_O2(run,PT,melt_wf,setup,species,models),mg.xg_H2(run,PT,melt_wf,setup,species,models),mg.xg_H2O(run,PT,melt_wf,setup,species,models),mg.xg_S2(run,PT,melt_wf,setup,species,models),mg.xg_SO2(run,PT,melt_wf,setup,species,models),mg.xg_H2S(run,PT,melt_wf,setup,species,models),mg.xg_CO2(run,PT,melt_wf,setup,species,models),mg.xg_CO(run,PT,melt_wf,setup,species,models),mg.xg_CH4(run,PT,melt_wf,setup,species,models),mg.xg_OCS(run,PT,melt_wf,setup,species,models),
                 P_sat_H2O_CO2_only, xg_H2O_H2O_CO2_only, xg_CO2_H2O_CO2_only,f_H2O_H2O_CO2_only, f_CO2_H2O_CO2_only,p_H2O_H2O_CO2_only, p_CO2_H2O_CO2_only,P_sat_H2O_CO2_only-PT["P"]]])
                             
        results = results.append(results2, ignore_index=True)
        results.to_csv('saturation_pressures.csv', index=False, header=False)
        print(n, setup.loc[run,"Sample"],PT["P"])
     
        
###################
### composition ###
###################

# calculate bulk composition, including if a gas phase is present
def bulk_composition(run,PT,melt_wf,setup,species,models):
    bulk_composition = models.loc["bulk_composition","option"]
    eq_Fe = models.loc["eq_Fe","option"]
    wm_CO2 = melt_wf["CO2"]
    wm_H2O = melt_wf["H2OT"]
    wm_ST = melt_wf["ST"]
    Fe3FeT = melt_wf["Fe3FeT"]
    Fe3Fe2_ = mg.Fe3Fe2(melt_wf)
    S6ST_ = mg.S6ST(run,PT,melt_wf,setup,species,models)

    if bulk_composition == "yes":
        wt_g = 0.
    elif bulk_composition == "wtg":
        wt_g = setup.loc[run,"wt_g"]/100.
    elif bulk_composition == "CO2":
        wt_C_ = ((species.loc['C','M']*(setup.loc[run,"initial_CO2wtpc"]/100.))/species.loc['CO2','M'])
        wt_g = ((wt_C_/species.loc["C","M"]) - (wm_CO2/species.loc["CO2","M"]))/(((mg.xg_CO2(run,PT,melt_wf,setup,species,models)+mg.xg_CO(run,PT,melt_wf,setup,species,models)+mg.xg_CH4(run,PT,melt_wf,setup,species,models)+mg.xg_OCS(run,PT,melt_wf,setup,species,models))/mg.Xg_tot(run,PT,melt_wf,setup,species,models)) - (wm_CO2/species.loc["CO2","M"]))    

    if bulk_composition == "CO2":
        wt_C = wt_C_
    else:
        wt_C = species.loc["C","M"]*((wt_g*(((mg.xg_CO2(run,PT,melt_wf,setup,species,models)+mg.xg_CO(run,PT,melt_wf,setup,species,models)+mg.xg_CH4(run,PT,melt_wf,setup,species,models)+mg.xg_OCS(run,PT,melt_wf,setup,species,models))/mg.Xg_tot(run,PT,melt_wf,setup,species,models)) - (wm_CO2/species.loc["CO2","M"]))) + (wm_CO2/species.loc["CO2","M"]))

    wt_H = 2.0*species.loc["H","M"]*((wt_g*(((mg.xg_H2O(run,PT,melt_wf,setup,species,models)+mg.xg_H2(run,PT,melt_wf,setup,species,models)+2.0*mg.xg_CH4(run,PT,melt_wf,setup,species,models)+mg.xg_H2S(run,PT,melt_wf,setup,species,models))/mg.Xg_tot(run,PT,melt_wf,setup,species,models)) - (wm_H2O/species.loc["H2O","M"]))) + (wm_H2O/species.loc["H2O","M"]))
    wt_S = species.loc["S","M"]*((wt_g*(((mg.xg_SO2(run,PT,melt_wf,setup,species,models)+2.0*mg.xg_S2(run,PT,melt_wf,setup,species,models)+mg.xg_H2S(run,PT,melt_wf,setup,species,models)+mg.xg_OCS(run,PT,melt_wf,setup,species,models))/mg.Xg_tot(run,PT,melt_wf,setup,species,models)) - (wm_ST/species.loc["S","M"]))) + (wm_ST/species.loc["S","M"]))
    
    Wt = setup.loc[run, "total_mass_g"]

    if eq_Fe == "no":
        wt_Fe = 0.0
    elif eq_Fe == "yes":
        total_dissolved_volatiles = (wm_CO2 + wm_H2O + wm_ST*(1.-S6ST_) + (species.loc["SO3","M"]*((wm_ST*S6ST_)/species.loc["S","M"])))
        wt_Fe = (1.-wt_g)*(((1.0-total_dissolved_volatiles)*mg.wm_Fe_nv(run,melt_wf,setup,species))/100.0) # wt fraction of Fe

    wt_O = species.loc["O","M"]*((wt_g*(((2.0*mg.xg_CO2(run,PT,melt_wf,setup,species,models) + mg.xg_CO(run,PT,melt_wf,setup,species,models) + 2.0*mg.xg_O2(run,PT,melt_wf,setup,species,models) + mg.xg_H2O(run,PT,melt_wf,setup,species,models) + 2.0*mg.xg_SO2(run,PT,melt_wf,setup,species,models) + mg.xg_OCS(run,PT,melt_wf,setup,species,models))/mg.Xg_tot(run,PT,melt_wf,setup,species,models)) - (wm_H2O/species.loc["H2O","M"]) - ((2.0*wm_CO2)/species.loc["CO2","M"]) - (3.0*(wm_ST*S6ST_)/species.loc["S","M"]))) + (wm_H2O/species.loc["H2O","M"]) + ((2.0*wm_CO2)/species.loc["CO2","M"]) + (3.0*(wm_ST*S6ST_)/species.loc["S","M"]) + (wt_Fe/species.loc["Fe","M"])*((1.5*Fe3Fe2_+1.0)/(Fe3Fe2_+1.0)))
    return wt_C, wt_O, wt_H, wt_S, wt_Fe, wt_g, Wt
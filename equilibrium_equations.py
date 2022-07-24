# equilibrium_equations.py

import pandas as pd
import numpy as np

import melt_gas as mg

def set_system(melt_wf,models):
    wt_H = melt_wf["HT"]
    wt_C = melt_wf["CT"]
    wt_S = melt_wf["ST"]
    if wt_H > 0.0 and wt_C == 0.0 and wt_S == 0.0:
        sys = "HOFe"
    elif wt_H == 0.0 and wt_C > 0.0 and wt_S == 0.0:
        sys = "COFe"
    elif wt_H == 0.0 and wt_C == 0.0 and wt_S > 0.0:
        sys = "SOFe"  
    elif wt_H > 0.0 and wt_C > 0.0 and wt_S == 0.0:
        sys = "CHOFe"
    elif wt_H > 0.0 and wt_C == 0.0 and wt_S > 0.0:
        sys = "SHOFe"
    elif wt_H == 0.0 and wt_C > 0.0 and wt_S > 0.0:
        sys = "SCOFe"
    elif wt_H > 0.0 and wt_C > 0.0 and wt_S > 0.0:
        sys = "SCHOFe"
    return sys

#################################################
### specitation of H and C at given P and fO2 ###
#################################################

def eq_C_melt(run,PT,melt_wf,species,setup,models): # equilibrium partitioning of C in the melt in CO system
    wt_C = melt_wf['CT'] # weight fraction
    K1 = mg.KCOg(PT) 
    K2 = mg.C_CO3(run,PT,setup,species,models) # mole fraction
    K3 = (mg.C_CO(PT,models))/1000000. # weight fraction
    M_C = species.loc['C','M']
    M_CO = species.loc['CO','M']
    M_CO2 = species.loc['CO2','M']
    M_m_ = mg.M_m_SO(run,setup,species)
    fO2 = mg.f_O2(run,PT,melt_wf,setup,species,models)
    a = K3*M_C*(M_CO2 - M_m_)
    b = K3*M_m_*M_C + K1*K2*fO2**0.5*M_CO*(M_C - M_CO2*wt_C + M_m_*wt_C)
    c = -wt_C*K1*K2*fO2**0.5*M_CO*M_m_
    xm_CO2_ = (-b + (b**2 - 4.*a*c)**0.5)/(2.*a) # mole fraction CO32-
    wm_CO2_ = (xm_CO2_*M_CO2)/((xm_CO2_*M_CO2) + ((1.-xm_CO2_)*M_m_)) # weight fraction CO32- as CO2
    wm_CO_ = ((wt_C - (wm_CO2_/M_CO2)*M_C)/M_C)*M_CO # weight fraction CO
    check = M_C*((wm_CO2_/M_CO2)+(wm_CO_/M_CO))
    return xm_CO2_, wm_CO2_, wm_CO_

def eq_H_melt(run,PT,melt_wf,species,setup,models,nr_step,nr_tol): # equilibrium partitioning of H in the melt in HO system
    wt_H = melt_wf['HT'] # weight fraction
    K1 = mg.KHOg(PT) 
    K2 = mg.C_H2O(PT,models) # mole fraction
    K3 = mg.C_H2(PT,models) # ppm
    M_H = species.loc['H','M']
    M_H2 = species.loc['H2','M']
    M_H2O = species.loc['H2O','M']
    M_m_ = mg.M_m_SO(run,setup,species)
    fO2 = mg.f_O2(run,PT,melt_wf,setup,species,models)
    X = ((K1*K2)/K3)*fO2**0.5 # xH2O**2/wH2 with H2O in mole fraction and H2 in ppm
    Y = X/(X+1.) # xH2O**2/(xH2O**2+wH2) with H2O in mole fration and H2 in ppm 
    a = Y*M_H2O - Y*M_m_ - M_H2O + M_m_
    b = (Y - 1.)*M_m_
    c = 1000000.*Y*(wt_H*M_H2O - wt_H*M_m_ - 2.*M_H)
    d = 1000000.*Y*wt_H*M_m_
    constants = [a, b, c, d]
    def f(x, constants):
        return a*x**3 + b*x**2 + c*x + d
    def df(x, constants):
        return 3.*a*x**2 + 2.*b*x + c
    def dx(x, constants):
        f_ = f(x, constants)
        result =(abs(0-f_))
        return result
    x0 = mg.xm_H2OT_so(run,melt_wf,setup,species)    
    delta1 = dx(x0, constants)
    while delta1 > nr_tol:
        f_ = f(x0, constants)
        df_ = df(x0, constants)
        x0 = x0 - nr_step*(f_/df_)
        delta1 = dx(x0, constants)
    xm_H2O_ = x0        
    wm_H2O_ = (xm_H2O_*M_H2O)/(xm_H2O_*M_H2O + (1.-xm_H2O_)*M_m_) # weight fraction H2O
    wm_H2_ = ((wt_H - (wm_H2O_/M_H2O)*(2*M_H))/M_H)*(2*M_H) # weight fraction H2
    return xm_H2O_, wm_H2O_, wm_H2_

def eq_CH_melt(run,PT,melt_wf,species,setup,models,nr_step,nr_tol,guessx,guessy):
    P = PT["P"]
    wt_C = melt_wf['CT']
    wt_H = melt_wf['HT']
    fO2 = mg.f_O2(run,PT,melt_wf,setup,species,models)
    
    # equilibrium constants
    K1_ = mg.KHOg(PT)
    K2_ = mg.KCOg(PT)
    K3_ = mg.KCOHg(PT)
    K4_ = mg.C_H2O(PT,models) # mole fraction
    K5_ = mg.C_CO3(run,PT,setup,species,models) # mole fraction
    K6_ = mg.C_H2(PT,models) # ppm
    K7_ = mg.C_CO(PT,models) # ppm
    K8_ = mg.C_CH4(PT,models) # ppm
   
    # molecular masses
    M_H = species.loc['H','M']
    M_C = species.loc['C','M']
    M_CO = species.loc['CO','M']
    M_H2O = species.loc['H2O','M']
    M_H2 = species.loc['H2','M']
    M_CO2 = species.loc['CO2','M']
    M_CH4 = species.loc['CH4','M']
    M_m_ = mg.M_m_SO(run,setup,species)
    
    constants = [wt_C, wt_H, K1_, K2_, K3_, K4_, K5_, K6_, K7_, K8_, M_C, M_H, M_CO, M_CO2, M_H2, M_H2O, M_CH4, M_m_, fO2]
    
    def mg_CH(xm_CO2_,xm_H2O_):
        Xm_t = xm_CO2_*M_CO2 + xm_H2O_*M_H2O + (1.0-xm_CO2_-xm_H2O_)*M_m_
        wm_H2O_ = (xm_H2O_*M_H2O)/Xm_t # weight fraction
        wm_CO2_ = (xm_CO2_*M_CO2)/Xm_t # weight fraction
        fH2O = (xm_H2O_**2.)/K4_
        fCO2 = xm_CO2_/K5_
        fH2 = fH2O/(K1_*fO2**0.5)
        wm_H2_ = (fH2*K6_)/1000000. # weight fraction
        fCO = fCO2/(K2_*fO2**0.5)
        wm_CO_ = (fCO*K7_)/1000000. # weight fraction
        fCH4 = (fCO2*fH2O**2.)/(K3_*fO2**2.)
        wm_CH4_ = (fCH4*K8_)/1000000. # weight fraction
        return Xm_t, wm_H2O_, wm_CO2_, wm_H2_, wm_CO_, wm_CH4_
    
    def f_CH(xm_CO2_,xm_H2O_):
        Xm_t, wm_H2O_, wm_CO2_, wm_H2_, wm_CO_, wm_CH4_ = mg_CH(xm_CO2_,xm_H2O_)
        wt_m_C = M_C*((wm_CO2_/M_CO2) + (wm_CO_/M_CO) + (wm_CH4_/M_CH4))
        wt_m_H = M_H*(2.*(wm_H2O_/M_H2O) + 2.*(wm_H2_/M_H2) + 4.*(wm_CH4_/M_CH4))
        mbC = (wt_C - wt_m_C)
        mbH = (wt_H - wt_m_H)
        return mbC, mbH, wt_m_C, wt_m_H, 0.
    
    def df_CH(xm_CO2_,xm_H2O_,constants):
        dmbC_C = -M_C*(xm_CO2_*(-M_CO2 + M_m_)/(M_CO2*xm_CO2_ + M_H2O*xm_H2O_ + M_m_*(-xm_CO2_ - xm_H2O_ + 1.0))**2 + 1/(M_CO2*xm_CO2_ + M_H2O*xm_H2O_ + M_m_*(-xm_CO2_ - xm_H2O_ + 1.0)) + 1.0e-6*K8_*fO2**(-2.0)*(xm_H2O_**2.0/K4_)**2.0/(K3_*K5_*M_CH4) + 1.0e-6*K7_*fO2**(-0.5)/(K2_*K5_*M_CO))
        dmbC_H = -M_C*(xm_CO2_*(-M_H2O + M_m_)/(M_CO2*xm_CO2_ + M_H2O*xm_H2O_ + M_m_*(-xm_CO2_ - xm_H2O_ + 1.0))**2 + 4.0e-6*K8_*fO2**(-2.0)*xm_CO2_*xm_H2O_**(-1.0)*(xm_H2O_**2.0/K4_)**2.0/(K3_*K5_*M_CH4))
        dmbH_C = -M_H*(2.0*xm_H2O_*(-M_CO2 + M_m_)/(M_CO2*xm_CO2_ + M_H2O*xm_H2O_ + M_m_*(-xm_CO2_ - xm_H2O_ + 1.0))**2 + 4.0e-6*K8_*fO2**(-2.0)*(xm_H2O_**2.0/K4_)**2.0/(K3_*K5_*M_CH4))
        dmbH_H = -M_H*(2.0*xm_H2O_*(-M_H2O + M_m_)/(M_CO2*xm_CO2_ + M_H2O*xm_H2O_ + M_m_*(-xm_CO2_ - xm_H2O_ + 1.0))**2 + 2.0/(M_CO2*xm_CO2_ + M_H2O*xm_H2O_ + M_m_*(-xm_CO2_ - xm_H2O_ + 1.0)) + 1.6e-5*K8_*fO2**(-2.0)*xm_CO2_*xm_H2O_**(-1.0)*(xm_H2O_**2.0/K4_)**2.0/(K3_*K5_*M_CH4) + 4.0e-6*K6_*fO2**(-0.5)*xm_H2O_**1.0/(K1_*K4_*M_H2))        
        return dmbC_C, dmbC_H, dmbH_C, dmbH_H
    
    xm_CO2_,xm_H2O_ = jac_newton(guessx,guessy,constants,f_CH,df_CH,nr_step,nr_tol)
    results1 = mg_CH(xm_CO2_,xm_H2O_)
    results2 = f_CH(xm_CO2_,xm_H2O_)
    return xm_CO2_,xm_H2O_, results1, results2

def eq_HS_melt(run,PT,melt_wf,species,setup,models,nr_step,nr_tol): # not sure this is right?
    wt_S = melt_wf['ST']
    wt_H = melt_wf['HT']
    fO2 = mg.f_O2(run,PT,melt_wf,setup,species,models)
    
    # equilibrium constants
    K1_ = mg.KHOSg(PT)
    K2_ = mg.C_H2O(PT,models) # mole fraction
    K3_ = mg.C_S(run,PT,melt_wf,setup,species,models)/1000000. # weight fraction
    K4_ = mg.C_SO4(run,PT,melt_wf,setup,species,models)/1000000. # weight fraction
    K5_ = mg.C_H2S(run,PT,melt_wf,setup,species,models)/1000000. # weight fraction
    K6_ = mg.KHOg(PT) 
    K7_ = mg.C_H2(PT,models)/1000000. # weight fraction
   
    # molecular masses
    M_H = species.loc['H','M']
    M_S = species.loc['S','M']
    M_H2O = species.loc['H2O','M']
    M_H2S = species.loc['H2S','M']
    M_m_ = mg.M_m_SO(run,setup,species)
    
    def dx(x):
        f_ = f(x)
        result =(abs(0-f_))
        return result

    def mg_HS(xm_H2O_):
        Xm_t = xm_H2O_*M_H2O + (1.0-xm_H2O_)*M_m_
        wm_H2O_ = (xm_H2O_*M_H2O)/Xm_t # weight fraction
        fH2O = (xm_H2O_**2.)/K2_
        fH2 = fH2O/(K6_*fO2**0.5)
        wm_H2_ = fH2*K7_ # weight fraction
        wm_S2m_ = wt_S/(1 + (K4_*fO2**2)/K3_ + (K1_*K5_*M_S*xm_H2O_**2)/(K2_*K3_*M_H2S)) # weight fraction
        wm_H2S_ = (K1_*K5_*xm_H2O_**2*wm_S2m_)/(K2_*K3_) # weight fraction
        wm_S6p_ = (K4_*fO2**2*wm_S2m_)/K3_ # weight fraction
        return Xm_t, wm_H2O_, wm_H2_, wm_S2m_, wm_S6p_, wm_H2S_
    
    def f(xm_H2O_):
        Xm_t, wm_H2O_, wm_H2_, wm_S2m_, wm_S6p_, wm_H2S_ = mg_HS(xm_H2O_)
        result = wm_H2_ + ((2.*M_H*wm_H2S_)/M_H2S) + (2.*M_H*xm_H2O_)/(M_H2O*xm_H2O_ + (1 - xm_H2O_)*M_m_) - wt_H
        return result
    
    def df(xm_H2O_):
        result = -4*K1_**2*K5_**2*M_H*M_S*wt_S*xm_H2O_**3/(K2_**2*K3_**2*M_H2S**2*(K1_*K5_*M_S*xm_H2O_**2/(K2_*K3_*M_H2S) + 1 + K4_*fO2**2/K3_)**2) + 4*K1_*K5_*M_H*wt_S*xm_H2O_/(K2_*K3_*M_H2S*(K1_*K5_*M_S*xm_H2O_**2/(K2_*K3_*M_H2S) + 1 + K4_*fO2**2/K3_)) + 2*M_H*xm_H2O_*(-M_H2O + M_m_)/(M_H2O*xm_H2O_ + M_m_*(1 - xm_H2O_))**2 + 2*M_H/(M_H2O*xm_H2O_ + M_m_*(1 - xm_H2O_)) + 2*K7_*fO2**(-0.5)*xm_H2O_/(K2_*K6_) 
        return result
    
    x0 = mg.xm_H2OT_so(run,melt_wf,setup,species)    
    delta1 = dx(x0)
    while delta1 > nr_tol:
        f_ = f(x0)
        df_ = df(x0)
        x0 = x0 - nr_step*(f_/df_)
        delta1 = dx(x0)
    xm_H2O_ = x0     
    Xm_t, wm_H2O_, wm_H2_, wm_S2m_, wm_S6p_, wm_H2S_ = mg_HS(xm_H2O_)
    return xm_H2O_, wm_H2O_, wm_H2_, wm_S2m_, wm_S6p_, wm_H2S_

def eq_CHS_melt(run,PT,melt_wf,species,setup,models,nr_step,nr_tol,guessx,guessy,guessz):
    wt_S = melt_wf['ST']
    wt_H = melt_wf['HT']
    wt_C = melt_wf['CT']
    fO2 = mg.f_O2(run,PT,melt_wf,setup,species,models)
    
    # equilibrium constants
    K1_ = mg.KHOSg(PT)
    K2_ = mg.C_H2O(PT,models) # mole fraction
    K3_ = mg.C_S(run,PT,melt_wf,setup,species,models)/1000000. # weight fraction
    K4_ = mg.C_SO4(run,PT,melt_wf,setup,species,models)/1000000. # weight fraction
    K5_ = mg.C_H2S(run,PT,melt_wf,setup,species,models)/1000000. # weight fraction
    K6_ = mg.KHOg(PT) 
    K7_ = mg.C_H2(PT,models)/1000000. # weight fraction
    K8_ = mg.KHOg(PT)
    K9_ = mg.KCOg(PT)
    K10_ = mg.KCOHg(PT)
    K11_ = mg.C_CO3(run,PT,setup,species,models) # mole fraction
    K12_ = mg.C_CO(PT,models)/1000000. # weight fraction
    K13_ = mg.C_CH4(PT,models)/1000000. # weight fraction
   
    # molecular masses
    M_H = species.loc['H','M']
    M_S = species.loc['S','M']
    M_H2O = species.loc['H2O','M']
    M_H2S = species.loc['H2S','M']
    M_m_ = mg.M_m_SO(run,setup,species)
    M_C = species.loc['C','M']
    M_CO = species.loc['CO','M']
    M_H2 = species.loc['H2','M']
    M_CO2 = species.loc['CO2','M']
    M_CH4 = species.loc['CH4','M']

    constants = [wt_C, wt_H, wt_S, K1_, K2_, K3_, K4_, K5_, K6_, K7_, K8_, K9_, K10_, K11_, K12_, K13_, M_C, M_H, M_S, M_CO, M_CO2, M_H2, M_H2O, M_CH4, M_m_, fO2]
    
    def dx(x):
        f_ = f(x)
        result =(abs(0-f_))
        return result

    def mg_CHS(xm_CO2_,xm_H2O_,wm_S2m_):
        Xm_t = xm_CO2_*M_CO2 + xm_H2O_*M_H2O + (1.0-xm_CO2_-xm_H2O_)*M_m_
        wm_H2O_ = (xm_H2O_*M_H2O)/Xm_t # weight fraction
        wm_CO2_ = (xm_CO2_*M_CO2)/Xm_t # weight fraction
        fH2O = (xm_H2O_**2.)/K2_
        fCO2 = xm_CO2_/K11_
        fH2 = fH2O/(K6_*fO2**0.5)
        wm_H2_ = fH2*K7_ # weight fraction
        fCO = fCO2/(K9_*fO2**0.5)
        wm_CO_ = fCO*K12_ # weight fraction
        fCH4 = (fCO2*fH2O**2.)/(K10_*fO2**2.)
        wm_CH4_ = fCH4*K13_ # weight fraction
        fS2 = (fO2*wm_S2m_**2.)/K3_**2.
        fH2S = (K1_*fS2**0.5*fH2O)/fO2**0.5
        wm_H2S_ = fH2S*K5_ # weight fraction
        wm_S6p_ = K4_*fS2**0.5*fO2**1.5 # weight fraction
        return Xm_t, wm_H2O_, wm_H2_, wm_CO2_, wm_CO_, wm_CH4_, wm_S2m_, wm_S6p_, wm_H2S_
    
    def f_CHS(xm_CO2_,xm_H2O_,wm_S2m_):
        Xm_t, wm_H2O_, wm_H2_, wm_CO2_, wm_CO_, wm_CH4_, wm_S2m_, wm_S6p_, wm_H2S_ = mg_CHS(xm_CO2_,xm_H2O_,wm_S2m_)
        wt_m_C = M_C*((wm_CO2_/M_CO2) + (wm_CO_/M_CO) + (wm_CH4_/M_CH4))
        wt_m_H = M_H*(2.*(wm_H2O_/M_H2O) + 2.*(wm_H2_/M_H2) + 4.*(wm_CH4_/M_CH4) + 2.*(wm_H2S_/M_H2S))
        wt_m_S = M_S*((wm_S2m_/M_S) + (wm_S6p_/M_S) + (wm_H2S_/M_H2S))
        wt_m_O = "na"
        mbC = (wt_C - wt_m_C)
        mbH = (wt_H - wt_m_H)
        mbS = (wt_S - wt_m_S)
        return mbC, mbH, mbS, wt_m_C, wt_m_H, wt_m_S, wt_m_O

    def df_CHS(xm_CO2_,xm_H2O_,wm_S2m_,constants):
        dmbC_C = -M_C*(xm_CO2_*(-M_CO2 + M_m_)/(M_CO2*xm_CO2_ + M_H2O*xm_H2O_ + M_m_*(-xm_CO2_ - xm_H2O_ + 1.0))**2 + 1/(M_CO2*xm_CO2_ + M_H2O*xm_H2O_ + M_m_*(-xm_CO2_ - xm_H2O_ + 1.0)) + K12_*fO2**(-0.5)/(K11_*K9_*M_CO) + K13_*fO2**(-2.0)*(xm_H2O_**2.0/K2_)**2.0/(K10_*K11_*M_CH4))
        dmbC_H = -M_C*(xm_CO2_*(-M_H2O + M_m_)/(M_CO2*xm_CO2_ + M_H2O*xm_H2O_ + M_m_*(-xm_CO2_ - xm_H2O_ + 1.0))**2 + 4.0*K13_*fO2**(-2.0)*xm_CO2_*xm_H2O_**(-1.0)*(xm_H2O_**2.0/K2_)**2.0/(K10_*K11_*M_CH4))
        dmbC_S = 0.
        dmbH_C = -M_H*(2.0*xm_H2O_*(-M_CO2 + M_m_)/(M_CO2*xm_CO2_ + M_H2O*xm_H2O_ + M_m_*(-xm_CO2_ - xm_H2O_ + 1.0))**2 + 4.0*K13_*fO2**(-2.0)*(xm_H2O_**2.0/K2_)**2.0/(K10_*K11_*M_CH4))
        dmbH_H = -M_H*(4.0*K1_*K5_*fO2**(-0.5)*xm_H2O_**1.0*(K3_**(-2.0)*fO2*wm_S2m_**2.0)**0.5/(K2_*M_H2S) + 2.0*xm_H2O_*(-M_H2O + M_m_)/(M_CO2*xm_CO2_ + M_H2O*xm_H2O_ + M_m_*(-xm_CO2_ - xm_H2O_ + 1.0))**2 + 2.0/(M_CO2*xm_CO2_ + M_H2O*xm_H2O_ + M_m_*(-xm_CO2_ - xm_H2O_ + 1.0)) + 4.0*K7_*fO2**(-0.5)*xm_H2O_**1.0/(K2_*K6_*M_H2) + 16.0*K13_*fO2**(-2.0)*xm_CO2_*xm_H2O_**(-1.0)*(xm_H2O_**2.0/K2_)**2.0/(K10_*K11_*M_CH4))
        dmbH_S = -2.0*K1_*K5_*M_H*fO2**(-0.5)*wm_S2m_**(-1.0)*xm_H2O_**2.0*(K3_**(-2.0)*fO2*wm_S2m_**2.0)**0.5/(K2_*M_H2S)
        dmbS_C = 0.
        dmbS_H = -2.0*K1_*K5_*M_S*fO2**(-0.5)*xm_H2O_**1.0*(K3_**(-2.0)*fO2*wm_S2m_**2.0)**0.5/(K2_*M_H2S)
        dmbS_S = -M_S*(1.0*K1_*K5_*fO2**(-0.5)*wm_S2m_**(-1.0)*xm_H2O_**2.0*(K3_**(-2.0)*fO2*wm_S2m_**2.0)**0.5/(K2_*M_H2S) + 1.0*K4_*fO2**1.5*wm_S2m_**(-1.0)*(K3_**(-2.0)*fO2*wm_S2m_**2.0)**0.5/M_S + 1/M_S)        
        return dmbC_C, dmbC_H, dmbC_S, dmbH_C, dmbH_H, dmbH_S, dmbS_C, dmbS_H, dmbS_S
    
    xm_CO2_,xm_H2O_,wm_S2m_ = jac_newton3(guessx,guessy,guessz,constants,f_CHS,df_CHS,nr_step,nr_tol)
    results1 = mg_CHS(xm_CO2_,xm_H2O_,wm_S2m_)
    results2 = f_CHS(xm_CO2_,xm_H2O_,wm_S2m_)
    return xm_CO2_,xm_H2O_,wm_S2m_, results1, results2

def melt_speciation(run,PT,melt_wf,setup,species,models,nr_step,nr_tol):
    system = set_system(melt_wf,models)
    wt_C = melt_wf['CT']
    wt_H = melt_wf['HT']
    wt_S = melt_wf['ST']
    M_H = species.loc['H','M']
    M_S = species.loc['S','M']
    M_C = species.loc['C','M']
    M_CO = species.loc['CO','M']
    M_H2O = species.loc['H2O','M']
    M_H2 = species.loc['H2','M']
    M_CO2 = species.loc['CO2','M']
    M_CH4 = species.loc['CH4','M']
    M_H2S = species.loc['H2S','M']
    
    if models.loc['insolubles','option'] == 'yes':
        if system == "HOFe":
            xm_H2O_, wm_H2O_, wm_H2_ = eq_H_melt(run,PT,melt_wf,species,setup,models,nr_step,nr_tol)
            xm_CO2_, wm_CO2_, wm_CO_, wm_CH4_, wm_H2S_, wm_S2m_, wm_S6p_ = 0., 0., 0., 0., 0., 0., 0.
            H2_HT = wm_H2_/wt_H
            H2O_HT = ((2.*(wm_H2O_/M_H2O))*M_H)/wt_H
            CH4_HT = 0.
            CO2_CT = 0.
            CO_CT = 0.
            CH4_CT = 0.
            H2S_HT = 0.
            S2m_ST = 0.
            S6p_ST = 0.
            H2S_ST = 0.
        elif system == "COFe":
            xm_CO2_, wm_CO2_, wm_CO_ = eq_C_melt(run,PT,melt_wf,species,setup,models)
            xm_H2O_, wm_H2O_, wm_H2_, wm_CH4_, wm_H2S_, wm_S2m_, wm_S6p_ = 0., 0., 0., 0., 0., 0., 0.
            CO_CT = ((wm_CO_/M_CO)*M_C)/wt_C
            CO2_CT = ((wm_CO2_/M_CO2)*M_C)/wt_C
            CH4_CT, H2O_HT, H2_HT, CH4_HT, H2S_HT, S2m_ST, S6p_ST, H2S_ST = 0., 0., 0., 0., 0., 0., 0., 0.           
        elif system == "SOFe":
            S6p_ST = mg.S6ST(run,PT,melt_wf,setup,species,models)
            S2m_ST = 1. - S6p_ST
            wm_S2m_ = S2m_ST*wt_S
            wm_S6p_ = S6p_ST*wt_S
            xm_H2O_, wm_H2O_, xm_CO2_, wm_CO2_, wm_H2_, wm_CO_, wm_CH4_, wm_H2S_, H2O_HT, H2_HT, CH4_HT, CO2_CT, CO_CT, CH4_CT, H2S_HT, H2S_ST = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        elif system == "CHOFe":
            guessx = mg.xm_CO2_so(run,melt_wf,setup,species)
            guessy = mg.xm_H2OT_so(run,melt_wf,setup,species)
            xm_CO2_,xm_H2O_, A, B = eq_CH_melt(run,PT,melt_wf,species,setup,models,nr_step,nr_tol,guessx,guessy)
            Xm_t, wm_H2O_, wm_CO2_, wm_H2_, wm_CO_, wm_CH4_ = A
            CO_CT = ((wm_CO_/M_CO)*M_C)/wt_C
            CO2_CT = ((wm_CO2_/M_CO2)*M_C)/wt_C
            CH4_CT = ((wm_CH4_/M_CH4)*M_C)/wt_C
            H2O_HT = ((2.*(wm_H2O_/M_H2O))*M_H)/wt_H
            H2_HT = wm_H2_/wt_H
            CH4_HT = ((4.*(wm_CH4_/M_CH4))*M_H)/wt_H
            wm_S2m_, wm_S6p_, wm_H2S_, H2S_HT, S2m_ST, S6p_ST, H2S_ST = 0., 0., 0., 0., 0., 0., 0.
        elif system == "SHOFe":
            xm_H2O_, wm_H2O_, wm_H2_, wm_S2m_, wm_S6p_, wm_H2S_ = eq_HS_melt(run,PT,melt_wf,species,setup,models,nr_step,nr_tol)
            H2O_HT = ((2.*(wm_H2O_/M_H2O))*M_H)/wt_H
            H2_HT = wm_H2_/wt_H
            H2S_HT = ((2.*(wm_H2S_/M_H2S))*M_H)/wt_H
            S2m_ST = wm_S2m_/wt_S
            S6p_ST = wm_S6p_/wt_S
            H2S_ST = (M_S*(wm_H2S_/M_H2S))/wt_S
            CH4_HT, CO2_CT, CO_CT, CH4_CT, xm_CO2_, wm_CO_, wm_CH4_, wm_CO2_  = 0., 0., 0., 0., 0., 0., 0., 0.
        elif system == "SCOFe":
            xm_CO2_, wm_CO2_, wm_CO_ = eq_C_melt(run,PT,melt_wf,species,setup,models)
            xm_H2O_, wm_H2O_, wm_H2_, wm_CH4_, wm_H2S_ = 0., 0., 0., 0., 0.
            CO_CT = ((wm_CO_/M_CO)*M_C)/wt_C
            CO2_CT = ((wm_CO2_/M_CO2)*M_C)/wt_C
            S6p_ST = mg.S6ST(run,PT,melt_wf,setup,species,models)
            S2m_ST = 1. - S6p_ST
            wm_S2m_ = S2m_ST*wt_S
            wm_S6p_ = S6p_ST*wt_S            
            CH4_CT, H2O_HT, H2_HT, CH4_HT, H2S_HT, H2S_ST, = 0., 0., 0., 0., 0., 0.
        elif system == "SCHOFe":
            if models.loc['H2S_m','option'] == 'yes':
                guessx = mg.xm_CO2_so(run,melt_wf,setup,species)
                guessy = mg.xm_H2OT_so(run,melt_wf,setup,species)
                guessz = wt_S
                xm_CO2_,xm_H2O_,wm_S2m_, A, B = eq_CHS_melt(run,PT,melt_wf,species,setup,models,nr_step,nr_tol,guessx,guessy,guessz)
                Xm_t, wm_H2O_, wm_H2_, wm_CO2_, wm_CO_, wm_CH4_, wm_S2m_, wm_S6p_, wm_H2S_ = A
                CO_CT = ((wm_CO_/M_CO)*M_C)/wt_C
                CO2_CT = ((wm_CO2_/M_CO2)*M_C)/wt_C
                CH4_CT = ((wm_CH4_/M_CH4)*M_C)/wt_C
                H2O_HT = ((2.*(wm_H2O_/M_H2O))*M_H)/wt_H
                H2_HT = wm_H2_/wt_H
                CH4_HT = ((4.*(wm_CH4_/M_CH4))*M_H)/wt_H
                H2S_HT = ((2.*(wm_H2S_/M_H2S))*M_H)/wt_H
                S6p_ST = wm_S6p_/wt_S
                S2m_ST = wm_S2m_/wt_S
                H2S_ST = (M_S*(wm_H2S_/M_H2S))/wt_S
            elif models.loc['H2S_m','option'] == 'no':
                guessx = mg.xm_CO2_so(run,melt_wf,setup,species)
                guessy = mg.xm_H2OT_so(run,melt_wf,setup,species)
                xm_CO2_,xm_H2O_, A, B = eq_CH_melt(run,PT,melt_wf,species,setup,models,nr_step,nr_tol,guessx,guessy)
                Xm_t, wm_H2O_, wm_CO2_, wm_H2_, wm_CO_, wm_CH4_ = A
                CO_CT = ((wm_CO_/M_CO)*M_C)/wt_C
                CO2_CT = ((wm_CO2_/M_CO2)*M_C)/wt_C
                CH4_CT = ((wm_CH4_/M_CH4)*M_C)/wt_C
                H2O_HT = ((2.*(wm_H2O_/M_H2O))*M_H)/wt_H
                H2_HT = wm_H2_/wt_H
                CH4_HT = ((4.*(wm_CH4_/M_CH4))*M_H)/wt_H
                S6p_ST = mg.S6ST(run,PT,melt_wf,setup,species,models)
                S2m_ST = 1. - S6p_ST
                wm_S2m_ = S2m_ST*wt_S
                wm_S6p_ = S6p_ST*wt_S
                wm_H2S_, H2S_HT, H2S_ST = 0., 0., 0.            
    else:
        wm_H2O_ = melt_wf["H2OT"]
        xm_H2O_ = mg.xm_H2OT_so(run,melt_wf,setup,species)
        if system == "COFe" or system == "SCOFe" or system == "SOFe":
            H2O_HT = 0.
        else:
            H2O_HT = 1.
        wm_CO2_ = melt_wf["CO2"]
        xm_CO2_ = mg.xm_CO2_so(run,melt_wf,setup,species)
        if system == "HOFe" or system == "SHOFe" or system == "SOFe":
            CO2_CT = 0.
        else:
            CO2_CT = 1.
        if system == "COFe" or system == "HOFe" or system == "CHOFe":
            S6p_ST, S2m_ST, wm_S2m_, wm_S6p_ = 0., 0., 0., 0.           
        else:
            S6p_ST = mg.S6ST(run,PT,melt_wf,setup,species,models)
            S2m_ST = 1. - S6p_ST  
            wm_S2m_ = S2m_ST*wt_S
            wm_S6p_ = S6p_ST*wt_S                
        wm_CO_, wm_H2_, wm_CH4_, wm_H2S_, CO_CT, CH4_CT, H2_HT, CH4_HT, H2S_HT, H2S_ST = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    return xm_H2O_, wm_H2O_, xm_CO2_, wm_CO2_, wm_H2_, wm_CO_, wm_CH4_, wm_H2S_, wm_S2m_, wm_S6p_, H2O_HT, H2_HT, CH4_HT, CO2_CT, CO_CT, CH4_CT, S6p_ST, S2m_ST, H2S_ST, H2S_HT

  
    
##############
### solver ###
##############

def newton_raphson(x0,constants,e1,step,eqs,deriv):
    # create results table
    results = pd.DataFrame([["guessx","diff","step"]])  
    results.to_csv('results_newtraph.csv', index=False, header=False)
    
    def dx(x,eqs):
        f_,wtg1,wtg2 = eqs(x)
        result =(abs(0-f_))
        return result
    
    delta1 = dx(x0,eqs)
    
    results1 = pd.DataFrame([[x0,delta1,step]]) 
    results = results.append(results1, ignore_index=True)
    results.to_csv('results_newtraph.csv', index=False, header=False)     

    while delta1 > e1:
        while x0 < 0.:
            results1 = pd.DataFrame([[x0,delta1,step]]) 
            results = results.append(results1, ignore_index=True)
            results.to_csv('results_newtraph.csv', index=False, header=False)     
            step = step/10.
            x0 = x0 - step*(f_/df_)
        f_,wtg1,wtg2 = eqs(x0)
        df_ = deriv(x0,constants)
        x0 = x0 - step*(f_/df_)
        delta1 = dx(x0,eqs)
        results1 = pd.DataFrame([[x0,delta1,step]]) 
        results = results.append(results1, ignore_index=True)
        results.to_csv('results_newtraph.csv', index=False, header=False)     
    return x0        

def jac_newton(x0,y0,constants,eqs,deriv,step,tol,maxiter=1000):

    # create results table
    results = pd.DataFrame([["guessx","guessy","diff1","diff2","step"]])  
    results.to_csv('results_jacnewton2.csv', index=False, header=False)
    diff1, diff2, wtg1,wtg2,wtg3 = eqs(x0,y0)
    results1 = pd.DataFrame([[x0,y0,diff1,diff2,step]]) 
    results = results.append(results1, ignore_index=True)
    results.to_csv('results_jacnewton2.csv', index=False, header=False) 

    def F(eqs,x,y):
        a = eqs(x,y)
        return np.array([a[0],a[1]])
    
    def x2jac(step,deriv,eqs,guessx,guessy):
        eq1_x, eq1_y, eq2_x, eq2_y = deriv
        Func = F(eqs,guessx,guessy)
        J = np.array([[eq1_x, eq1_y],[eq2_x, eq2_y]])
        det = J[0][0]*J[1][1] - J[0][1]*J[1][0]
        inv_J = (1/det)*np.array(([J[-1][-1],-(J[0][-1])],[-(J[-1][0]),J[0][0]]), dtype=object)
        new_guess = np.array([guessx, guessy], dtype=object) - step*np.dot(inv_J, Func)
        return new_guess[0], new_guess[-1], J
    
    for iter in range(maxiter):
        deriv_ = deriv(x0,y0,constants)
        guessx, guessy, J = x2jac(step,deriv_,eqs,x0, y0)
        while guessx < 0.0 or guessy < 0.0:
            step = step/10.
            guessx, guessy, J = x2jac(step,deriv_,eqs,x0,y0)
        diff1, diff2, wtg1,wtg2,wtg3 = eqs(guessx,guessy)
        if abs(diff1) < tol and abs(diff2) < tol:
            return guessx, guessy
        elif np.isnan(float(guessx)) or np.isnan(float(guessy)):
            print("nan encountered")
        x0 = guessx
        y0 = guessy
        results1 = pd.DataFrame([[guessx, guessy,diff1,diff2,step]])
        results = results.append(results1, ignore_index=True)
        results.to_csv('results_jacnewton2.csv', index=False, header=False) 

def jac_newton3(x0,y0,z0,constants,eqs,deriv,step,tol,maxiter=1000):

# create results table
    results = pd.DataFrame([["guessx","guessy","guessz","diff1","diff2","diff3","step"]])  
    results.to_csv('results_jacnewton3.csv', index=False, header=False)
    diff1, diff2, diff3, wtg1,wtg2,wtg3,wtg4 = eqs(x0,y0,z0)
    results1 = pd.DataFrame([[x0,y0,z0,diff1,diff2,diff3,step]]) 
    results = results.append(results1, ignore_index=True)
    results.to_csv('results_jacnewton3.csv', index=False, header=False) 

    def F(eqs,x,y,z):
        a = eqs(x,y,z)
        return np.array([a[0],a[1],a[2]])
    
    def x3jac(step,deriv,eqs,guessx,guessy,guessz,constants):
        eq1_x, eq1_y, eq1_z, eq2_x, eq2_y, eq2_z, eq3_x, eq3_y, eq3_z = deriv
        Func = F(eqs,guessx,guessy,guessz)
        J = np.array([[eq1_x, eq1_y, eq1_z],[eq2_x, eq2_y, eq2_z],[eq3_x, eq3_y, eq3_z]])
        m1, m2, m3, m4, m5, m6, m7, m8, m9 = J.ravel()
        determinant = m1*m5*m9 + m4*m8*m3 + m7*m2*m6 - m1*m6*m8 - m3*m5*m7 - m2*m4*m9
        inv_jac = np.array([[m5*m9-m6*m8, m3*m8-m2*m9, m2*m6-m3*m5],
                        [m6*m7-m4*m9, m1*m9-m3*m7, m3*m4-m1*m6],
                        [m4*m8-m5*m7, m2*m7-m1*m8, m1*m5-m2*m4]])/determinant
        dot = np.dot(inv_jac, Func)  # To get the arrays as 2 3x1 columns
        new_guess = np.array([guessx, guessy, guessz]) - step*dot  # Returns a 3x1 array of guessx, guessy, guessz
        return new_guess[0], new_guess[1], new_guess[2], J  # guessx, guessy, guessz, Jacobian

    for iter in range(maxiter):
        deriv_ = deriv(x0,y0,z0,constants)
        guessx, guessy, guessz, J = x3jac(step,deriv_,eqs,x0,y0,z0,constants)
        while guessx < 0.0 or guessy < 0.0 or guessz < 0.0:
            step = step/10.
            guessx, guessy, guessz, J = x3jac(step,deriv_,eqs,x0,y0,z0,constants)
        diff1, diff2, diff3, wtg1,wtg2,wtg3,wtg4 = eqs(guessx,guessy,guessz)
        if abs(diff1) < tol and abs(diff2) < tol and abs(diff3) < tol:
            return guessx, guessy, guessz
        elif np.isnan(float(guessx)) or np.isnan(float(guessy)) or np.isnan(float(guessz)):
            print("nan encountered")
        x0 = guessx
        y0 = guessy
        z0 = guessz
        results1 = pd.DataFrame([[guessx, guessy,guessz,diff1,diff2,diff3,step]])
        results = results.append(results1, ignore_index=True)
        results.to_csv('results_jacnewton3.csv', index=False, header=False)  
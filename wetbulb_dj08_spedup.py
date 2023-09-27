#!/usr/bin/env python
# coding: utf-8

# SHORT HEADER
#
# [Twb,Teq,epott]=WetBulb(TemperatureC,Pressure,Humidity,[HumidityMode])
#
# Calculate wet-bulb temperature, equivalent temperature, and equivalent
# potential temperature from air temperature, atmospheric pressure, and specific humidity.
#
# Required input units: air temperature in C, pressure in Pa, specific humidity in kg/kg
# Output: wet-bulb temperature in C, equivalent temperature and equivalent potential temperature in K
#
# Example usage:
# From a Jupyter notebook
# Twb=WetBulb(25.,100000.,0.015,0)[0]     #should return 21.73 C
#
# From the command line
# python speedywetbulb.py 25. 100000. 0.015 > out.txt   #ditto

# Runtime on a MacBook Pro: approximately 0.3 sec for 10^6 calculations



# DETAILED HEADER
#
# Calculates wet-bulb temperature and associated variables using the Davies-Jones 2008 method.
# This entails calculating the lifting condensation temperature (Bolton 1980 eqn 22),
# then the moist potential temperature (Bolton 1980 eqn 24), 
# then the equivalent potential temperature (Bolton 1980 eqn 39),
# and finally, from equivalent potential temp, equivalent temp and theta_w (Davies-Jones 2008 eqn 3.5-3.8), 
# an accurate 'first guess' of wet-bulb temperature (Davies-Jones 2008 eqn 4.8-4.11). 
# The Newton-Raphson method is used for 2 iterations, 
# to obtain the final calculated wet-bulb temperature (Davies-Jones 2008 eqn 2.6).
#
# Reference:  Bolton: The computation of equivalent potential temperature.
# 	      Monthly weather review (1980) vol. 108 (7) pp. 1046-1053
#	      Davies-Jones: An efficient and accurate method for computing the
#	      wet-bulb temperature along pseudoadiabats. Monthly Weather Review
#	      (2008) vol. 136 (7) pp. 2764-2785
# 	      Flatau et al: Polynomial fits to saturation vapor pressure.
#	      Journal of Applied Meteorology (1992) vol. 31 pp. 1507-1513

#
# Ported from HumanIndexMod by Jonathan R Buzan, April 2016
# Ported to Python by Xianxiang Li, February 2019
#
# Further optimizations with numba and bug correction applied by Alex Goodman, April 2023,
# with consultation and inline comments by Colin Raymond

# Additional bugs noticed and corrections proposed by Rob Warren, 
# implemented here by Colin Raymond, August 2023



# Import packages

import sys
import numpy as np
import numba as nb
import time



# Import input arguments
TemperatureC=np.float64(sys.argv[1]);
Pressure=np.float64(sys.argv[2]);
Humidity=np.float64(sys.argv[3]);



# Set constants

SHR_CONST_TKFRZ = np.float64(273.15)
lambd_a = np.float64(3.504)    	# Inverse of Heat Capacity
alpha = np.float64(17.67) 	    # Constant to calculate vapor pressure
beta = np.float64(243.5)		# Constant to calculate vapor pressure
epsilon = np.float64(0.6220)	# Conversion between pressure/mixing ratio
es_C = np.float64(611.2)		# Vapor Pressure at Freezing STD (Pa)
y0 = np.float64(3036)		    # constant
y1 = np.float64(1.78)		    # constant
y2 = np.float64(0.448)		    # constant
Cf = SHR_CONST_TKFRZ	# Freezing Temp (K)
p0 = np.float64(100000)	    # Reference Pressure (Pa)
constA = np.float64(2675) 	 # Constant used for extreme cold temperatures (K)
vkp = np.float64(0.2854)	 # Heat Capacity




#Define QSat_2 function

@nb.njit(fastmath=True)
def QSat_2(T_k, p_t, p0ndplam):
    # Constants used to calculate es(T)
    # Clausius-Clapeyron
    tcfbdiff = T_k - Cf + beta
    es = es_C * np.exp(alpha*(T_k - Cf)/(tcfbdiff))
    dlnes_dT = alpha * beta/((tcfbdiff)*(tcfbdiff))
    pminuse = p_t - es
    de_dT = es * dlnes_dT

    # Constants used to calculate rs(T)
    rs = epsilon * es/(p0ndplam - es + np.spacing(1)) #eps

    # avoid bad numbers
    if rs > 1 or rs < 0:
        rs = np.nan
        
    return es,rs,dlnes_dT 



#Define main wet-bulb-temperature function

@nb.njit(fastmath=True)
def WetBulb(TemperatureC,Pressure,Humidity,HumidityMode=0):
    ###
    #Unless necessary, default to using specific humidity as input (simpler and tends to reduce error margins)#
    ###

    """
    INPUTS:
      TemperatureC	   2-m air temperature (degrees Celsius)
      Pressure	       Atmospheric Pressure (Pa)
      Humidity         Humidity -- meaning depends on HumidityMode
      HumidityMode
        0 (Default): Humidity is specific humidity (kg/kg)
        1: Humidity is relative humidity (#, max = 100)
      TemperatureC, Pressure, and Humidity should either be scalars or arrays of identical dimension.
    OUTPUTS:
      Twb	    wet bulb temperature (C)
      Teq	    Equivalent Temperature (K)
      epott 	Equivalent Potential Temperature (K)
    """
    TemperatureK = TemperatureC + SHR_CONST_TKFRZ
    pnd = (Pressure/p0)**(vkp)
    p0ndplam = p0*pnd**lambd_a

    C = SHR_CONST_TKFRZ;		# Freezing Temperature
    T1 = TemperatureK;		# Use holder for T

    es, rs, _ = QSat_2(TemperatureK, Pressure, p0ndplam) # first two returned values
    

    if HumidityMode==0:
        qin = Humidity                   # specific humidity
        mixr = (qin / (1-qin))           # corrected by Rob Warren
        vape = (Pressure * mixr) / (epsilon + mixr) #corrected by Rob Warren
        relhum = 100.0 * vape/es         # corrected by Rob Warren
    elif HumidityMode==1:
        relhum = Humidity                # relative humidity (%)
        vape = es * relhum * 0.01        # vapor pressure (Pa)
        mixr = epsilon * vape / (Pressure-vape)  #corrected by Rob Warren

    mixr = mixr * 1000
    
    # Calculate Equivalent Pot. Temp (Pressure, T, mixing ratio (g/kg), pott, epott)
    # Calculate Parameters for Wet Bulb Temp (epott, Pressure)
    D = 1.0/(0.1859*Pressure/p0 + 0.6512)
    k1 = -38.5*pnd*pnd + 137.81*pnd - 53.737
    k2 = -4.392*pnd*pnd + 56.831*pnd - 0.384

    # Calculate lifting condensation level
    tl = (1.0/((1.0/((T1 - 55))) - (np.log(relhum/100.0)/2840.0))) + 55.0

    # Theta_DL: Bolton 1980 Eqn 24.
    theta_dl = T1*((p0/(Pressure-vape))**vkp) * ((T1/tl)**(mixr*0.00028))
    # EPT: Bolton 1980 Eqn 39.
    epott = theta_dl * np.exp(((3.036/tl)-0.00178)*mixr*(1 + 0.000448*mixr))
    Teq = epott*pnd	# Equivalent Temperature at pressure
    X = (C/Teq)**3.504
    
    # Calculates the regime requirements of wet bulb equations.
    invalid = Teq > 600 or Teq < 200
    hot = Teq > 355.15
    cold = X<1   #corrected by Rob Warren
        
    if invalid:
        return np.nan, np.nan, epott

    # Calculate Wet Bulb Temperature, initial guess
    # Extremely cold regimes: if X.gt.D, then need to calculate dlnesTeqdTeq

    es_teq, rs_teq, dlnes_dTeq = QSat_2(Teq, Pressure, p0ndplam)
    if X<=D:
        wb_temp = C + (k1 - 1.21 * cold - 1.45 * hot - (k2 - 1.21 * cold) * X + (0.58 / X) * hot)
    else:
        wb_temp = Teq - ((constA*rs_teq)/(1 + (constA*rs_teq*dlnes_dTeq)))

    # Newton-Raphson Method
    maxiter = 2
    iter = 0
    delta = 1e6

    while delta>0.01 and iter<maxiter:
        foftk_wb_temp, fdwb_temp = DJ(wb_temp, Pressure, p0ndplam)
        delta = (foftk_wb_temp - X)/fdwb_temp
        delta = np.minimum(10,delta)
        delta = np.maximum(-10,delta) #max(-10,delta)
        wb_temp = wb_temp - delta
        Twb = wb_temp
        iter = iter+1

    Tw_final=np.round(Twb-C,2)
    
    return Tw_final,Teq,epott



# Define parallelization functions for wet-bulb (optional)

@nb.njit(fastmath=True)
def WetBulb_all(tempC, Pres, relHum, Hum_mode):
    Twb = np.empty_like(tempC)
    Teq = np.empty_like(tempC)
    epott = np.empty_like(tempC)
    for i in nb.prange(Twb.size):
        Twb[i], Teq[i], epott[i] = WetBulb(tempC[i], Pres[i], relHum[i], Hum_mode)

@nb.njit(fastmath=True, parallel=True)
def WetBulb_par(tempC, Pres, relHum, Hum_mode):
    Twb = np.empty_like(tempC)
    Teq = np.empty_like(tempC)
    epott = np.empty_like(tempC)
    for i in nb.prange(Twb.size):
        Twb[i], Teq[i], epott[i] = WetBulb(tempC[i], Pres[i], relHum[i], Hum_mode)



# Define helper functions for usage in the Davies-Jones wet-bulb algorithm

@nb.njit(fastmath=True)
def DJ(T_k, p_t, p0ndplam):
    # Constants used to calculate es(T)
    # Clausius-Clapeyron
    tcfbdiff = T_k - Cf + beta
    es = es_C * np.exp(alpha*(T_k - Cf)/(tcfbdiff))
    dlnes_dT = alpha * beta/((tcfbdiff)*(tcfbdiff))
    pminuse = p_t - es
    de_dT = es * dlnes_dT

    # Constants used to calculate rs(T)
    rs = epsilon * es/(p0ndplam - es + np.spacing(1)) #eps)
    prersdt = epsilon * p_t/((pminuse)*(pminuse))
    rsdT = prersdt * de_dT

    # Constants used to calculate g(T)
    rsy2rs2 = rs + y2*rs*rs
    oty2rs = 1 + 2.0*y2*rs
    y0tky1 = y0/T_k - y1
    goftk = y0tky1 * (rs + y2 * rs * rs)
    gdT = - y0 * (rsy2rs2)/(T_k*T_k) + (y0tky1)*(oty2rs)*rsdT

    # Calculations for calculating f(T,ndimpress)
    foftk = ((Cf/T_k)**lambd_a)*(1 - es/p0ndplam)**(vkp*lambd_a)* \
        np.exp(-lambd_a*goftk)
    fdT = -lambd_a*(1.0/T_k + vkp*de_dT/pminuse + gdT) * foftk  #derivative corrected by Qinqin Kong

    return foftk,fdT



#Benchmark speeds for 10^6 wet-bulb calculations (optional)

dotest=0;
if dotest==1:
    numvals=10**6;

    starttime=time.time();
    temps_c=np.linspace(-20,40,numvals);
    pres=np.linspace(85000,105000,numvals);
    humidities=np.linspace(0.002,0.25,numvals);
    HumidityMode=0;
    for i in range(0,numvals-1):
        myres=WetBulb(temps_c[i],pres[i],humidities[i],HumidityMode);
    endtime=time.time();
    print('Regular execution time:', endtime-starttime, 'seconds')

    starttime=time.time();
    temps_c=np.linspace(-20,40,numvals);
    pres=np.linspace(85000,105000,numvals);
    relhumidities=np.linspace(5,100,numvals);
    myres=WetBulb_all(temps_c,pres,relhumidities,HumidityMode);
    endtime=time.time();
    print('WetBulb_all execution time:', endtime-starttime, 'seconds')

    starttime=time.time();
    temps_c=np.linspace(-20,40,numvals);
    pres=np.linspace(85000,105000,numvals);
    relhumidities=np.linspace(5,100,numvals);
    myres=WetBulb_par(temps_c,pres,relhumidities,HumidityMode);
    endtime=time.time();
    print('WetBulb_par execution time:', endtime-starttime, 'seconds')



# Calculate desired wet-bulb temperatures
Twb=WetBulb(TemperatureC,Pressure,Humidity,0)[0]
print(Twb)





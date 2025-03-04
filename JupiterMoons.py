import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
import pandas as pd
import astropy.stats as astats
import scipy.stats


JupX = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet3', usecols = 'B', dtype=float).to_numpy().flatten()
JupY = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet3', usecols = 'C', dtype=float).to_numpy().flatten()
IoX = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet3', usecols = 'D', dtype=float).to_numpy().flatten()
IoY = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet3', usecols = 'E', dtype=float).to_numpy().flatten()
EuroX = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet3', usecols = 'F', dtype=float).to_numpy().flatten()
EuroY = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet3', usecols = 'G', dtype=float).to_numpy().flatten()
GanyX = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet3', usecols = 'J', dtype=float).to_numpy().flatten()
GanyY = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet3', usecols = 'K', dtype=float).to_numpy().flatten()
CalliX = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet3', usecols = 'H', dtype=float).to_numpy().flatten()
CalliY = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet3', usecols = 'I', dtype=float).to_numpy().flatten()
theta = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet3', usecols = 'L', dtype=float).to_numpy().flatten()
scale = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet3', usecols = 'm', dtype=float).to_numpy().flatten()
flip = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet3', usecols = 'n', dtype=bool).to_numpy().flatten()
JupErrX = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet3', usecols = 'O', dtype=float).to_numpy().flatten()
JupErrY = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet3', usecols = 'P', dtype=float).to_numpy().flatten()
Time_mins = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet3', usecols = 'X', dtype=float).to_numpy().flatten()


#JupX = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet4', usecols = 'X', dtype=float).to_numpy().flatten()
#JupY = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet4', usecols = 'X', dtype=float).to_numpy().flatten()
#EuroX = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet4', usecols = 'X', dtype=float).to_numpy().flatten()
#EuroY = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet4', usecols = 'X', dtype=float).to_numpy().flatten()
#theta = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet4', usecols = 'X', dtype=float).to_numpy().flatten()
#scale = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet4', usecols = 'X', dtype=float).to_numpy().flatten()
#flip = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet4', usecols = 'X', dtype=float).to_numpy().flatten()
#JupErrX = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet4', usecols = 'X', dtype=float).to_numpy().flatten()
#JupErrY = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet4', usecols = 'X', dtype=float).to_numpy().flatten()
#Time_mins = pd.read_excel(r'c:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025.xlsx', sheet_name='Sheet4', usecols = 'X', dtype=float).to_numpy().flatten()


def rotate_coords(Jx, Jy, x, y, theta, scale = scale, flip = flip):
    theta = theta*np.pi/180
    if flip == True:
        theta += np.pi
    J = np.vstack([Jx, Jy])
    r = np.vstack([x, y])
    A = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    r_prime = A@r
    J_prime = A@J

    r_prime = (r_prime - J_prime)*scale

    return r_prime

def data(Jx, Jy, x, y, theta, s = scale, f = flip):
    x_data, y_data = [], []
    for i in range(len(Jx)):
        x_prime, y_prime = rotate_coords(Jx[i], Jy[i], x[i], y[i], theta[i], s[i], f[i])
        x_data = np.append(x_data, x_prime)
        y_data = np.append(y_data, y_prime)
    return x_data, y_data




def chi_squared(model_params, model, x_data, y_data, y_err):
    return np.sum(((y_data - model(x_data, *model_params))/y_err)**2)


# Stack all relevant arrays to check for NaNs in any of them
data_stackC = np.column_stack([JupX, JupY, CalliX, CalliY, Time_mins])
data_stackG = np.column_stack([JupX, JupY, GanyX, GanyY, Time_mins])
data_stackE = np.column_stack([JupX, JupY, EuroX, EuroY, Time_mins])
data_stackI = np.column_stack([JupX, JupY, IoX, IoY, Time_mins])

# Create a mask that filters out any row with NaN values
valid_maskC = ~np.isnan(data_stackC).any(axis=1)
valid_maskG = ~np.isnan(data_stackG).any(axis=1)
valid_maskE = ~np.isnan(data_stackE).any(axis=1)
valid_maskI = ~np.isnan(data_stackI).any(axis=1)

# Apply mask to all arrays
JupXC, JupYC = JupX[valid_maskC], JupY[valid_maskC]
CalliX, CalliY = CalliX[valid_maskC], CalliY[valid_maskC]
Time_minsC = Time_mins[valid_maskC]  # Ensuring Time_mins aligns with filtered values

JupXG, JupYG = JupX[valid_maskG], JupY[valid_maskG]
GanyX, GanyY = GanyX[valid_maskG], GanyY[valid_maskG]
Time_minsG = Time_mins[valid_maskG]

#print(JupXG)
#print(GanyX)

JupXE, JupYE = JupX[valid_maskE], JupY[valid_maskE]
EuroX, EuroY = EuroX[valid_maskE], EuroY[valid_maskE]
Time_minsE = Time_mins[valid_maskE]

JupXI, JupYI = JupX[valid_maskI], JupY[valid_maskI]
IoX, IoY = IoX[valid_maskI], IoY[valid_maskI]
Time_minsI = Time_mins[valid_maskI]

###### Ganymede ######
x_dataC, y_dataC = data(JupXC, JupYC, CalliX, CalliY, theta)
x_dataG, y_dataG = data(JupXG, JupYG, GanyX, GanyY, theta)
x_dataE, y_dataE = data(JupXE, JupYE, EuroX, EuroY, theta)
x_dataI, y_dataI = data(JupXI, JupYI, IoX, IoY, theta)


def sinusoidI(t, T, c):
    return 150*np.sin((t+c)*(2*np.pi)/T)

def sinusoidE(t, T, c):
    return (max(x_dataE)+5)*np.sin((t+c)*(2*np.pi)/T)

def sinusoidG(t, T, c):
    return (max(x_dataG))*np.sin((t+c)*(2*np.pi)/T)

def sinusoidC(t, T, c):
    return 500*np.sin((t+c)*(2*np.pi)/T)


def modelI(x, *params):
    return sinusoidI(x, params[0], params[1], params[2])

def modelG(x, *params):
    return sinusoidG(x, params[0], params[1])

def modelE(x, *params):
    return sinusoidE(x, params[0], params[1])

def modelC(x, *params):
    return sinusoidC(x, params[0], params[1], params[2])



#print(x_data)
initial_valuesIx = [30, 2, 0]
initial_valuesIy = [15, 2, 0]

initial_valuesEx = [3.5, 0]
initial_valuesEy = [40, 3.5, 0]

initial_valuesGx = [7, 0]
initial_valuesGy = [0, 7, 0]

initial_valuesCx = [80, 16.6, 0]
initial_valuesCy = [50, 16.6, 0]
################

def CurveFit(x, initialx, Time):
    popt, cov = opt.curve_fit(modelG,Time/(24*60), x, sigma = np.ones(len(x))*0.001, absolute_sigma=True, p0=initialx, check_finite=True, maxfev=50000)
    #popt2, cov2 = opt.curve_fit(model,Time/(24*60), y, absolute_sigma=True, p0=initialy, check_finite=True, maxfev=50000)
    return popt, cov

print(CurveFit(x_dataG, initial_valuesGx, Time_minsG)[0])

#popt, cov = opt.curve_fit(model,Time_mins/(24*60), x_data, absolute_sigma=True, p0=initial_valuesx, check_finite=True, maxfev=50000)


#print('Optimised parameters = ', popt, '\n')
#print('Covariance matrix = \n', cov)

chi_squared_min = chi_squared(CurveFit(x_dataG, initial_valuesGx, Time_minsG)[0], modelG, x_dataG, Time_minsG, 10)
degrees_of_freedom = len(x_dataG) - len(CurveFit(x_dataG, initial_valuesGx, Time_minsG)[0])

print('chi^2_min = {}'.format(chi_squared_min))
print('reduced chi^2 = {}'.format(chi_squared_min/degrees_of_freedom))


smooth_time = np.linspace(0, 7*9.5, 100000)

plt.plot(Time_minsG/(24*60), x_dataG, 'o', label = 'Data')
plt.plot(smooth_time, modelG(smooth_time, *CurveFit(x_dataG, initial_valuesGx, Time_minsG)[0]), label = 'Fit')
plt.show()



#########Jackknife##########
from scipy.special import erfinv

#JKx, JKy, JKP = astats.jackknife_resampling(x_dataE), astats.jackknife_resampling(y_dataE), astats.jackknife_resampling(Time_mins)

'''
stat = CurveFit(x_dataE, y_dataE, initial_valuesCx, initial_valuesEy, Time_minsE)[0][1]
n = x_dataE.shape[0]

#for i in range(n):
#        t = opt.curve_fit(model,JKP[i]/(24*60), JKx[i], absolute_sigma=True, p0=initial_valuesEx, check_finite=True, maxfev=50000)[0]
#        print(t)

n = x_data.shape[0]
t = np.zeros(len(JKx))
for i in range(n):
    p = opt.curve_fit(model,JKP[i]/(24*60), JKx[i], absolute_sigma=True, p0=initial_valuesCx, check_finite=True, maxfev=50000)[0][1]
    t[i] = p

#print(t)
mean = np.mean(t)
bias = (n-1)*(mean - stat)
std = np.sqrt((n-1)*np.mean((t - mean)*(t - mean), axis=0))
estimate = stat - bias
z_score = np.sqrt(2.0) * erfinv(0.5)
conf_interval = estimate + z_score * np.array((-std, std))

print((estimate))
print((bias))
print((std))
print((conf_interval))
'''
    




def JackKnife(x, confidence_level, Time, initialx):
    JKx, JKP = astats.jackknife_resampling(x), astats.jackknife_resampling(Time)
    n = x.shape[0]
    t = np.zeros(len(JKx))
    for i in range(n):
        p = opt.curve_fit(modelG,JKP[i]/(24*60), JKx[i], sigma = np.ones(len(JKx[i])) ,absolute_sigma=True, p0=initialx, check_finite=True, maxfev=50000)[0][1]
        t[i] = p

    stat = CurveFit(x, initialx, Time)[0][1]
    mean = np.mean(t)
    bias = (n-1)*(mean - stat)
    std = np.sqrt((n-1)*np.mean((t - mean)*(t - mean), axis=0))
    estimate = stat - bias
    z_score = np.sqrt(2.0) * erfinv(confidence_level)
    conf_interval = estimate + z_score * np.array((-std, std))
    
    return estimate, bias, std, conf_interval

print(JackKnife(x_dataG, 0.8, Time_minsG, initial_valuesGx,)) 




'''
def Period(Position, initial_values):
    f = astats.jackknife_resampling(Position)
    
    for i in range(len(JKP)):
        def T(Position):
            popt, _ = opt.curve_fit(model, JKP[i]/(24*60), f[i], 
                                absolute_sigma=True, p0=initial_values, check_finite=True, maxfev=50000)
            return popt[1]
    estimate, bias, stderr, conf_interval = astats.jackknife_stats(Position, T, 0.95)
    
        
    return estimate, bias, stderr, conf_interval

'''
#print(Period(x_data, initial_valuesEx)) 
#print(Period(y_data, initial_valuesEy))
#print(CurveFit(x_data, y_data, initial_valuesEx, initial_valuesEy)[0])




#Specifically for x
#for i in range(len(JKP)):
#    def Period2(Position):
#        popt, _ = opt.curve_fit(model, JKP[i]/(24*60), JKx[i], 
#                                absolute_sigma=True, p0=initial_valuesEx, check_finite=True, maxfev=50000)
#        return popt[1]
#    
#    estimate, bias, stderr, conf_interval = astats.jackknife_stats(x_data, Period2, 0.95)

#print('Jackknife Estimate = ', estimate)
#print('Bias = ', bias)
#print('Standard Error = ', stderr)
#print('Confidence Interval = ', conf_interval)

###########################




#########Boosstrap##########

import numpy as np
from itertools import combinations, permutations, product
from collections.abc import Sequence
from dataclasses import dataclass, field
import inspect

from scipy._lib._util import (rng_integers)
#from scipy._lib._array_api import array_namespace, is_numpy, xp_moveaxis_to_end
#from scipy.special import ndtr, ndtri, comb, factorial


'''
#gives a random sample of data points 
def bootstrap_resample(sample, n_resamples=None, rng=None):
    """Bootstrap resample the sample."""
    n = sample.shape[-1]

    i = rng_integers(rng, 0, n, (n_resamples, n))

    resamples = sample[..., i]
    return resamples


#print(bootstrap_resample(x_data, len(x_data)))



def Shuffle(x, y, n_resamples, rng=None):
    c = np.array(zip(x, y))
    #n = c.shape[-1]
    i = rng_integers(rng, 0, len(x_data), (n_resamples, len(x_data)))

    resamples = c[..., i]
    x_data, y_data = resamples
    return x_data, y_data

#print(x_data, y_data)

print(Shuffle(x_data, y_data, len(x_data)))
'''



#def bootstrap(data, statistic, n_resamples):
#    for i in range(len(data)):




#t = scipy.stats.bootstrap(datas, opt.curve_fit(model,Time_mins/(24*60), x_data, sigma = 10000, absolute_sigma=True, p0=initial_valuesx, check_finite=True, maxfev=50000), n_resamples=9999, batch=None, vectorized=None, paired=False, 
#                          axis=0, confidence_level=0.95, alternative='two-sided', method='BCa', bootstrap_result=None)
#print(t)

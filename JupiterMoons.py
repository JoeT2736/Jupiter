import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
import pandas as pd
import astropy.stats as astats
import scipy.stats
import seaborn as sns

# Read Data with NaN Handling
JupXI = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='IoData', usecols = 'A', dtype=float).to_numpy().flatten()
JupYI = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='IoData', usecols = 'B', dtype=float).to_numpy().flatten()
IoX = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='IoData', usecols = 'C', dtype=float).to_numpy().flatten()
IoY = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='IoData', usecols = 'D', dtype=float).to_numpy().flatten()
thetaI = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='IoData', usecols = 'E', dtype=float).to_numpy().flatten()
scaleI = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='IoData', usecols = 'F', dtype=float).to_numpy().flatten()
flipI = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='IoData', usecols = 'G', dtype=bool).to_numpy().flatten()
JupErrXI = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='IoData', usecols = 'H', dtype=float).to_numpy().flatten()
JupErrYI = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='IoData', usecols = 'I', dtype=float).to_numpy().flatten()
Time_minsI = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='IoData', usecols = 'N', dtype=float).to_numpy().flatten()


JupXE = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='EuroData', usecols = 'A', dtype=float).to_numpy().flatten()
JupYE = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='EuroData', usecols = 'B', dtype=float).to_numpy().flatten()
EuroX = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='EuroData', usecols = 'C', dtype=float).to_numpy().flatten()
EuroY = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='EuroData', usecols = 'D', dtype=float).to_numpy().flatten()
thetaE = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='EuroData', usecols = 'E', dtype=float).to_numpy().flatten()
scaleE = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='EuroData', usecols = 'F', dtype=float).to_numpy().flatten()
flipE = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='EuroData', usecols = 'G', dtype=bool).to_numpy().flatten()
JupErrXE = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='EuroData', usecols = 'H', dtype=float).to_numpy().flatten()
JupErrYE = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='EuroData', usecols = 'I', dtype=float).to_numpy().flatten()
Time_minsE = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='EuroData', usecols = 'N', dtype=float).to_numpy().flatten()



JupXG = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='GanyData', usecols = 'A', dtype=float).to_numpy().flatten()
JupYG = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='GanyData', usecols = 'B', dtype=float).to_numpy().flatten()
GanyX = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='GanyData', usecols = 'C', dtype=float).to_numpy().flatten()
GanyY = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='GanyData', usecols = 'D', dtype=float).to_numpy().flatten()
thetaG = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='GanyData', usecols = 'E', dtype=float).to_numpy().flatten()
scaleG = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='GanyData', usecols = 'F', dtype=float).to_numpy().flatten()
flipG = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='GanyData', usecols = 'G', dtype=bool).to_numpy().flatten()
JupErrXG = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='GanyData', usecols = 'H', dtype=float).to_numpy().flatten()
JupErrYG = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='GanyData', usecols = 'I', dtype=float).to_numpy().flatten()
Time_minsG = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='GanyData', usecols = 'N', dtype=float).to_numpy().flatten()




JupXC = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='CalliData', usecols = 'A', dtype=float).to_numpy().flatten()
JupYC = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='CalliData', usecols = 'B', dtype=float).to_numpy().flatten()
CalliX = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='CalliData', usecols = 'C', dtype=float).to_numpy().flatten()
CalliY = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='CalliData', usecols = 'D', dtype=float).to_numpy().flatten()
thetaC = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='CalliData', usecols = 'E', dtype=float).to_numpy().flatten()
scaleC = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='CalliData', usecols = 'F', dtype=float).to_numpy().flatten()
flipC = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='CalliData', usecols = 'G', dtype=bool).to_numpy().flatten()
JupErrXC = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='CalliData', usecols = 'H', dtype=float).to_numpy().flatten()
JupErrYC = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='CalliData', usecols = 'I', dtype=float).to_numpy().flatten()
Time_minsC = pd.read_excel(r'C:\Users\PHYUG.MDS\Downloads\Jupiter Data 24 01 2025(2).xlsx', sheet_name='CalliData', usecols = 'N', dtype=float).to_numpy().flatten()



Data_stackI = np.column_stack([JupXI, JupYI, IoX, IoY, Time_minsI])
Data_stackE = np.column_stack([JupXE, JupYE, EuroX, EuroY, Time_minsE])
Data_stackG = np.column_stack([JupXG, JupYG, GanyX, GanyY, Time_minsG])
Data_stackC = np.column_stack([JupXC, JupYC, CalliX, CalliY, Time_minsC])

'''
#Scaled Jupiter centre error
def JupErr(Xerr, Yerr, scale):
    X, Y = [], []
    for i in range(len(Xerr)):
        Xe = X[i] * scale
        Ye = Y[i] * scale
        X.append(Xe)
        Y.append(Ye)

    return np.array(X), np.array(Y)
'''




# Rotation Function
def rotate_coords(Jx, Jy, x, y, theta, scale, flip):
    theta = theta * np.pi / 180
    if flip:
        theta += np.pi
    A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    J = np.vstack([Jx, Jy])
    r = np.vstack([x, y])
    r_prime = A @ r
    J_prime = A @ J
    return (r_prime - J_prime) * scale
'''
def rotate_coords_JupCentre(Jx, Jy, theta, scale, flip):
    theta = theta * np.pi / 180
    if flip:
        theta += np.pi
    A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    J = np.vstack([Jx, Jy])
    J_prime = A @ J
    return J_prime * scale
'''
'''
def Jupiter_centre(Jx, Jy, theta, scale, flip):
    x_data, y_data = [], []
    for i in range(len(Jx)):
        x_prime, y_prime = rotate_coords(Jx[i], Jy[i], theta[i], scale[i], flip[i])
        x_data.append(x_prime)
        y_data.append(y_prime)
    return np.array(x_data), np.array(y_data)
'''
# Data Processing Function
def data(Jx, Jy, x, y, theta, scale, flip):
    x_data, y_data = [], []
    for i in range(len(Jx)):
        x_prime, y_prime = rotate_coords(Jx[i], Jy[i], x[i], y[i], theta[i], scale[i], flip[i])
        x_data.append(x_prime)
        y_data.append(y_prime)
    return np.array(x_data), np.array(y_data)

# Sinusoidal Model


def sinusoid(t, A, T, c):
    return A*np.sin((t + c) * (2 * np.pi) / T)

def model(x, *params):
    return sinusoid(x, params[0], params[1], params[2])





# Curve Fitting Function
def CurveFit(model, x, initial_values, Time):
        popt, cov = opt.curve_fit(
            model, Time/(24*60), x, sigma=np.ones(len(x))*0.5, 
            absolute_sigma=True, p0=initial_values, check_finite=True, maxfev=50000
        )
        return popt, cov



# Process Data
x_dataI, y_dataI = data(JupXI, JupYI, IoX, IoY, thetaI, scaleI, flipI)
x_dataE, y_dataE = data(JupXE, JupYE, EuroX, EuroY, thetaE, scaleE, flipE)
x_dataG, y_dataG = data(JupXG, JupYG, GanyX, GanyY, thetaG, scaleG, flipG)
x_dataC , y_dataC = data(JupXC, JupYC, CalliX, CalliY, thetaC, scaleC, flipC)

'''
x_JCI, y_JCI = Jupiter_centre(JupXI, JupYI, thetaI, scaleI, flipI)
x_JCE, y_JCE = Jupiter_centre(JupXE, JupYE, thetaE, scaleE, flipE)
x_JCG, y_JCG = Jupiter_centre(JupXG, JupYG, thetaG, scaleG, flipG)
x_JCC, y_JCC = Jupiter_centre(JupXC, JupYC, thetaC, scaleC, flipC)
'''

#print('Io data', x_dataI, y_dataI, '/n')
#print('Euro data', x_dataE, y_dataE, '/n')
#print('Ganymede data', x_dataG, y_dataG, '/n')
#print('Callisto data', x_dataC, y_dataC, '/n')


# Initial values

#initial_valuesIx = [130, 1.769, 0]
#initial_valuesEx = [200, 3.551, 0]
#initial_valuesGx = [280, 7.154, 0]
#initial_valuesCx = [500, 16.689, 0]


initial_valuesIx = [1.769, 0]
initial_valuesEx = [3.551, 0]
initial_valuesGx = [7.154, 0]
initial_valuesCx = [16.689, 0]

#initial_valuesIx = [1, 0]
#initial_valuesEx = [2.5, 0]
#initial_valuesGx = [5, 0]
#initial_valuesCx = [14, 0]




x_dataI = x_dataI.flatten()
y_dataI = y_dataI.flatten()
Time_minsI = Time_minsI.flatten()

x_dataE = x_dataE.flatten()
y_dataE = y_dataE.flatten()
Time_minsE = Time_minsE.flatten()

x_dataG = x_dataG.flatten()
y_dataG = y_dataG.flatten()
Time_minsG = Time_minsG.flatten()

x_dataC = x_dataC.flatten()
y_dataC= y_dataC.flatten()
Time_minsC = Time_minsC.flatten()



def sinusoidI(t, T, c):
    return (max(abs(x_dataI))+5)*np.sin((t + c) * (2 * np.pi) / T)

def modelI(x, *params):
    return sinusoidI(x, params[0], params[1])



def sinusoidE(t, T, c):
    return (max(abs(x_dataE))+5)*np.sin((t + c) * (2 * np.pi) / T)

def modelE(x, *params):
    return sinusoidE(x, params[0], params[1])



def sinusoidG(t, T, c):
    return (max(abs(x_dataG))+5)*np.sin((t + c) * (2 * np.pi) / T)

def modelG(x, *params):
    return sinusoidG(x, params[0], params[1])



def sinusoidC(t, T, c):
    return (max(abs(x_dataC))+20)*np.sin((t + c) * (2 * np.pi) / T)

def modelC(x, *params):
    return sinusoidC(x, params[0], params[1])







#print(x_dataI.shape, Time_minsI.shape)
#print(x_dataE.shape, Time_minsE.shape)
#print(x_dataG.shape, Time_minsG.shape)
#print(x_dataC.shape, Time_minsC.shape)



# Perform Curve Fitting
#with set amplitude in sinusoid function
poptI, covI = CurveFit(modelI, x_dataI, initial_valuesIx, Time_minsI)
poptE, covE = CurveFit(modelE, x_dataE, initial_valuesEx, Time_minsE)
poptG, covG = CurveFit(modelG, x_dataG, initial_valuesGx, Time_minsG)
poptC, covC = CurveFit(modelC, x_dataC, initial_valuesCx, Time_minsC)

#amplitude in intial values
#poptI, covI = CurveFit(model, x_dataI, initial_valuesIx, Time_minsI)
#poptE, covE = CurveFit(model, x_dataE, initial_valuesEx, Time_minsE)
#poptG, covG = CurveFit(model, x_dataG, initial_valuesGx, Time_minsG)
#poptC, covC = CurveFit(model, x_dataC, initial_valuesCx, Time_minsC)

print('Optimised parameters for Io = ', poptI, '\n')

print('Optimised parameters for Europa = ', poptE, '\n')

print('Optimised parameters for Ganymede = ', poptG, '\n')

print('Optimised parameters for Callisto = ', poptC, '\n')



def chi_squared(model_params, model, x_data, y_data, y_err):
    return np.sum(((y_data - model(x_data, *model_params)) / y_err) ** 2)

chi_squared_min = chi_squared(poptG, modelG, x_dataG, Time_minsG, 10)
degrees_of_freedom = len(x_dataG) - len(poptG)

#print(f'chi^2_min Ganymede = {chi_squared_min}')
#print(f'reduced chi^2 Ganymede = {chi_squared_min / degrees_of_freedom}')

# Plot
#smooth_time = np.linspace(0, 7*9.5, 100000)

def smooth_time(t):
    return np.linspace(min(t), max(t), 100000)/(24*60)



def residuals(position, model):
    return -abs(position) + abs(model)

print(modelI(Time_minsI, *poptI))
print(x_dataI)
print(residuals(x_dataI, modelI(Time_minsI, *poptI)))

def normal_residuals(position, model):
    return residuals(position, model)/max(residuals(position, model))


from matplotlib import gridspec

plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

ax1 = plt.subplot(gs[0])
ax1.errorbar(Time_minsI/(24*60), x_dataI, yerr=0.5, marker='o', label='Data', ls='none')
ax1.plot(smooth_time(Time_minsI), modelI(smooth_time(Time_minsI), *poptI), label='Fit')
plt.setp(ax1.get_xticklabels(), visible='False')
#ax1.set_xlabel('Arcsec')
ax1.set_ylabel('Arcsec')
ax1.set_title('Io')
ax1.legend()
ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.axhline(y=0, color='black', linestyle='--')
ax2.errorbar(Time_minsI/(24*60), normal_residuals(x_dataI, modelI(Time_minsI, *poptI)), marker='o', linestyle='None')
ax2.set_xlabel('Time (days)')
plt.show()



plt.figure(figsize=(6, 6))
plt.errorbar(Time_minsE / (24 * 60), x_dataE, yerr=0.5, marker='o', label='Data', ls='none')
plt.plot(smooth_time(Time_minsE), modelE(smooth_time(Time_minsE), *poptE), label='Fit')
plt.title('Europa')
plt.xlabel('Time (days)')
plt.ylabel('Arcsec')
plt.legend()

plt.figure(figsize=(6, 6))
plt.errorbar(Time_minsG / (24 * 60), x_dataG, yerr=0.5, marker='o', label='Data', ls='none')
plt.plot(smooth_time(Time_minsG), modelG(smooth_time(Time_minsG), *poptG), label='Fit')
plt.title('Ganymede')
plt.xlabel('Time (days)')
plt.ylabel('Arcsec')
plt.legend()

plt.figure(figsize=(6, 6))
plt.errorbar(Time_minsC / (24 * 60), x_dataC, yerr=0.5, marker='o', label='Data', ls='none')
plt.plot(smooth_time(Time_minsC), modelC(smooth_time(Time_minsC), *poptC), label='Fit')
plt.title('Callisto')
plt.xlabel('Time (days)')
plt.ylabel('Arcsec')
plt.legend()


#plt.show()







plt.figure(figsize=(6, 6))
plt.xlim(-175, 175)
plt.ylim(-30, 30)
plt.errorbar(x_dataI, y_dataI, xerr=0.5, yerr=0.5, marker='o', ls='none')
plt.errorbar(0, 0, xerr=0.5, yerr=0.5, marker='o', color='red')
plt.xlabel('Arcsec')
plt.ylabel('Arcsec')
plt.title('Io')

plt.figure(figsize=(6, 6))
plt.xlim(-250, 250)
plt.ylim(-50, 50)
plt.errorbar(x_dataE, y_dataE, xerr=0.5, yerr=0.5, marker='o', ls='none')
plt.errorbar(0, 0, xerr=0.5, yerr=0.5, marker='o', color='red')
plt.xlabel('Arcsec')
plt.ylabel('Arcsec')
plt.title('Europa')

plt.figure(figsize=(6, 6))
plt.xlim(-350, 350)
plt.ylim(-60, 60)
plt.errorbar(x_dataG, y_dataG, xerr=0.5, yerr=0.5, marker='o', ls='none')
plt.errorbar(0, 0, xerr=0.5, yerr=0.5, marker='o', color='red')
plt.xlabel('Arcsec')
plt.ylabel('Arcsec')
plt.title('Ganymede')

plt.figure(figsize=(6, 6))
plt.xlim(-500, 500)
plt.ylim(-125, 125)
plt.errorbar(x_dataC, y_dataC, xerr=0.5, yerr=0.5, marker='o', ls='none')
plt.errorbar(0, 0, xerr=0.5, yerr=0.5, marker='o', color='red')
plt.xlabel('Arcsec')
plt.ylabel('Arcsec')
plt.title('Callisto')



####Plot of orbit of moons, radius is max values of amplitude from curvefit function (first value)

thetas = np.linspace( 0 , 2 * np.pi , 150 )
 
aI = poptI[0] * np.cos( thetas )
bI = poptI[0] * np.sin( thetas )

aE = poptE[0] * np.cos( thetas )
bE = poptE[0] * np.sin( thetas )

aG = poptG[0] * np.cos( thetas )
bG = poptG[0] * np.sin( thetas )

aC = poptC[0] * np.cos( thetas )
bC = poptC[0] * np.sin( thetas )
 
figure, axes = plt.subplots( 1 )
 
axes.plot(aI, bI, label='Io' )
axes.plot(aE, bE, label='Europa')
axes.plot(aG, bG, label='Ganymede')
axes.plot(aC, bC, label='Callisto')
axes.plot(0,0, color='red')
axes.set_aspect( 1 )
plt.legend()

#plt.show()




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(aI, bI, np.zeros(len(thetas)),label='Io')
ax.plot(aE, bE, np.zeros(len(thetas)),label='Europa')
ax.plot(aG, bG, np.zeros(len(thetas)),label='Ganymede')
ax.plot(aC, bC, np.zeros(len(thetas)),label='Callisto')
ax.set_xlabel('(Arcsec)')
ax.set_ylabel('(Arcsec)')
#ax.plot(0, 0, 0, color='red')
#ax.set_aspect(1)


plt.show()

















#########Jackknife##########
from scipy.special import erfinv
'''
JKx, JKy, JKP = astats.jackknife_resampling(x_dataC), astats.jackknife_resampling(y_dataC), astats.jackknife_resampling(Time_minsC)


stat = CurveFit(modelC, x_dataC, initial_valuesCx, Time_minsC)[0][0]
#print(stat)


#for i in range(n):
#        t = opt.curve_fit(modelE,JKP[i]/(24*60), JKx[i], absolute_sigma=True, p0=initial_valuesEx, check_finite=True, maxfev=50000)[0]
#        print(t)

n = x_dataC.shape[0]
t = np.zeros(len(JKx))
for i in range(n):
    popt, _ = opt.curve_fit(modelC, JKx[i], JKP[i]/(24*60), p0=initial_valuesCx, absolute_sigma=True, check_finite=True, maxfev=50000)
    #print(popt)
    t[i] = popt[0]

print(t)
mean = np.mean(t)
print(mean)
print(stat)
bias = (n-1)*(mean - stat)
std = np.sqrt((n-1)*np.mean((t - mean)*(t - mean), axis=0))
estimate = stat - bias
z_score = np.sqrt(2.0) * erfinv(0.8)
conf_interval = estimate + z_score * np.array((-std, std))

print((estimate))
print((bias))
print((std))
print((conf_interval))
'''
    




def JackKnife(model, x, confidence_level, Time, initialx):
    JKx, JKP = astats.jackknife_resampling(x), astats.jackknife_resampling(Time)
    n = x.shape[0]
    t = np.zeros(len(JKx))
    for i in range(n):
        popt, _ = opt.curve_fit(model, JKx[i], JKP[i]/(24*60), sigma = np.ones(len(JKx[i])), 
                                p0=initialx, absolute_sigma=True, check_finite=True, maxfev=50000)
        t[i] = popt[0]

    stat = CurveFit(model, x, initialx, Time)[0][0]
    mean = np.mean(t)
    bias = (n-1)*(mean - stat)
    std = np.sqrt((n-1)*np.mean((t - mean)*(t - mean), axis=0))
    estimate = stat - bias
    z_score = np.sqrt(2.0) * erfinv(confidence_level)
    conf_interval = estimate + z_score * np.array((-std, std))
    
    return estimate, mean, bias, std, conf_interval


print('Io Jackknife', JackKnife(modelI, x_dataI, 0.8, Time_minsI, initial_valuesIx), '\n')
print('Europa Jackknife:', JackKnife(modelE, x_dataE, 0.8, Time_minsE, initial_valuesEx), '\n')
print('Ganymede Jackknife:', JackKnife(modelG, x_dataG, 0.8, Time_minsG, initial_valuesGx), '\n')
print('Callisto Jackknife:', JackKnife(modelC, x_dataC, 0.8, Time_minsC, initial_valuesCx), '\n')
 




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



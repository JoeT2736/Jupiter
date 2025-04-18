import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
import pandas as pd
import astropy.stats as astats
import scipy.stats
#import seaborn as sns

# Read Data with NaN Handling
JupXI = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='IoData', usecols = 'A', dtype=float).to_numpy().flatten()
JupYI = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='IoData', usecols = 'B', dtype=float).to_numpy().flatten()
IoX = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='IoData', usecols = 'C', dtype=float).to_numpy().flatten()
IoY = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='IoData', usecols = 'D', dtype=float).to_numpy().flatten()
thetaI = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='IoData', usecols = 'E', dtype=float).to_numpy().flatten()
scaleI = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='IoData', usecols = 'F', dtype=float).to_numpy().flatten()
flipI = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='IoData', usecols = 'G', dtype=bool).to_numpy().flatten()
JupErrXI = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='IoData', usecols = 'H', dtype=float).to_numpy().flatten()
JupErrYI = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='IoData', usecols = 'I', dtype=float).to_numpy().flatten()
Time_minsI = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='IoData', usecols = 'N', dtype=float).to_numpy().flatten()


JupXE = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='EuroData', usecols = 'A', dtype=float).to_numpy().flatten()
JupYE = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='EuroData', usecols = 'B', dtype=float).to_numpy().flatten()
EuroX = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='EuroData', usecols = 'C', dtype=float).to_numpy().flatten()
EuroY = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='EuroData', usecols = 'D', dtype=float).to_numpy().flatten()
thetaE = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='EuroData', usecols = 'E', dtype=float).to_numpy().flatten()
scaleE = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='EuroData', usecols = 'F', dtype=float).to_numpy().flatten()
flipE = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='EuroData', usecols = 'G', dtype=bool).to_numpy().flatten()
JupErrXE = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='EuroData', usecols = 'H', dtype=float).to_numpy().flatten()
JupErrYE = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='EuroData', usecols = 'I', dtype=float).to_numpy().flatten()
Time_minsE = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='EuroData', usecols = 'N', dtype=float).to_numpy().flatten()



JupXG = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='GanyData', usecols = 'A', dtype=float).to_numpy().flatten()
JupYG = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='GanyData', usecols = 'B', dtype=float).to_numpy().flatten()
GanyX = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='GanyData', usecols = 'C', dtype=float).to_numpy().flatten()
GanyY = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='GanyData', usecols = 'D', dtype=float).to_numpy().flatten()
thetaG = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='GanyData', usecols = 'E', dtype=float).to_numpy().flatten()
scaleG = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='GanyData', usecols = 'F', dtype=float).to_numpy().flatten()
flipG = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='GanyData', usecols = 'G', dtype=bool).to_numpy().flatten()
JupErrXG = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='GanyData', usecols = 'H', dtype=float).to_numpy().flatten()
JupErrYG = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='GanyData', usecols = 'I', dtype=float).to_numpy().flatten()
Time_minsG = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='GanyData', usecols = 'N', dtype=float).to_numpy().flatten()




JupXC = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='CalliData', usecols = 'A', dtype=float).to_numpy().flatten()
JupYC = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='CalliData', usecols = 'B', dtype=float).to_numpy().flatten()
CalliX = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='CalliData', usecols = 'C', dtype=float).to_numpy().flatten()
CalliY = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='CalliData', usecols = 'D', dtype=float).to_numpy().flatten()
thetaC = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='CalliData', usecols = 'E', dtype=float).to_numpy().flatten()
scaleC = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='CalliData', usecols = 'F', dtype=float).to_numpy().flatten()
flipC = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='CalliData', usecols = 'G', dtype=bool).to_numpy().flatten()
JupErrXC = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='CalliData', usecols = 'H', dtype=float).to_numpy().flatten()
JupErrYC = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='CalliData', usecols = 'I', dtype=float).to_numpy().flatten()
Time_minsC = pd.read_excel(r'F:\Lab Reports\Jupiter Data 24 01 2025.xlsx', sheet_name='CalliData', usecols = 'N', dtype=float).to_numpy().flatten()





'''

Data_stackI = np.column_stack([JupXI, JupYI, IoX, IoY, Time_minsI])
Data_stackE = np.column_stack([JupXE, JupYE, EuroX, EuroY, Time_minsE])
Data_stackG = np.column_stack([JupXG, JupYG, GanyX, GanyY, Time_minsG])
Data_stackC = np.column_stack([JupXC, JupYC, CalliX, CalliY, Time_minsC])


#Scaled Jupiter centre error
def JupErr(Xerr, Yerr, scale):
    X, Y = [], []
    for i in range(len(Xerr)):
        Xe = X[i] * scale
        Ye = Y[i] * scale
        X.append(Xe)
        Y.append(Ye)

    return np.array(X), np.array(Y)





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

def rotate_coords_JupCentre(Jx, Jy, theta, scale, flip):
    theta = theta * np.pi / 180
    if flip:
        theta += np.pi
    A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    J = np.vstack([Jx, Jy])
    J_prime = A @ J
    return J_prime * scale


def Jupiter_centre(Jx, Jy, theta, scale, flip):
    x_data, y_data = [], []
    for i in range(len(Jx)):
        x_prime, y_prime = rotate_coords(Jx[i], Jy[i], theta[i], scale[i], flip[i])
        x_data.append(x_prime)
        y_data.append(y_prime)
    return np.array(x_data), np.array(y_data)

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


x_JCI, y_JCI = Jupiter_centre(JupXI, JupYI, thetaI, scaleI, flipI)
x_JCE, y_JCE = Jupiter_centre(JupXE, JupYE, thetaE, scaleE, flipE)
x_JCG, y_JCG = Jupiter_centre(JupXG, JupYG, thetaG, scaleG, flipG)
x_JCC, y_JCC = Jupiter_centre(JupXC, JupYC, thetaC, scaleC, flipC)


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
#plt.show()



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


plt.show()
'''

 





















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

#print(rotate_coords(JupXG, JupYG, GanyX, GanyY, thetaG, scaleG, flipG))



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

#print(data(JupXG, JupYG, GanyX, GanyY, thetaG, scaleG, flipG))

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

def sinusoidIy(t, T, c):
    return (max(abs(y_dataI))+5)*np.sin((t + c) * (2 * np.pi) / T)

def modelIy(x, *params):
    return sinusoidIy(x, params[0], params[1])



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
#poptIy, covIy = CurveFit(modelIy, y_dataI, initial_valuesIx, Time_minsI)

poptE, covE = CurveFit(modelE, x_dataE, initial_valuesEx, Time_minsE)
poptG, covG = CurveFit(modelG, x_dataG, initial_valuesGx, Time_minsG)
poptC, covC = CurveFit(modelC, x_dataC, initial_valuesCx, Time_minsC)


'''
#amplitude in intial values
initial_valuesIx = [10, 3, 0]
poptI2, covI2 = CurveFit(model, x_dataI, initial_valuesIx, Time_minsI)
initial_valuesEx =[20, 5, 0]
poptE2, covE2 = CurveFit(model, x_dataE, initial_valuesEx, Time_minsE)
initial_valuesGx = [25, 7, 0]
poptG2, covG2 = CurveFit(model, x_dataG, initial_valuesGx, Time_minsG)
initial_valuesCx = [30, 13, 0]
poptC2, covC2 = CurveFit(model, x_dataC, initial_valuesCx, Time_minsC)

print('Optimised parameters for Io = ', poptIx, '\n')
print('Optimised parameters for Io = ', poptI2, '\n')

print('Optimised parameters for Europa = ', poptE, '\n')
print('Optimised parameters for Europa = ', poptE2, '\n')

print('Optimised parameters for Ganymede = ', poptG, '\n')
print('Optimised parameters for Ganymede = ', poptG2, '\n')

print('Optimised parameters for Callisto = ', poptC, '\n')
print('Optimised parameters for Callisto = ', poptC2, '\n')
'''


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

#print(modelI(Time_minsI, *poptI))
#print(x_dataI)
#print(residuals(x_dataI, modelI(Time_minsI, *poptI)))

def normal_residuals(position, model):
    return residuals(position, model)/max(residuals(position, model))


from matplotlib import gridspec

'''
plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

ax1 = plt.subplot(gs[0])
ax1.errorbar(Time_minsI/(24*60), x_dataI, yerr=0.5, marker='o', label='Data', ls='none')
ax1.plot(smooth_time(Time_minsI), modelI(smooth_time(Time_minsI), *poptI), label='Fit')
#plt.setp(ax1.get_xticklabels(), visible='False')
#ax1.set_xlabel('Arcsec')
ax1.set_ylabel('Arcsec')
ax1.set_title('Io x')
ax1.legend()
#ax2 = plt.subplot(gs[1], sharex=ax1)
#ax2.axhline(y=0, color='black', linestyle='--')
#ax2.errorbar(Time_minsI/(24*60), normal_residuals(x_dataI, modelI(Time_minsI, *poptI)), marker='o', linestyle='None')
#ax2.set_xlabel('Time (days)')
plt.show()


#plt.figure(figsize=([6, 6]))
#gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)
#ax1 = plt.subplot(gs[0])
#ax1.errorbar(Time_minsI/(24*60), y_dataI, yerr=0.5, marker='o', label='Data', ls='none')
#ax1.plot(smooth_time(Time_minsI), modelIy(smooth_time(Time_minsI), *poptIy), label='Fit')
#ax1.set_title('Io y')




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
'''

#plt.show()






'''
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
'''


####Plot of orbit of moons, radius is max values of amplitude from curvefit function (first value)
'''
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
'''

#plt.show()














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
    




def ellipse_fit(x,y):
    # least squares fit of an ellipse using matrix eqn

    J = np.vstack([x**2, x*y, y**2, x, y]).T
    K = np.ones_like(x)
    JT = J.transpose()
    JTJ = np.dot(JT,J)
    invJTJ = np.linalg.inv(JTJ)
    vector = np.dot(invJTJ, np.dot(JT,K))

    return np.append(vector, -1)

def convert_to_physical(A, B, C, D, E, F):
    x0 = (2*C*D - B*E)/(B**2-4*A*C)
    y0 = (2*A*E - B*D)/(B**2-4*A*C)
    a = -( np.sqrt(2*(A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F)*((A+C)+(np.sqrt((A-C)**2 + B**2)))) )/(B**2-4*A*C) 
    b = -( np.sqrt(2*(A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F)*((A+C)-(np.sqrt((A-C)**2 + B**2)))) )/(B**2-4*A*C)
    theta = np.arctan2(-B, C-A)/2
    return x0, y0, a, b, theta

def model_ellipse(x, y, *params):
    return params[0]*x**2 + params[1]*x*y + params[2]*y**2 + params[3]*x + params[4]*y + params[5]

#print(ellipse_fit(x_dataG, y_dataG))

def JackKnife2D(model, x, y, confidence):
    JKx, JKy = astats.jackknife_resampling(x), astats.jackknife_resampling(y)
    n = x.shape[0] 
    A = np.zeros(len(JKx))
    B = np.zeros(len(JKx))
    C = np.zeros(len(JKx))
    D = np.zeros(len(JKx))
    E = np.zeros(len(JKx))
    F = np.zeros(len(JKx))

    for i in range(n):
        fit = ellipse_fit(JKx[i], JKy[i])
        A[i] = fit[0]
        B[i] = fit[1]
        C[i] = fit[2]
        D[i] = fit[3]
        E[i] = fit[4]
        F[i] = fit[5]

    statA = ellipse_fit(x, y)[0]
    statB = ellipse_fit(x, y)[1]
    statC = ellipse_fit(x, y)[2]
    statD = ellipse_fit(x, y)[3]
    statE = ellipse_fit(x, y)[4]
    statF = ellipse_fit(x, y)[5]

    meanA = np.mean(A)
    meanB = np.mean(B)
    meanC = np.mean(C)
    meanD = np.mean(D)
    meanE = np.mean(E)
    meanF = np.mean(F)

    biasA = (n-1)*(meanA - statA)
    biasB = (n-1)*(meanB - statB)
    biasC = (n-1)*(meanC - statC)
    biasD = (n-1)*(meanD - statD)
    biasE = (n-1)*(meanE - statE)
    biasF = (n-1)*(meanF - statF)

    stdA = np.sqrt((n-1)*np.mean((A - meanA)*(A - meanA), axis=0))
    stdB = np.sqrt((n-1)*np.mean((B - meanB)*(B - meanB), axis=0))
    stdC = np.sqrt((n-1)*np.mean((C - meanC)*(C - meanC), axis=0))
    stdD = np.sqrt((n-1)*np.mean((D - meanD)*(D - meanD), axis=0))
    stdE = np.sqrt((n-1)*np.mean((E - meanE)*(E - meanE), axis=0))
    stdF = np.sqrt((n-1)*np.mean((F - meanF)*(F - meanF), axis=0))

    estimateA = statA - biasA
    estimateB = statB - biasB
    estimateC = statC - biasC
    estimateD = statD - biasD
    estimateE = statE - biasE
    estimateF = statF - biasF

    z_score = np.sqrt(2.0) * erfinv(confidence)

    conf_intervalA = estimateA + z_score * np.array((-stdA, stdA))
    conf_intervalB = estimateB + z_score * np.array((-stdB, stdB))
    conf_intervalC = estimateC + z_score * np.array((-stdC, stdC))
    conf_intervalD = estimateD + z_score * np.array((-stdD, stdD))
    conf_intervalE = estimateE + z_score * np.array((-stdE, stdE))
    conf_intervalF = estimateF + z_score * np.array((-stdF, stdF))

    estimate = np.array([np.abs(estimateA), np.abs(estimateB), np.abs(estimateC), estimateD, estimateE, estimateF])

    return estimate

A = JackKnife2D(model_ellipse, x_dataG, y_dataG, 0.8)[0]
B = JackKnife2D(model_ellipse, x_dataG, y_dataG, 0.8)[1]
C = JackKnife2D(model_ellipse, x_dataG, y_dataG, 0.8)[2]
D = JackKnife2D(model_ellipse, x_dataG, y_dataG, 0.8)[3]
E = JackKnife2D(model_ellipse, x_dataG, y_dataG, 0.8)[4]
F = JackKnife2D(model_ellipse, x_dataG, y_dataG, 0.8)[5]

#print(ellipse_fit(x_dataG, y_dataG))
#print(JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8))

#-( np.sqrt(2*(A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F)*((A+C)+(np.sqrt((A-C)**2 + B**2)))) )/(B**2-4*A*C) 

'''
A = ellipse_fit(x_dataG, y_dataG)[0]
B = ellipse_fit(x_dataG, y_dataG)[1]
C = ellipse_fit(x_dataG, y_dataG)[2]
D = ellipse_fit(x_dataG, y_dataG)[3]
E = ellipse_fit(x_dataG, y_dataG)[4]
F = ellipse_fit(x_dataG, y_dataG)[5]
'''

a = A*E**2
b = C*D**2
c = B*D*E
d = B**2 - 4*A*C
e = np.sqrt((A-C)**2 + B**2)
f = B**2-4*A*C


#print(c)



'''
print(convert_to_physical(ellipse_fit(x_dataG, y_dataG)[0], ellipse_fit(x_dataG, y_dataG)[1], ellipse_fit(x_dataG, y_dataG)[2],
                         ellipse_fit(x_dataG, y_dataG)[3], ellipse_fit(x_dataG, y_dataG)[4],ellipse_fit(x_dataG, y_dataG)[5]))

print(convert_to_physical(JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[0], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[1],
                         JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[2], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[3],
                         JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[4], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[5]))
'''

'''
def ellipse_eq(x, y, x0, y0, a, b, theta):
    term1 = ((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta))**2 / a**2
    term2 = ((-(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta))**2) / b**2
    return term1 + term2 - 1  # ellipse is defined by ellipse_eq == 0

# Define the plotting grid.

N = 5000
x_min, x_max = -400, 400  # use names that don't conflict with x0_model, y0_model
y_min, y_max = -100, 100
xs = np.linspace(x_min, x_max, N)
ys = np.linspace(y_min, y_max, N)
X, Y = np.meshgrid(xs, ys)

# Evaluate the ellipse equation on the grid.
Z = ellipse_eq(X, Y, convert_to_physical(JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[0], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[1],
                            JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[2], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[3],
                            JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[4], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[5])[0],
                    convert_to_physical(JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[0], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[1],
                            JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[2], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[3],
                            JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[4], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[5])[1],
                    convert_to_physical(JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[0], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[1],
                            JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[2], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[3],
                            JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[4], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[5])[2],
                    convert_to_physical(JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[0], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[1],
                            JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[2], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[3],
                            JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[4], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[5])[3],
                    convert_to_physical(JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[0], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[1],
                            JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[2], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[3],
                            JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[4], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[5])[4])

# Plot the contour corresponding to the ellipse (where Z == 0).

#print(Z)

plt.figure(figsize=(6, 6))
#plt.scatter(x_dataG, x_dataG, color='black')
plt.errorbar(x_dataG, x_dataG, xerr = np.ones(len(x_dataG)), yerr = np.ones(len(x_dataG)), ls = 'none', color='black')
plt.contour(X, Y, Z)
plt.xlabel("X")
plt.ylabel("Y")
#plt.show()
'''

















################################ 1D Jackknife ########################################


initial_valuesIx = [130, 1.769, 0]
initial_valuesEx = [200, 3.551, 0]
initial_valuesGx = [300, 7.154, 0]
initial_valuesCx = [500, 16.689, 0]

initial_valuesIy = [20, 1.769, 0]
initial_valuesEy = [25, 3.551, 0]
initial_valuesGy = [40, 7.154, 0]
initial_valuesCy = [50, 16.689, 0]


def JackKnife(model, x, confidence_level, Time, initialx):
    JKx, JKP = astats.jackknife_resampling(x), astats.jackknife_resampling(Time)
    n = x.shape[0]
    Radius = np.zeros(len(JKx))
    Period = np.zeros(len(JKx))
    Start_angle = np.zeros(len(JKx))

    for i in range(n):
        popt, _ = opt.curve_fit(model, JKP[i]/(24*60), JKx[i], sigma = np.ones(len(JKx[i])), 
                                p0=initialx, absolute_sigma=True, check_finite=True, maxfev=50000)
        Radius[i] = popt[0]
        Period[i] = popt[1]
        Start_angle[i] = popt[2]

    stat1 = CurveFit(model, x, initialx, Time)[0][0]
    stat2 = CurveFit(model, x, initialx, Time)[0][1]
    stat3 = CurveFit(model, x, initialx, Time)[0][2]

    mean1 = np.mean(Radius)
    mean2 = np.mean(Period)
    mean3 = np.mean(Start_angle)

    bias1 = (n-1)*(mean1 - stat1)
    bias2 = (n-1)*(mean2 - stat2)
    bias3 = (n-1)*(mean3 - stat3)

    std1 = np.sqrt((n-1)*np.mean((Radius - mean1)*(Radius - mean1), axis=0))
    std2 = np.sqrt((n-1)*np.mean((Period - mean2)*(Period - mean2), axis=0))
    std3 = np.sqrt((n-1)*np.mean((Start_angle - mean3)*(Start_angle - mean3), axis=0))

    estimate1 = stat1 - bias1
    estimate2 = stat2 - bias2
    estimate3 = stat3 - bias3

    z_score = np.sqrt(2.0) * erfinv(confidence_level)

    conf_interval1 = estimate1 + z_score * np.array((-std1, std1))
    conf_interval2 = estimate2 + z_score * np.array((-std2, std2))
    conf_interval3 = estimate3 + z_score * np.array((-std3, std3))

    estimate = np.array([estimate1, estimate2, estimate3])
    std = np.array([std1, std2, std3])


    return estimate, std

#print(JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx))


'''
print(CurveFit(model, x_dataI, initial_valuesIx, Time_minsI)[0])
print(JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx))
print(JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy))

print(CurveFit(model, x_dataE, initial_valuesEx, Time_minsE)[0])
print(JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx))
print(JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy))

print(CurveFit(model, x_dataG, initial_valuesGx, Time_minsG)[0])
print(JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx))
print(JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy))

print(CurveFit(model, x_dataC, initial_valuesCx, Time_minsC)[0])
print(JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx))
print(JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy))
'''

import scienceplots
import os
import matplotlib as mpl
'''
mpl.rcParams['lines.linewidth'] = 2
plt.style.use(['science', 'grid'])

mpl.style.use('science')
fig, ax = plt.subplots()
ax.errorbar(Time_minsI/(24*60), x_dataI, yerr=0.5, label = 'Data')
ax.plot(smooth_time(Time_minsI), model(smooth_time(Time_minsI), JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0],
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[1],
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[2]), label='Fit')
ax.legend()
ax.set_title('Io x')
ax.autoscale(tight=True)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Arcsec')
fig.savefig('figures/Iox.jpg', dpi=300)
'''




'''
with plt.style.context(["science"]):
    fig, ax = plt.subplots()
    for p in [10, 15, 20, 30, 50, 100]:
        ax.plot(x, model(x, p), label=p)
    ax.legend(title="Order")
    ax.autoscale(tight=True)
    ax.set(**pparam)
    fig.savefig("figures/fig01a.jpg", dpi=300)
    plt.close()
'''

smooth_phase = np.linspace(0, 1, 300)
phaseI = np.linspace(0, 1, len(x_dataI))
time_phase = (Time_minsI / (24 * 60)) / JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1] % 1

def model_values(model, time, Radius, Period, Start_angle):
    return model(time, Radius, Period, Start_angle)

def residuals(x, modelval):
    return abs(x) - abs(modelval)

def normal_resid(x, modelval, modelstd):
    return residuals(x, modelval)/modelstd

def standard_deviation(x, modelval, N):
    return np.sqrt(np.sum((x - modelval)**2)/(N))


residualsI = residuals(x_dataI ,model_values(model, time_phase, JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][0],
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1],
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][2]))


print(residualsI)

normalresidI = normal_resid(x_dataI, residualsI ,standard_deviation(x_dataI, model_values(model, time_phase, 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][0],
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1],
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][2]), len(x_dataI)))


print(normalresidI)

#######fix normalised resiudlas



plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

ax1 = plt.subplot(gs[0])
ax1.errorbar(time_phase, x_dataI, yerr=0.5, marker='o', markersize=5, capsize=3, label='Data', ls='none')
ax1.plot(smooth_phase, model(smooth_phase*JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1], 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][0],
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1], 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][2]), label='Fit', linewidth=2)
ax1.set_ylabel('Arcsec', fontsize=14)
ax1.legend(fontsize=12)
ax1.set_title('Io x', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=12)

ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.axhline(0, color='gray', linestyle='--', linewidth=1)  # Reference line at zeroq
ax2.errorbar(time_phase, normalresidI, yerr=0.5/(standard_deviation(x_dataI, model_values(model, time_phase, 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][0],
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1],
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][2]), len(x_dataI))), 
                                        marker='o', markersize=5, capsize=3, ls='none')
ax2.set_xlabel('Time Phase', fontsize=14)
ax2.set_ylabel('Residuals', fontsize=14)
plt.tight_layout()
plt.show()


'''
plt.figure(figsize=(6, 6))
plt.errorbar(Time_minsE/(24*60), x_dataE, yerr=0.5, marker='o', label='Data', ls='none')
plt.plot(smooth_time(Time_minsE), model(smooth_time(Time_minsE), JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0],
                                        JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[1], 
                                        JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[2]), label='Fit')
plt.title('Europa x')
plt.title(r'Europa $x$-Position Over Time', fontsize=16, fontweight='bold')
plt.xlabel(r'Time (days)', fontsize=14)
plt.ylabel(r'Arcsec', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)



plt.figure(figsize=(6, 6))
plt.errorbar(Time_minsG/(24*60), x_dataG, yerr=0.5, marker='o', label='Data', ls='none')
plt.plot(smooth_time(Time_minsG), model(smooth_time(Time_minsG), JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0],
                                        JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[1], 
                                        JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[2]), label='Fit')
plt.title('Ganymede x')
plt.xlabel('Time (days)')
plt.ylabel('Arcsec')
plt.legend()


plt.figure(figsize=(6, 6))
plt.errorbar(Time_minsC/(24*60), x_dataC, yerr=0.5, marker='o', label='Data', ls='none')
plt.plot(smooth_time(Time_minsC), model(smooth_time(Time_minsC), JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0],
                                        JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[1],
                                        JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[2]), label='Fit')
plt.title('Callisto x')
plt.xlabel('Time (days)')
plt.ylabel('Arcsec')
plt.legend()
#plt.show()


plt.figure(figsize=(6, 6))
plt.errorbar(Time_minsI/(24*60), y_dataI, yerr=0.5, marker='o', label='Data', ls='none')
plt.plot(smooth_time(Time_minsI), model(smooth_time(Time_minsI), JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0],
                                        JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[1],
                                        JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[2]), label='Fit')
plt.title('Io y')
plt.xlabel('Time (days)')
plt.ylabel('Arcsec')
plt.legend()


plt.figure(figsize=(6, 6))
plt.errorbar(Time_minsE/(24*60), y_dataE, yerr=0.5, marker='o', label='Data', ls='none')
plt.plot(smooth_time(Time_minsE), model(smooth_time(Time_minsE), JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0],
                                        JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[1],
                                        JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[2]), label='Fit')
plt.title('Europa y')
plt.xlabel('Time (days)')
plt.ylabel('Arcsec')
plt.legend()


plt.figure(figsize=(6, 6))
plt.errorbar(Time_minsG/(24*60), y_dataG, yerr=0.5, marker='o', label='Data', ls='none')
plt.plot(smooth_time(Time_minsG), model(smooth_time(Time_minsG), JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0],
                                        JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[1],
                                        JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[2]), label='Fit')
plt.title('Ganymede y')
plt.xlabel('Time (days)')
plt.ylabel('Arcsec')
plt.legend()


plt.figure(figsize=(6, 6))
plt.errorbar(Time_minsC/(24*60), y_dataC, yerr=0.5, marker='o', label='Data', ls='none')
plt.plot(smooth_time(Time_minsC), model(smooth_time(Time_minsC), JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0],
                                        JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[1],
                                        JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[2]), label='Fit')
plt.title('Callisto y')
plt.xlabel('Time (days)')
plt.ylabel('Arcsec')
plt.legend()
plt.show()
'''









#print('Io Jackknife', JackKnife(modelI, x_dataI, 0.8, Time_minsI, initial_valuesIx), '\n')
#print('Europa Jackknife:', JackKnife(modelE, x_dataE, 0.8, Time_minsE, initial_valuesEx), '\n')
#print('Ganymede Jackknife:', JackKnife(modelG, x_dataG, 0.8, Time_minsG, initial_valuesGx), '\n')
#print('Callisto Jackknife:', JackKnife(modelC, x_dataC, 0.8, Time_minsC, initial_valuesCx), '\n')
 





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

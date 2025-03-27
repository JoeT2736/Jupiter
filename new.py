
print(JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][0],
                                        JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1], 
                                        JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][2])

print(JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][0],
                                        JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1], 
                                        JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][2])

'''
print(JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[2][0],
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[2][1], 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[2][2])

print(JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[3][0],
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[3][1], 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[3][2])
'''




print(JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[4][0],
                                        JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[4][1], 
                                        JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[4][2])

print(JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[4][0],
                                        JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[4][1], 
                                        JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[4][2])


##### Used for phase plots #####
smooth_phase = np.linspace(0, 1, 300)

def time_phase(time, period):
    return (time/(24*60))/period % 1

smooth_time = np.linspace(20, 70, 100000)


################# Residuals not correct ####################

###### Fucntions for ressiduals ########
def model_values(model, time, Radius, Period, Start_angle):
    return model(time, Radius, Period, Start_angle)

#print(model(time_phase(Time_minsI, JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1]), 
#            JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][0],
#            JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1],
#            JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][2]))

#print(x_dataI)


popt_x, _ = opt.curve_fit(model, Time_minsI/(24*60), x_dataI, sigma=np.ones(len(x_dataI))*0.5, 
                          absolute_sigma=True, p0=initial_valuesIx, maxfev=50000)

fitted_values_x = model(Time_minsI/(24*60), JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][0],
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1], 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][2])
residuals_x = x_dataI - fitted_values_x

mad_x = np.mean(np.abs(residuals_x - np.median(residuals_x)))  # Use MAD for normalization
normalized_residuals_x = residuals_x / mad_x

#print(residuals_x)
#print(normalized_residuals_x)


#print(*popt_x)
#print(JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][0],
#                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1], 
#                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][2])



def residuals(x, modelval):
    return x - modelval

def normal_resid(x, modelval):
    return residuals(x, modelval)/np.std(residuals(x, modelval))

'''
print(np.std(residuals(x_dataI, model(time_phase(Time_minsI,JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1]), 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][0],
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1], 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][2]))))
'''






'''
def standard_deviation(x, modelval, N):
    return np.sqrt(np.sum((x - np.mean(modelval))**2)/(N))

print(standard_deviation(x_dataI, model(time_phase(Time_minsI,JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1]), 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][0],
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1], 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][2]), len(x_dataI)))
'''


#print(time_phase(Time_minsI,JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1]))


#print(model(smooth_phase*JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1], 
#                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][0],
#                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1], 
#                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][2]))

'''
print(normal_resid(x_dataI, model(time_phase(Time_minsI,JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1]), 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][0],
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1], 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][2]), standard_deviation(x_dataI, 
                                         model(time_phase(Time_minsI,JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1]), 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][0],
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1], 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][2]), len(Time_minsI))))
'''








########################### 1D plots for x and y of each moon #############################

#My bad its so long, couldnt figure out how to make it shorter


############# Io x plot ####################

######### Use each end of confidence interval to make an outline of the furthest or closest the moons orbit could be ##########
'''
plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0) 

ax1 = plt.subplot(gs[0])                                   #0.8 is the confidence level (not sure what is should be set to yet)                                       
ax1.errorbar(time_phase(Time_minsI,JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1]), #*JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1],
             x_dataI, yerr=0.5, marker='o', markersize=5, capsize=3, label='Data', ls='none', color='black') #^^^including above makes the y-axis be in terms of days instead of phase 

ax1.plot(smooth_phase, #*JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1],
         model(smooth_phase*JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1], 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][0],
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1], 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][2]), 
                                        label='Fit', linewidth=2, color='orange')
ax1.set_ylabel('X (Arcsec)', fontsize=18)
ax1.legend(fontsize=12)
ax1.set_title('Io x', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, top=True, right=True)
ax1.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, top=True, right=True)
ax1.minorticks_on()

ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.axhline(0, color='orange', linestyle='--', linewidth=2)  # Reference line at zeroq
ax2.errorbar(time_phase(Time_minsI,JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1]), #*JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1], 
             normal_resid(x_dataI, model(Time_minsI/(24*60), #time_phase(Time_minsI,JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1]), 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][0],
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][1], 
                                        JackKnife(model, x_dataI, 0.8, Time_minsI, initial_valuesIx)[0][2])),
                                        marker='o', markersize=5, capsize=3, ls='none', color='black')
  
ax2.set_xlabel('Phase', fontsize=18)
ax2.set_ylabel('Norm Residuals', fontsize=18)
#ax2.set_ylim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=16, width=2, length=7, right=True)
ax2.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, right=True)
ax2.minorticks_on()
#plt.savefig('figures/Iox.jpg', dpi=300, bbox_inches='tight')
plt.tight_layout()
#plt.show()




############# Io y plot ####################

plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

ax1 = plt.subplot(gs[0])
ax1.errorbar(time_phase(Time_minsI,JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][1]), #*JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][1], 
             y_dataI, yerr=0.5, marker='o', markersize=5, capsize=3, label='Data', ls='none', color='black')

ax1.plot(smooth_phase, #*JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][1], 
         model(smooth_phase*JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][1], 
                                        JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][0],
                                        JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][1], 
                                        JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][2]), 
                                        label='Fit', linewidth=2, color='orange')
ax1.set_ylabel('Y (Arcsec)', fontsize=18)
ax1.legend(fontsize=12)
ax1.set_title('Io y', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, top=True, right=True)
ax1.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, top=True, right=True)
ax1.minorticks_on()

ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.axhline(0, color='orange', linestyle='--', linewidth=2)  # Reference line at zeroq
ax2.errorbar(time_phase(Time_minsI,JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][1]), #*JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][1], 
             normal_resid(y_dataI, model(Time_minsI/(24*60), #(model, time_phase(Time_minsI, JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][1]),
                                        JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][0],
                                        JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][1],
                                        JackKnife(model, y_dataI, 0.8, Time_minsI, initial_valuesIy)[0][2])),
                                        marker='o', markersize=5, capsize=3, ls='none', color='black')
  
ax2.set_xlabel('Phase', fontsize=18)
ax2.set_ylabel('Norm Residuals', fontsize=18)
#ax2.set_ylim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=16, width=2, length=7, right=True)
ax2.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, right=True)
ax2.minorticks_on()
#plt.savefig('figures/Iox.jpg', dpi=300, bbox_inches='tight')
plt.tight_layout()
#plt.show()






############# Europa x plot ####################

plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

ax1 = plt.subplot(gs[0])
ax1.errorbar(time_phase(Time_minsE,JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][1]), #*JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][1], 
             x_dataE, yerr=0.5, marker='o', markersize=5, capsize=3, label='Data', ls='none', color='black')

ax1.plot(smooth_phase, #*JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][1], 
         model(smooth_phase*JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][1],
                                        JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][0],
                                        JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][1],
                                        JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][2]), label='Fit', linewidth=2, color='purple')
ax1.set_ylabel('X (Arcsec)', fontsize=16)
ax1.legend(fontsize=12)
#ax1.set_title('Europa x', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, top=True, right=True)
ax1.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, top=True, right=True)
ax1.minorticks_on()

ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.axhline(0, color='purple', linestyle='--', linewidth=2)  # Reference line at zeroq
ax2.errorbar(time_phase(Time_minsE,JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][1]), #*JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][1], 
             normal_resid(x_dataE, model(Time_minsE/(24*60), #)(model, time_phase(Time_minsE, JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][1]),
                                        JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][0],
                                        JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][1],
                                        JackKnife(model, x_dataE, 0.8, Time_minsE, initial_valuesEx)[0][2])),
                                        marker='o', markersize=5, capsize=3, ls='none', color='black')
ax2.set_xlabel('Phase', fontsize=16)
ax2.set_ylabel('Norm Residuals', fontsize=16)
#ax2.set_ylim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, right=True)
ax2.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, right=True)
ax2.minorticks_on()
#plt.savefig('figures/Europax.jpg', dpi=300, bbox_inches='tight')
plt.tight_layout()
#plt.show()


############# Europa y plot ####################

plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

ax1 = plt.subplot(gs[0])
ax1.errorbar(time_phase(Time_minsE, JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][1]), #*JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][1], 
             y_dataE, yerr=0.5, marker='o', markersize=5, capsize=3, label='Data', ls='none', color='black')

ax1.plot(smooth_phase, #*JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][1], 
         model(smooth_phase*JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][1],
                                        JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][0],
                                        JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][1],
                                        JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][2]), label='Fit', linewidth=2, color='purple')
ax1.set_ylabel('Y (Arcsec)', fontsize=18)
ax1.legend(fontsize=12)
ax1.set_title('Europa y', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, top=True, right=True)
ax1.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, top=True, right=True)
ax1.minorticks_on()

ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.axhline(0, color='purple', linestyle='--', linewidth=2)  # Reference line at zeroq
ax2.errorbar(time_phase(Time_minsE,JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][1]), #*JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][1], 
             normal_resid(y_dataE, model(Time_minsE/(24*60), #(model, time_phase(Time_minsE, JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][1]),
                                        JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][0],
                                        JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][1],
                                        JackKnife(model, y_dataE, 0.8, Time_minsE, initial_valuesEy)[0][2])),
                                        marker='o', markersize=5, capsize=3, ls='none', color='black')
ax2.set_xlabel('Phase', fontsize=18)
ax2.set_ylabel('Norm Residuals', fontsize=18)
#ax2.set_ylim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=16, width=2, length=7, right=True)
ax2.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, right=True)
ax2.minorticks_on()
#plt.savefig('figures/Europax.jpg', dpi=300, bbox_inches='tight')
plt.tight_layout()
#plt.show()
'''




############# Ganymede x plot ####################

plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

ax1 = plt.subplot(gs[0])
ax1.errorbar(time_phase(Time_minsG,JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1]), #*JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1], 
             x_dataG, yerr=0.5, marker='o', markersize=5, capsize=3, label='Data', ls='none', color='black')
ax1.plot(smooth_phase, #*JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1],
          model(smooth_phase*JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1],
                                        JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][0],
                                        JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1],
                                        JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][2]), label='Fit', linewidth=2, color='green')

ax1.plot(smooth_phase, #*JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1],
          model(smooth_phase*JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1],
                                        JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[4][0][0],
                                        JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1],
                                        JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][2]), label='Fit', linewidth=2, color='orange', linestyle='--', alpha=0.7)

ax1.plot(smooth_phase, #*JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1],
          model(smooth_phase*JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1],
                                        JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[4][0][1],
                                        JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1],
                                        JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][2]), label='Fit', linewidth=2, color='purple', linestyle='--', alpha=0.7)

ax1.set_ylabel('X (Arcsec)', fontsize=18)
ax1.legend(fontsize=12)
ax1.set_title('Ganymede x', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, top=True, right=True)
ax1.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, top=True, right=True)
ax1.minorticks_on()

ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.axhline(0, color='green', linestyle='--', linewidth=2)  # Reference line at zeroq
ax2.errorbar(time_phase(Time_minsG,JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1]), #*JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1], 
             normal_resid(x_dataG, model(Time_minsG/(24*60), #(model, time_phase(Time_minsG, JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1]),
                                        JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][0],
                                        JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1],
                                        JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][2])),
                                        marker='o', markersize=5, capsize=3, ls='none', color='black')
ax2.set_xlabel('Phase', fontsize=18)
ax2.set_ylabel('Norm Residuals', fontsize=18)
#ax2.set_ylim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=16, width=2, length=7, right=True)
ax2.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, right=True)
ax2.minorticks_on()
#plt.savefig('figures/ganymedex.jpg', dpi=300, bbox_inches='tight')
plt.tight_layout()
#plt.show()



############# Ganymede y plot ####################

plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

ax1 = plt.subplot(gs[0])
ax1.errorbar(time_phase(Time_minsG,JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1]), #*JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1], 
             y_dataG, yerr=0.5, marker='o', markersize=5, capsize=3, label='Data', ls='none', color='black')
ax1.plot(smooth_phase, #*JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1], 
         model(smooth_phase*JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1],
                                        JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][0],
                                        JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1],
                                        JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][2]), label='Fit', linewidth=2, color='green')
ax1.set_ylabel('Y (Arcsec)', fontsize=18)
ax1.legend(fontsize=12)
ax1.set_title('Ganymede y', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, top=True, right=True)
ax1.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, top=True, right=True)
ax1.minorticks_on()

ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.axhline(0, color='green', linestyle='--', linewidth=2)  # Reference line at zeroq
ax2.errorbar(time_phase(Time_minsG,JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1]), #*JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1], 
             normal_resid(y_dataG, model(Time_minsG/(24*60), #(model, time_phase(Time_minsG, JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1]),
                                        JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][0],
                                        JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][1],
                                        JackKnife(model, y_dataG, 0.8, Time_minsG, initial_valuesGy)[0][2])),
                                        marker='o', markersize=5, capsize=3, ls='none', color='black')
ax2.set_xlabel('Phase', fontsize=18)
ax2.set_ylabel('Norm Residuals', fontsize=18)
#ax2.set_ylim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=16, width=2, length=7, right=True)
ax2.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, right=True)
ax2.minorticks_on()
#plt.savefig('figures/ganymedex.jpg', dpi=300, bbox_inches='tight')
plt.tight_layout()
#plt.show()



############ Ganymede Prediction plot ############






'''
############# Callisto x plot ####################

plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

ax1 = plt.subplot(gs[0])
ax1.errorbar(time_phase(Time_minsC,JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][1]), #*JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][1], 
             x_dataC, yerr=0.5, marker='o', markersize=5, capsize=3, label='Data', ls='none', color='black')
ax1.plot(smooth_phase, #*JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][1], 
         model(smooth_phase*JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][1],
                                        JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][0],
                                        JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][1],
                                        JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][2]), label='Fit', linewidth=2, color='Blue')
ax1.set_ylabel('X (Arcsec)', fontsize=18)
ax1.legend(fontsize=12)
ax1.set_title('Callisto x', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, top=True, right=True)
ax1.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, top=True, right=True)
ax1.minorticks_on()

ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.axhline(0, color='blue', linestyle='--', linewidth=2)  # Reference line at zeroq
ax2.errorbar(time_phase(Time_minsC,JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][1]), #*JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][1], 
             normal_resid(x_dataC, model(Time_minsC/(24*60), #(model, time_phase(Time_minsC, JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][1]),
                                        JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][0],
                                        JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][1],
                                        JackKnife(model, x_dataC, 0.8, Time_minsC, initial_valuesCx)[0][2])),
                                        marker='o', markersize=5, capsize=3, ls='none', color='black')
ax2.set_xlabel('Phase', fontsize=18)
ax2.set_ylabel('Norm Residuals', fontsize=18)
#ax2.set_ylim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=16, width=2, length=7, right=True)
ax2.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, right=True)
ax2.minorticks_on()
#plt.savefig('figures/callistox.jpg', dpi=300, bbox_inches='tight')
plt.tight_layout()
#plt.show()



############# Callisto y plot ####################

plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

ax1 = plt.subplot(gs[0])
ax1.errorbar(time_phase(Time_minsC,JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][1]), #*JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][1], 
             y_dataC, yerr=0.5, marker='o', markersize=5, capsize=3, label='Data', ls='none', color='black')
ax1.plot(smooth_phase, #*JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][1], 
         model(smooth_phase*JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][1],
                                        JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][0],
                                        JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][1],
                                        JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][2]), label='Fit', linewidth=2, color='Blue')
ax1.set_ylabel('Y (Arcsec)', fontsize=18)
ax1.legend(fontsize=12)
ax1.set_title('Callisto y', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=7, top=True, right=True)
ax1.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, top=True, right=True)
ax1.minorticks_on()

ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.axhline(0, color='blue', linestyle='--', linewidth=2)  # Reference line at zeroq
ax2.errorbar(time_phase(Time_minsC,JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][1]), #*JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][1], 
             normal_resid(y_dataC, model(Time_minsC/(24*60), #(model/(24*60), #time_phase(Time_minsC, JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][1]),
                                        JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][0],
                                        JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][1],
                                        JackKnife(model, y_dataC, 0.8, Time_minsC, initial_valuesCy)[0][2])),
                                        marker='o', markersize=5, capsize=3, ls='none', color='black')
ax2.set_xlabel('Phase', fontsize=18)
ax2.set_ylabel('Norm Residuals', fontsize=18)
#ax2.set_ylim(-1, 1)
ax2.tick_params(axis='both', which='major', labelsize=16, width=2, length=7, right=True)
ax2.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3, right=True)
ax2.minorticks_on()
#plt.savefig('figures/callistox.jpg', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()
'''



#################### 2D Jackknife ####################


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



'''
###### Values of orbital parameters ######
print(convert_to_physical(ellipse_fit(x_dataG, y_dataG)[0], ellipse_fit(x_dataG, y_dataG)[1], ellipse_fit(x_dataG, y_dataG)[2],
                         ellipse_fit(x_dataG, y_dataG)[3], ellipse_fit(x_dataG, y_dataG)[4],ellipse_fit(x_dataG, y_dataG)[5]))

print(convert_to_physical(JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[0], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[1],
                         JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[2], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[3],
                         JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[4], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[5]))
'''



def ellipse_eq(x, y, x0, y0, a, b, theta):
    term1 = ((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta))**2 / a**2
    term2 = ((-(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta))**2) / b**2
    return term1 + term2 - 1  # ellipse is defined by ellipse_eq == 0

# Define the plotting grid.

N = 5000
x_min, x_max = -700, 700  # use names that don't conflict with x0_model, y0_model
y_min, y_max = -100, 100
xs = np.linspace(x_min, x_max, N)
ys = np.linspace(y_min, y_max, N)
X, Y = np.meshgrid(xs, ys)



def Z(X, Y, xdata, ydata):
    return ellipse_eq(X, Y, convert_to_physical(JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[0], JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[1],
                            JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[2], JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[3],
                            JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[4], JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[5])[0],  #x0
                    convert_to_physical(JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[0], JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[1],
                            JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[2], JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[3],
                            JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[4], JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[5])[1],  #y0
                    convert_to_physical(JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[0], JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[1],
                            JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[2], JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[3],
                            JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[4], JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[5])[2],  #a
                    convert_to_physical(JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[0], JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[1],
                            JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[2], JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[3],
                            JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[4], JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[5])[3],  #b
                    convert_to_physical(JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[0], JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[1],
                            JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[2], JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[3],
                            JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[4], JackKnife2D(ellipse_fit, xdata, ydata, 0.8)[5])[4])  #theta



#print(x_dataE)
#print(EuroX)



############ 2D plot for orbit of all moons #############


plt.figure(figsize=(8, 8))

#plt.errorbar(x_dataI, y_dataI, xerr = np.ones(len(x_dataI)), yerr = np.ones(len(x_dataI)), ls = 'none', color='black', markersize=5, capsize=3, marker='o')#, alpha=0.5)
#io = plt.contour(X, Y, Z(X, Y, x_dataI, y_dataI), levels=[0], colors='orange', linewidths=2)

#plt.errorbar(x_dataC, y_dataC, xerr = np.ones(len(x_dataC)), yerr = np.ones(len(x_dataC)), ls = 'none', color='black', markersize=5, capsize=3, marker='o')#, alpha=0.5)
#calli = plt.contour(X, Y, Z(X, Y, x_dataC, y_dataC), levels=[0], colors='blue', linewidths=2)


plt.errorbar(x_dataG, y_dataG, xerr = np.ones(len(x_dataG)), yerr = np.ones(len(x_dataG)), ls = 'none', color='black', markersize=5, capsize=3, marker='o', label='Data')#, alpha=0.5)
gany = plt.contour(X, Y, Z(X, Y, x_dataG, y_dataG), levels=[0], colors='green', linewidths=2)


                                 #### Europa does not work correctly ####
#plt.errorbar(x_dataE, y_dataE, xerr = np.ones(len(x_dataE)), yerr = np.ones(len(x_dataE)), ls = 'none', color='black', markersize=5, capsize=3, marker='o', label='Europa')
#euro = plt.contour(X, Y, Z(X, Y, x_dataE, y_dataE), levels=[0], colors='purple')


plt.scatter(0, 0, color='red', label='Jupiter')


proxy = [plt.Rectangle((0,0),1,1,fc = 'green')]# , plt.Rectangle((0,0),1,1,fc = 'purple'), plt.Rectangle((0,0),1,1,fc = 'orange'), plt.Rectangle((0,0),1,1,fc = 'blue')]#, plt.Rectangle((0,0),1,1,fc = 'red')]
plt.legend(proxy, ['Ganymede'], fontsize=18) #, 'Europa', 'Io', 'Callisto'], fontsize=18)


plt.ylim(-80, 80)
plt.xlim(-600, 600)
plt.xlabel("X (Arcsec)", fontsize=18)
plt.ylabel("Y (Arcsec)", fontsize=18)

plt.tick_params(axis='both', which='major', labelsize=16, length=8, width=1.5, top=True, right=True)
plt.tick_params(axis='both', which='minor', labelsize=16, length=4, width=1, top=True, right=True)
plt.minorticks_on()

plt.tight_layout()
#plt.show()



'''
##### Io values #####
print(convert_to_physical(ellipse_fit(x_dataI, y_dataI)[0], ellipse_fit(x_dataI, y_dataI)[1], ellipse_fit(x_dataI, y_dataI)[2],
                         ellipse_fit(x_dataI, y_dataI)[3], ellipse_fit(x_dataI, y_dataI)[4],ellipse_fit(x_dataI, y_dataI)[5]))

print(convert_to_physical(JackKnife2D(ellipse_fit, x_dataI, y_dataI, 0.8)[0], JackKnife2D(ellipse_fit, x_dataI, y_dataI, 0.8)[1],
                            JackKnife2D(ellipse_fit, x_dataI, y_dataI, 0.8)[2], JackKnife2D(ellipse_fit, x_dataI, y_dataI, 0.8)[3],
                            JackKnife2D(ellipse_fit, x_dataI, y_dataI, 0.8)[4], JackKnife2D(ellipse_fit, x_dataI, y_dataI, 0.8)[5]))



##### Europa values #####
print(convert_to_physical(ellipse_fit(x_dataE, y_dataE)[0], ellipse_fit(x_dataE, y_dataE)[1], ellipse_fit(x_dataE, y_dataE)[2],
                            ellipse_fit(x_dataE, y_dataE)[3], ellipse_fit(x_dataE, y_dataE)[4],ellipse_fit(x_dataE, y_dataE)[5]))

print(convert_to_physical(JackKnife2D(ellipse_fit, x_dataE, y_dataE, 0.8)[0], JackKnife2D(ellipse_fit, x_dataE, y_dataE, 0.8)[1],
                            JackKnife2D(ellipse_fit, x_dataE, y_dataE, 0.8)[2], JackKnife2D(ellipse_fit, x_dataE, y_dataE, 0.8)[3],
                            JackKnife2D(ellipse_fit, x_dataE, y_dataE, 0.8)[4], JackKnife2D(ellipse_fit, x_dataE, y_dataE, 0.8)[5]))
'''


##### Ganymede values #####
#print(convert_to_physical(ellipse_fit(x_dataG, y_dataG)[0], ellipse_fit(x_dataG, y_dataG)[1], ellipse_fit(x_dataG, y_dataG)[2],
#                         ellipse_fit(x_dataG, y_dataG)[3], ellipse_fit(x_dataG, y_dataG)[4],ellipse_fit(x_dataG, y_dataG)[5]))

print(convert_to_physical(JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[0], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[1],
                         JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[2], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[3],
                         JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[4], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[5]))


'''
##### Callisto values #####
print(convert_to_physical(ellipse_fit(x_dataC, y_dataC)[0], ellipse_fit(x_dataC, y_dataC)[1], ellipse_fit(x_dataC, y_dataC)[2],
                         ellipse_fit(x_dataC, y_dataC)[3], ellipse_fit(x_dataC, y_dataC)[4],ellipse_fit(x_dataC, y_dataC)[5]))

print(convert_to_physical(JackKnife2D(ellipse_fit, x_dataC, y_dataC, 0.8)[0], JackKnife2D(ellipse_fit, x_dataC, y_dataC, 0.8)[1],
                         JackKnife2D(ellipse_fit, x_dataC, y_dataC, 0.8)[2], JackKnife2D(ellipse_fit, x_dataC, y_dataC, 0.8)[3],
                         JackKnife2D(ellipse_fit, x_dataC, y_dataC, 0.8)[4], JackKnife2D(ellipse_fit, x_dataC, y_dataC, 0.8)[5]))
'''


def AnglFreq(Period):
    w = 2*np.pi/Period
    return w

#print(AnglFreq(JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1]))

def Future_Position(Period, Time, Xradius, Yradius, Startx, Starty):
    tot = AnglFreq(Period)*Time
    Rotations = tot/(2*np.pi)
    a = int(Rotations)
    Phase = (Rotations - a)
    Xpos = Startx + Phase*Xradius
    Ypos = Starty + Phase*Yradius
    return tot, Rotations, Phase, Xpos, Ypos


###### use max values of x and y from the plot in calc below ######
############ y radius not correct from convert_to_physical ############# correct plot though (not sure how)


print(Future_Position(JackKnife(model, x_dataG, 0.8, Time_minsG, initial_valuesGx)[0][1], 8*365.25*24*60*60, 
                      convert_to_physical(JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[0], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[1],
                         JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[2], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[3],
                         JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[4], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[5])[2],
                         convert_to_physical(JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[0], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[1],
                         JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[2], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[3],
                         JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[4], JackKnife2D(ellipse_fit, x_dataG, y_dataG, 0.8)[5])[3],
                         0, 0))
b = 2*np.pi
#print(b)










plt.show()







import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



R = 8.314

co2_h0 = -393.52 # kJ/mol
feo_h0 = -272 # kJ/mol
coo_h0 = -237 # kJ/mol 
nio_h0 = -240
cnt_h0 = 4.7 # kJ/mol
h2o_h0 = -241.83 # kJ/mol
c2h4_h0 = 52.5 # kJ/mol
fe_h0 = 0
co_h0 = 0 # solid
ni_h0 = 0
# co_h0 = 18 # liquid

co2_s0 = 213.8 # J/mol/K
feo_s0 = 60.75
coo_s0 = 52.85
nio_s0 = 37.99
cnt_s0 = 5.6
h2o_s0 = 188.8
c2h4_s0 = 219.3
fe_s0 = 27.3
co_s0 = 30.07 # solid
ni_s0 = 29.87
# co_s0 = 41 # liquid

co_deltaH0 = 4*coo_h0 + c2h4_h0 - co2_h0 - cnt_h0 - 2*h2o_h0 - 4*co_h0
co_deltaS0 = 4*coo_s0 + c2h4_s0 - co2_s0 - cnt_s0 - 2*h2o_s0 - 4*co_s0
co_deltaS0 /= 1000 # convert to kJ
co_deltaG0 = co_deltaH0 - 298.15*co_deltaS0


data = pd.read_csv('CampaignSummary.csv')

plt.rc('font', size=14)
plt.rc('axes', labelsize=14)

fig1, ax1 = plt.subplots()

lnq = np.log(data['VAL_MFCSP1']) - np.log(data['VAL_MFCSP2']/10) - (2*np.log(data['Water']/10**5))

sc1 = ax1.scatter(lnq, data['VAL_TEMP']+273, c=data['G Band Area Result'], cmap='Blues',
	edgecolors='k', linewidths=0.15, s=20*data['G Band Area Result']+10)
cbar = fig1.colorbar(sc1, ax=ax1)
cbar.set_label('Normalized G Band Area')

# ax1.set_xlim([xzoommin, xzoommax])
# ax1.set_ylim([yzoommin, yzoommax])

xmin, xmax = ax1.get_xlim()
ymin, ymax = ax1.get_ylim()

X, Y = np.meshgrid(np.linspace(xmin, xmax, num=50), np.linspace(ymin, ymax, num=50))

Z = co_deltaG0 + (R*X*Y/1000)

cs1 = ax1.contour(X, Y, Z, levels=12, colors='black', zorder=2)
ax1.clabel(cs1, cs1.levels, inline=True, fmt=lambda a: '{} kJ/mol'.format(int(a)), zorder=1)

ax1.set_ylim([700, ymax])
ax1.set_xlabel(r'Ln($\frac{\mathrm{[C_2H_4]}}{\mathrm{[CO_2][H_2O]^2}}$)')
ax1.set_ylabel('Temperature [K]')

fig1.tight_layout()
# fig1.savefig('yield_thermodynamic_scatter.png', dpi=300)

fig1.savefig('yield_thermodynamic_scatter.png', dpi=300)


fig2, ax2 = plt.subplots()

lnq = np.log(data['VAL_MFCSP1']) - np.log(data['VAL_MFCSP2']/10) - (2*np.log(data['Water']/10**5))

sc2 = ax2.scatter(lnq, data['VAL_TEMP']+273, c=data['Diameter Control Result'], cmap='Oranges',
	edgecolors='k', linewidths=0.15, s=data['Diameter Control Result']*200+10)
cbar = fig2.colorbar(sc2, ax=ax2)
cbar.set_label('Diameter Control Metric v3')


# ax2.set_xlim([xzoommin, xzoommax])
# ax2.set_ylim([yzoommin, yzoommax])

xmin, xmax = ax2.get_xlim()
ymin, ymax = ax2.get_ylim()

X, Y = np.meshgrid(np.linspace(xmin, xmax, num=50), np.linspace(ymin, ymax, num=50))

Z = co_deltaG0 + (R*X*Y/1000)

cs2 = ax2.contour(X, Y, Z, levels=12, colors='black', zorder=2)
ax2.clabel(cs2, cs2.levels, inline=True, fmt=lambda a: '{} kJ/mol'.format(int(a)), zorder=1)

ax2.set_ylim([700, ymax])

ax2.set_xlabel(r'Ln($\frac{\mathrm{[C_2H_4]}}{\mathrm{[CO_2][H_2O]^2}}$)')
ax2.set_ylabel('Temperature [K]')

fig2.tight_layout()
# fig2.savefig('diameter_thermodynamic_scatter.png', dpi=300)

fig2.savefig('diameter_thermodynamic_scatter_v3.png', dpi=300)



fig3, ax3 = plt.subplots()

lnq = np.log(data['VAL_MFCSP1']) - np.log(data['VAL_MFCSP2']/10) - (2*np.log(data['Water']/10**5))

sc3 = ax3.scatter(lnq, data['VAL_TEMP']+273, c=data['Experiment Number'], cmap='gist_rainbow')
cbar = fig3.colorbar(sc3, ax=ax3)
cbar.set_label('Experiment Number')


# ax3.set_xlim([xzoommin, xzoommax])
# ax3.set_ylim([yzoommin, yzoommax])

xmin, xmax = ax3.get_xlim()
ymin, ymax = ax3.get_ylim()

X, Y = np.meshgrid(np.linspace(xmin, xmax, num=50), np.linspace(ymin, ymax, num=50))

Z = co_deltaG0 + (R*X*Y/1000)

cs3 = ax3.contour(X, Y, Z, levels=12, colors='black', zorder=2)
ax3.clabel(cs3, cs3.levels, inline=True, fmt=lambda a: '{} kJ/mol'.format(int(a)), zorder=1)

ax3.set_ylim([700, ymax])
ax3.set_xlabel(r'Ln($\frac{\mathrm{[C_2H_4]}}{\mathrm{[CO_2][H_2O]^2}}$)')
ax3.set_ylabel('Temperature [K]')

fig3.tight_layout()
# fig3.savefig('number_thermodynamic_scatter.png', dpi=300)

fig3.savefig('number_thermodynamic_scatter.png', dpi=300)

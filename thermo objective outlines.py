import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, convex_hull_plot_2d


R = 8.314

data = pd.read_csv('CampaignSummary.csv')
data = data[data['G Band Area Result'] >= 0.1]
# data = data[data['Experiment Number'] < 101]

plt.rc('font', size=13)
plt.rc('axes', labelsize=13)
plt.rc('lines', markersize=9, linewidth=1.5)

# norm = cm.colors.Normalize(vmin=0, vmax=data['G Band Area Result'].max())
norm = cm.colors.Normalize(vmin=0, vmax=90)


data['lnq'] = np.log(data['VAL_MFCSP1']) - np.log(data['VAL_MFCSP2']/10) - (2*np.log(data['Water']/10**5))
data['tk'] = data['VAL_TEMP'] + 273
data['Gibbs'] = 62 - R * data['lnq'] * data['tk'] / 1000

xmin, xmax = min(data['lnq']), max(data['lnq'])
print(xmin, xmax)
ymin, ymax = min(data['tk']), max(data['tk'])
print(ymin, ymax)

n_points = 100
X = np.linspace(20, 30, num=n_points)
Y = np.linspace(700, 1300, num=n_points)

X_arr, Y_arr = np.meshgrid(X, Y)

Density = np.zeros((n_points, n_points))
G_array = np.zeros((n_points, n_points))
D_array = np.zeros((n_points, n_points))

# for i, (x0, x1) in enumerate(zip(X[:-1], X[1:])):
# 	for j, (y0, y1) in enumerate(zip(Y[:-1], Y[1:])):
# 		

# for n, (q, t) in enumerate(zip(data['lnq'], tk)):
# 	Xq = X - q
# 	Yt = Y - t
# 	j = np.where( Xq > 0, Xq, np.inf).argmin()
# 	i = np.where(Yt > 0, Yt, np.inf).argmin()


# 	Density[i, j] += 1
# 	G_array[i, j] += data['G Band Area Result'][n]
# 	D_array[i, j] += data['Diameter Control Result'][n]

# D_array_result = np.divide(D_array, Density, where=Density>0)
# G_density = np.divide(G_array, Density, where=Density>0)


fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()
# fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12,5))


# --- Yield Data ------
good_data = data[data['G Band Area Result'] >= data['G Band Area Result'].quantile(0.9)]

good_points = np.asarray(list(zip(good_data['lnq'], good_data['tk'])))


hull = ConvexHull(good_points)

simplex = hull.simplices[0]
ax.plot(good_points[simplex, 0], good_points[simplex, 1], color='xkcd:light blue', linewidth=2, linestyle='--', label='90% yield')
for simplex in hull.simplices[1:]:
	ax.plot(good_points[simplex, 0], good_points[simplex, 1], color='xkcd:light blue', linewidth=2, linestyle='--')


great_data = data[data['G Band Area Result'] >= data['G Band Area Result'].quantile(0.95)]

great_points = np.asarray(list(zip(great_data['lnq'], great_data['tk'])))


hull = ConvexHull(great_points)

# print([simplex for simplex in hull.simplices])
simplex = hull.simplices[0]
ax.plot(great_points[simplex, 0], great_points[simplex, 1], color='xkcd:blue', linewidth=2, label='95% yield')
for simplex in hull.simplices[1:]:
	ax.plot(great_points[simplex, 0], great_points[simplex, 1], color='xkcd:blue', linewidth=2)



# --- Diameter Data -------
good_data = data[data['Diameter Control Result'] >= data['Diameter Control Result'].quantile(0.9)]


good_points = np.asarray(list(zip(good_data['lnq'], good_data['tk'])))


hull = ConvexHull(good_points)

simplex = hull.simplices[0]
ax.plot(good_points[simplex, 0], good_points[simplex, 1], color='xkcd:light orange', linewidth=2, linestyle='--', label='90% diameter')
for simplex in hull.simplices:
	ax.plot(good_points[simplex, 0], good_points[simplex, 1], color='xkcd:light orange', linestyle='--', linewidth=2)


great_data = data[data['Diameter Control Result'] >= data['Diameter Control Result'].quantile(0.95)]

great_points = np.asarray(list(zip(great_data['lnq'], great_data['tk'])))


hull = ConvexHull(great_points)

simplex = hull.simplices[0]
ax.plot(great_points[simplex, 0], great_points[simplex, 1], color='xkcd:orange', linewidth=2, linestyle='-', label='95% diameter')
for simplex in hull.simplices:
	ax.plot(great_points[simplex, 0], great_points[simplex, 1], color='xkcd:orange', linestyle='-', linewidth=2)


# plot size and color related to pareto

norm_yield = data['G Band Area Result'] / data['G Band Area Result'].max()
data['pareto mag'] = np.sqrt(norm_yield**2 + data['Diameter Control Result']**2)
data['pareto angle'] = np.rad2deg(np.arctan(data['Diameter Control Result'] / norm_yield))
best_data = data[data['Diameter Control Result'] >= data['Diameter Control Result'].quantile(0.9)]

# sc = ax.scatter(lnq, tk, c=data['pareto angle'], cmap='brg', norm=norm, s=100*data['pareto mag'], linewidth=0, zorder=1, alpha=0.7)
# ax.scatter(lnq, tk, alpha=0.6, linewidth=0, c='tab:gray')
sc = ax.scatter(data['lnq'], data['tk'], c=data['pareto mag'], cmap='Greys', s=50, alpha=0.8, linewidth=0)
# ax.scatter(best_data['lnq'], best_data['tk'], c='k', s=50)


# sc1 = ax1.scatter(norm_yield, data['Diameter Control Result'], c=data['pareto angle'], cmap='brg', norm=norm, s=80*data['pareto mag']+10, linewidth=0, zorder=1, alpha=0.7)
# sub_data = data[data['Experiment Number'] % 2 == 0]
sub_data = data
sc1 = ax1.scatter(norm_yield, sub_data['Diameter Control Result'], c=sub_data['Experiment Number'], cmap='gist_rainbow', alpha=0.7, linewidth=0)#, s=80*sub_data['pareto mag']+10, zorder=1)

# ax.scatter(lnq_good, tk_good, c=good_data['G Band Area Result'], norm=norm)
# ax.scatter(lnq, tk, c=data['G Band Area Result'], cmap='Blues', alpha=0.5, linewidth=0, zorder=6)
# ax.scatter(lnq, tk, c=data['Diameter Control Result'], cmap='Reds', alpha=0.5, linewidth=0, zorder=6)

ax.set_xlim([22, 27])
ax.set_ylim([800, 1100])

ax.set_xlabel('Ln$\left(\dfrac{[\mathrm{C}_2\mathrm{H}_4]}{[\mathrm{CO}_2][\mathrm{H}_2\mathrm{O}]^2}\\right)$')
ax.set_ylabel('Temperature [K]')
ax.set_title('Experimental Space')
ax.legend()

# ax1.set_xlim([-0.1,1.1])
# ax1.set_ylim([-0.1,1.1])

ax1.set_xlabel('Normalized Yield')
ax1.set_ylabel('Normalized Diameter Control')
ax1.set_title('Pareto Plot')

cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('Pareto Vector Magnitude')

cbar = fig.colorbar(sc1, ax=ax1)
cbar.set_label('Experiment Number')
# cbar.set_label('Pareto Angle')



# ax2.scatter(data['Gibbs'], norm_yield)
# ax2.scatter(data['Gibbs'], data['Diameter Control Result'])

# ax3.scatter(data['Experiment Number'], data['Diameter Control Result']+norm_yield, c=data['pareto angle'], cmap='brg', norm=norm)


# ax.hist(data['G Band Area Result'])
# ax.imshow(G_array, origin='lower')

# ax.set_xticks(X)

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

# ax1.imshow(Density, origin='lower', cmap='Blues')
# ax1.set_title('Experimental Density')

# ax2.imshow(G_array, origin='lower', cmap='Blues')
# ax2.set_title('Total G Band Area')

# ax3.imshow(G_density, origin='lower', cmap='Blues')
# ax3.set_title('Average G Band Area')

# ax4.imshow(D_array_result, origin='lower', cmap='Oranges')
# ax4.set_title('Average 532 Diameter Control')

fig.text(0.02, 0.98, '(a)', va='top', fontsize='x-large')
fig1.text(0.02, 0.98, '(b)', va='top', fontsize='x-large')

fig.tight_layout()
fig1.tight_layout()
plt.show()

fig.savefig('best_outlines.png', dpi=300)
fig1.savefig('pareto_plot.png', dpi=300)


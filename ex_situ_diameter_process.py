import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plothelp
import pyBayesThresh as pbt
import renishaw_wire as renishaw
from sys import float_info
from scipy.optimize import minimize_scalar

def mean(*args):
	totals = []
	n = len(args)

	totals = args[0].copy()

	for dat in args[1:]:
		for i, datum in enumerate(dat):
			totals[i] += datum

	means = np.array(totals) / n
	return means


def align_spectra(c):
	global pre_532, post_532
	rows = pre_532['x'].between(200, 800)
	return np.abs(np.sum(np.subtract(post_532['y'][rows], c+pre_532['y'][rows])))

def prevent_zero_division(data):
	correct_data = data.copy()
	for i, datum in enumerate(data):
		if abs(datum) < float_info.epsilon:
			correct_data[i] = float_info.epsilon
	return correct_data

def r2d(wavenumbers):
	wavenumbers = prevent_zero_division(wavenumbers)
	return 248/wavenumbers


def denoise_spectrum(y_values, denoise_level=1, wavelet='sym4', noise_est='level_independent', thresh_rule='median'):
    intensities, en = pbt.wavelet_denoise(y_values, denoise_level, 
        wav_name=wavelet, noise_est=noise_est, thresh_rule=thresh_rule)
    
    return intensities

def find_biggest_bounds(wavenumbers, intensities):
	global ares_noise_thresh

	rows = wavenumbers.between(200, 350)

	processed = np.clip(intensities[rows] - ares_noise_thresh, a_min=0, a_max=None)
	x = wavenumbers[rows]

	# plt.plot(x, processed)
	# plt.show()
	# exit()

	rbm_area = np.trapz(processed, x=x)

	p_integral = [np.trapz(processed[:i], x=x[:i]) for i in range(len(x))]
	norm_integral = p_integral / rbm_area
	norm_integral = denoise_spectrum(norm_integral)

	gradient = np.gradient(norm_integral, x) * rbm_area
	grad_gradient = np.gradient(gradient, x)

	zero_crossings = np.where(np.diff(np.sign(grad_gradient) >= 0))[0]
	zero_crossings = [zc+1 for zc in zero_crossings[1:] if grad_gradient[zc] <= 0]
	zero_crossings.insert(0, 0)

	peak_areas = [norm_integral[right]-norm_integral[left] for left, right in zip(zero_crossings[:-1], zero_crossings[1:])]
	sort_indices = np.argsort(peak_areas)

	# print(peak_areas[sort_indices[-1]])

	if not sort_indices.any():
		return 100, 100

	left = zero_crossings[sort_indices[-1]]
	right = zero_crossings[sort_indices[-1]+1]
	
	return wavenumbers[rows].iloc[left], wavenumbers[rows].iloc[right]


def diameter_control_calc(wavenumbers, intensities, patch_pillar, laser):
	fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

	dead_rows = wavenumbers.between(550, 750)
	noise_mean = intensities[dead_rows].mean()
	noise_sdev = intensities[dead_rows].std()
	noise_thresh = 5 * noise_sdev

	rows = wavenumbers.between(150, 400)

	ax1.plot(wavenumbers[rows], intensities[rows], label='raw')

	processed = np.clip(intensities[rows] - noise_mean - noise_thresh, a_min=0, a_max=None)
	x = wavenumbers[rows]
	ax1.plot(x, processed, label='subtract noise')

	rbm_area = np.trapz(processed, x=x)

	p_integral = [np.trapz(processed[:i], x=x[:i]) for i in range(len(x))]
	norm_integral = p_integral / rbm_area
	norm_integral = denoise_spectrum(norm_integral)

	ax2.plot(x, norm_integral, label='integral')
	gradient = np.gradient(norm_integral, x) * rbm_area
	# gradient = gradient.shift(-1)
	grad_gradient = np.gradient(gradient, x)
	# grad_gradient = grad_gradient.shift(-1)

	zero_crossings = np.where(np.diff(np.sign(grad_gradient) >= 0))[0]
	zero_crossings = [zc+1 for zc in zero_crossings[1:] if grad_gradient[zc] <= 0]
	zero_crossings.insert(0, 0)

	ax3.plot(x, gradient, label='redifferentiated')

	ax4.plot(x, intensities[rows])

	for z in zero_crossings:
		ax4.axvline(x.iloc[z], color='k', linestyle='--')

	peak_areas = [norm_integral[right]-norm_integral[left] for left, right in zip(zero_crossings[:-1], zero_crossings[1:])]
	sort_indices = np.argsort(peak_areas)
	n_peaks = len([area for area in peak_areas if area >= 0.1])

	ax4.set_title(str(peak_areas[sort_indices[-1]]))

	title = f'{patch_pillar} {laser}'
	fig1.suptitle(title)
	fig1.tight_layout()
	plt.show()
	plt.close()


	return peak_areas[sort_indices[-1]], zero_crossings

def rbms_in_biggest(wavenumbers, intensities, left, right):

	dead_rows = wavenumbers.between(550, 750)
	noise_mean = intensities[dead_rows].mean()
	noise_sdev = intensities[dead_rows].std()
	noise_thresh = 5 * noise_sdev

	rows = wavenumbers.between(150, 350)

	processed = np.clip(intensities[rows] - noise_mean - noise_thresh, a_min=0, a_max=None)
	x = wavenumbers[rows]

	total_area = np.trapz(processed, x=x)

	target_rows = x.between(left, right)
	control_area = np.trapz(processed[target_rows], x=x[target_rows])

	agree = control_area / total_area
	disagree = 1 - agree


	return agree, (agree-disagree)


def si_area(wavenumbers, intensities):
	rows = wavenumbers.between(500, 540)
	area = np.trapz(intensities[rows], x=wavenumbers[rows])

	return np.abs(area)

def find_renishaw_files(search):

	for (root, dirs, files) in os.walk('Renishaw/Campaign 4/'):
		files = [file for file in files if search in file]

		if files:
			break

	return files

plothelp.set_presentation()

summary = pd.read_csv('Campaign 4/CampaignSummary.csv')

global ares_noise_thresh
ares_noise = pd.read_csv('../ARES Analyzers/noise_model.csv', skiprows=4, names=['wavenumbers', 'intensities'])
ares_noise_thresh = 5 * ares_noise['intensities'][ares_noise['wavenumbers'].between(150, 350)].std()

summary['488 control'] = np.zeros(summary.shape[0])
summary['633 control'] = np.zeros(summary.shape[0])
summary['785 control'] = np.zeros(summary.shape[0])
summary['488 diff'] = np.zeros(summary.shape[0])
summary['633 diff'] = np.zeros(summary.shape[0])
summary['785 diff'] = np.zeros(summary.shape[0])
summary['Control Center'] = np.zeros(summary.shape[0])
summary['488 raw control'] = np.zeros(summary.shape[0])
summary['488 raw center'] = np.zeros(summary.shape[0])
summary['633 raw control'] = np.zeros(summary.shape[0])
summary['633 raw center'] = np.zeros(summary.shape[0])
summary['785 raw control'] = np.zeros(summary.shape[0])
summary['785 raw center'] = np.zeros(summary.shape[0])
summary['Control Width'] = np.zeros(summary.shape[0])

for i, experiment in summary.iterrows():

	number = str(int(experiment['Experiment Number'])).zfill(3)
	ares_control = experiment['Diameter Control Result']


	patch = experiment['Patch']
	pillar = experiment['Pillar']
	patch_pillar = f'{ int(patch) }-{ int(pillar) }'

	# find renishaw data for experiment, if it exists
	renishaw_files = find_renishaw_files(patch_pillar)
	if not renishaw_files:
		continue

	renishaw_files = [f'Renishaw/Campaign 4/{file}' for file in renishaw_files]

	files_488 = []
	files_633 = []
	files_785 = []
	for file in renishaw_files:
		
		if '488' in file:
			files_488.append(file)
		elif '633' in file:
			files_633.append(file)
		else:
			files_785.append(file)

	fig, (ax, ax2) = plt.subplots(1, 2, figsize=[9, 5], width_ratios=[5, 1])
	ax2.axis('off')

	# read 488 data, sum where multiple
	data_488 = renishaw.wire_read(files_488[0], as_pandas=True)
	x_488 = data_488['XLST']
	y_488 = data_488['DATA']
	for file in files_488[1:]:
		data = renishaw.wire_read(file, as_pandas=True)
		y_488 += data['DATA']

	# read 633 data, sum where multiple
	data_633 = renishaw.wire_read(files_633[0], as_pandas=True)
	x_633 = data_633['XLST']
	y_633 = data_633['DATA']
	for file in files_633[1:]:
		data = renishaw.wire_read(file, as_pandas=True)
		y_633 += data['DATA']

	# read 785 data, sum where multiple
	data_785 = renishaw.wire_read(files_785[0], as_pandas=True)
	x_785 = data_785['XLST']
	y_785 = data_785['DATA']
	for file in files_785[1:]:
		data = renishaw.wire_read(file, as_pandas=True)
		y_785 += data['DATA']

	global pre_532, post_532
	post_532 = pd.read_csv(f'Campaign 4/All Experiments/Experiment_{number}/Postscan.csv', skiprows=3, names=['x', 'y'])
	pre_532 = pd.read_csv(f'Campaign 4/All Experiments/Experiment_{number}/Prescan.csv', skiprows=3, names=['x', 'y'])

	align = minimize_scalar(align_spectra)

	# ax.plot(post_532['x'], post_532['y'])
	post_532['y'] = np.subtract(post_532['y'], align.x + pre_532['y'])
	# ax.plot(post_532['x'], post_532['y'])

	# plt.show()
	# exit()


	x_532 = post_532['x']
	# y_532 = post_532['y']/si_area(post_532['x'], post_532['y']) - pre_532['y']/si_area(pre_532['x'], pre_532['y'])
	y_532 = post_532['y']
	y_532 = denoise_spectrum(y_532)

	# y_488 /= si_area(x_488, y_488)
	# y_532 /= si_area(x_532, y_532)
	# y_633 /= si_area(x_633, y_633)
	# y_785 /= si_area(x_785, y_785)

	left, right = find_biggest_bounds(x_532, y_532)

	control_488, diff_488 = rbms_in_biggest(x_488, y_488, left, right)
	control_633, diff_633 = rbms_in_biggest(x_633, y_633, left, right)
	control_785, diff_785 = rbms_in_biggest(x_785, y_785, left, right)

	bounds_488 = find_biggest_bounds(x_488, y_488)
	raw_control_488, _ = rbms_in_biggest(x_488, y_488, bounds_488[0], bounds_488[1])
	bounds_633 = find_biggest_bounds(x_633, y_633)
	raw_control_633, _ = rbms_in_biggest(x_633, y_633, bounds_633[0], bounds_633[1])
	bounds_785 = find_biggest_bounds(x_785, y_785)
	raw_control_785, _ = rbms_in_biggest(x_785, y_785, bounds_785[0], bounds_785[1])



	# print(488, control_488)
	# print(633, control_633)
	# print(785, control_785)

	y_785 /= y_785[x_785.between(150, 400)].max()
	y_633 /= y_633[x_633.between(150, 400)].max()
	y_532 /= y_532[x_532.between(150, 400)].max()
	y_488 /= y_488[x_488.between(150, 400)].max()

	rows_532_dotted = x_532.between(0, 201)
	rows_532_solid = x_532.between(200, 500)

	offset = 1
	ax.plot(x_785, y_785, color='xkcd:burnt red', label='785')
	ax.plot(x_633, y_633 + 1 * offset, color='xkcd:red', label='633')
	ax.plot(x_532[rows_532_dotted], y_532[rows_532_dotted] + 2*offset, color='xkcd:green', linestyle='dotted')
	ax.plot(x_532[rows_532_solid], y_532[rows_532_solid] + 2 * offset, color='xkcd:green', label='532')
	ax.plot(x_488, y_488 + 3 * offset, color='xkcd:bright teal', label='488')
	ax1 = ax.secondary_xaxis('top', functions=(r2d, r2d))
	ax1.set_xlabel('Diameter (nm)')
	ax1.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])

	ax.axvline(left, color='k', linestyle='--')
	ax.axvline(right, color='k', linestyle='--')

	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles[::-1], labels[::-1], loc='upper right', ncol=2)


	ax.set_xlim(150, 400)
	ax.set_ylim(0, 5)

	results_table = [
	f'ARES: {round(ares_control, 3)}',
	f'488: {round(control_488, 3)}', 
	f'633: {round(control_633, 3)}',
	f'785: {round(control_785, 3)}',
	]
	ax2.text(0, 0.5, '\n'.join(results_table), ha='left', va='center')

	summary['488 control'][i] = control_488
	summary['633 control'][i] = control_633
	summary['785 control'][i] = control_785
	summary['488 diff'][i] = diff_488
	summary['633 diff'][i] = diff_633
	summary['785 diff'][i] = diff_785

	summary['488 raw control'][i] = raw_control_488
	summary['633 raw control'][i] = raw_control_633
	summary['785 raw control'][i] = raw_control_785
	summary['488 raw center'][i] = np.mean(bounds_488)
	summary['633 raw center'][i] = np.mean(bounds_633)
	summary['785 raw center'][i] = np.mean(bounds_785)

	summary['Control Center'][i] = (left + right) / 2
	summary['Control Width'][i] = right - left
	

	y = round(experiment['G Band Area Result'], 3)
	d = round(experiment['Diameter Control Result'], 3)
	title = f'Experiment {number}  Y: {y}'
	fig.suptitle(title)

	fig.tight_layout()

	# plt.show()
	# exit()
	fig.savefig(f'RBM Waterfall Plots/Experiment_{number}_rbms.png', dpi=300)

	plt.close()

	fig, ax = plt.subplots(figsize=(8, 5))

	props = dict(boxstyle='round', facecolor='white', alpha=0.3)

	offset = 1
	rows = x_785.between(150, 400)
	ax.plot(x_785[rows], y_785[rows], color='xkcd:burnt red', label='785')
	ax.text(x_785[rows].tail(1), y_785[rows].tail(1), '785 nm', va='bottom', ha='right')
	ax.text(right+2, 0.9, str(round(control_785, 3)), ha='left', va='top', bbox=props)

	rows = x_633.between(150, 400)
	ax.plot(x_633[rows], y_633[rows] + 1*offset, color='xkcd:red', label='633')
	ax.text(x_633[rows].tail(1), y_633[rows].tail(1) + 1*offset, '633 nm', va='bottom', ha='right')
	ax.text(right+2, 1.9, str(round(control_633, 3)), ha='left', va='top', bbox=props)

	rows = x_532.between(150, 400)
	ax.plot(x_532[rows_532_dotted], y_532[rows_532_dotted] + 2*offset, color='xkcd:green', linestyle='dotted')
	ax.plot(x_532[rows_532_solid], y_532[rows_532_solid] + 2 * offset, color='xkcd:green', label='532')
	ax.text(x_532[rows].tail(1), y_532[rows][-1] + 2*offset, '532 nm', va='bottom', ha='right')
	ax.text(right+2, 2.9, str(round(ares_control, 3)), ha='left', va='top', bbox=props)

	rows = x_488.between(150, 400)
	ax.plot(x_488, y_488 + 3 * offset, color='xkcd:bright teal', label='488')
	ax.text(x_488[rows].tail(1), y_488[rows].tail(1) + 3*offset, '488 nm', va='bottom', ha='right')
	ax.text(right+2, 3.9, str(round(control_488, 3)), ha='left', va='top', bbox=props)

	ax.text(395, 4.35, f'Mean Agreement: {round(np.mean([control_488, control_633, control_785]), 3)}', ha='right', va='top', bbox=props)

	ax1 = ax.secondary_xaxis('top', functions=(r2d, r2d))
	ax.set_xlabel('Raman Shift (cm$^{-1}$)')
	ax.set_yticks([])
	ax1.set_xlabel('CNT Diameter (nm)')
	ax1.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])

	ax.axvline(left, color='k', linestyle='--')
	ax.axvline(right, color='k', linestyle='--')

	ax.set_xlim(150, 400)
	ax.set_ylim(0, 4.5)
	fig.tight_layout()

	letter = 'a'
	plothelp.subfig_label(fig, f'({letter})')
	fig.savefig(f'Reformatted Waterfalls/Experiment_{number}_rbms_{letter}.png', dpi=300)

	plt.close()

	# exit()

means = mean(summary['488 control'], summary['633 control'], summary['785 control'])
summary['Total Control'] = summary['488 control'] + summary['633 control'] + summary['785 control']
summary['Mean Control'] = means
summary['488 Agreement'] = summary['488 control'] / summary['Diameter Control Result']
summary['633 Agreement'] = summary['633 control'] / summary['Diameter Control Result']
summary['785 Agreement'] = summary['785 control'] / summary['Diameter Control Result']
summary['Mean Agreement'] = mean(summary['488 Agreement'], summary['633 Agreement'], summary['785 Agreement'])
summary.to_csv('processed summary.csv', index=False)
		








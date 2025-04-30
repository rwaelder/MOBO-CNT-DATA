import argparse
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.optimize import minimize_scalar
import sys

import helpers
import logger
import pyBayesThresh as pbt




def early_exit(results):
    print(json.dumps(results))
    sys.exit()

def align_spectra(c):
    global prescan, postscan
    rows = prescan['wavenumbers'].between(200, 800)
    return np.abs(np.sum(np.subtract(postscan['intensities'][rows], c + prescan['intensities'][rows])))

def denoise_spectrum(y_values, denoise_level=1, wavelet='sym4', noise_est='level_independent', thresh_rule='median'):
    intensities, en = pbt.wavelet_denoise(y_values, denoise_level, 
        wav_name=wavelet, noise_est=noise_est, thresh_rule=thresh_rule)
    
    return intensities
    

def analyze(spectrum_file, log, save_plot=True):
    sys.stderr = log
    analyzer_name = 'diameter_control_v3'
    logger.log_analyzer(analyzer_name, spectrum_file)
    results = {'results' : [0], 'errors' : [], 'labels' : ['Diameter Control Result']}

    postscan_file = spectrum_file
    prescan_file = os.path.join(os.path.dirname(postscan_file), 'Prescan.csv')
    noise_model_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'noise_model.csv')
    experiment_number = os.path.dirname(postscan_file).split('_')[-1]

    try:
        noise_model = helpers.read_ares_spectrum(noise_model_file)
        log.write('opened noise model')
    except FileNotFoundError:
        error_message = 'could not open noise model file'
        log.write(error_message)
        results['results'] = [-1]
        results['errors'] = error_message
        early_exit(results)

    global prescan, postscan

    try:
        postscan = helpers.read_ares_spectrum(postscan_file)
        log.write(f'opened postscan spectrum file {postscan_file}')
        
    except FileNotFoundError:
        error_message = f'could not open postscan spectrum file {postscan_file}'
        log.write(error_message)
        results['results'] = [-1]
        results['errors'].append(error_message)
        
        early_exit(results)
    
    try:
        prescan = helpers.read_ares_spectrum(prescan_file)
        log.write(f'opened prescan spectrum file {prescan_file}')
        
    except FileNotFoundError:
        error_message = f'could not open prescan spectrum file {prescan_file}'
        log.write(error_message)
        results['results'] = [-1]
        results['errors'].append(error_message)
        
        early_exit(results)

    global rbm_rows
    rbm_limits = (200, 350)
    rbm_rows = prescan['wavenumbers'].between(rbm_limits[0], rbm_limits[1])


    noise_mean = noise_model['intensities'][rbm_rows].mean()
    noise_sdev = noise_model['intensities'][rbm_rows].std()
    noise_thresh = 5*noise_sdev
    resolution = prescan['wavenumbers'].iloc[1] - prescan['wavenumbers'].iloc[0]

    align = minimize_scalar(align_spectra)

    processed = pd.DataFrame({'wavenumbers' : postscan['wavenumbers'][rbm_rows], 'intensities' : postscan['intensities'][rbm_rows] } )

    processed['subtracted'] = np.subtract(postscan['intensities'][rbm_rows], align.x + prescan['intensities'][rbm_rows])

    processed['sub_denoised'] = denoise_spectrum(processed['subtracted'])

    processed['sub_denoised_clip'] = np.clip(processed['sub_denoised'] - noise_thresh, a_min=0, a_max=None)

    rbm_area = np.trapz(processed['sub_denoised_clip'], x=processed['wavenumbers'])


    processed['integral'] = [np.trapz(processed['sub_denoised_clip'].iloc[:i], x=processed['wavenumbers'].iloc[:i]) for i in range(len(processed['sub_denoised_clip']))]
    processed['norm_integral'] = processed['integral'] / rbm_area
    processed['norm_integral_smoothed'] = denoise_spectrum(processed['norm_integral'])
    processed['gradient'] = np.gradient(processed['norm_integral_smoothed'], processed['wavenumbers']) * rbm_area
    processed['gradient'] = processed['gradient'].shift(-1)
    processed['grad_gradient'] = np.gradient(processed['gradient'], processed['wavenumbers'])
    processed['grad_gradient'] = processed['grad_gradient'].shift(-1)

    zero_crossings = np.where(np.diff(np.sign(processed['grad_gradient']) >= 0))[0]
    zero_crossings = [zc+1 for zc in zero_crossings[1:] if processed['grad_gradient'].iloc[zc] <= 0]
    zero_crossings.insert(0, 0)
    
    peak_areas = [processed['norm_integral_smoothed'].iloc[right]-processed['norm_integral_smoothed'].iloc[left] for left, right in zip(zero_crossings[:-1], zero_crossings[1:])]
    peak_widths = [processed['wavenumbers'].iloc[right] - processed['wavenumbers'].iloc[left] for left, right in zip(zero_crossings[:-1], zero_crossings[1:])]
    sort_indices = np.argsort(peak_areas)
    n_peaks = len([area for area in peak_areas if area >= 0.1])
    if n_peaks == 0:
        n_peaks = 100
    
    # get two biggest peaks
    try:
        dominant = peak_areas[sort_indices[-1]]
        dominant_width = peak_widths[sort_indices[-1]]
    except IndexError:
        dominant = 0
        dominant_width = 150
    try:
        subdominant = peak_areas[sort_indices[-2]]
    except IndexError:
        subdominant = 0
    
    # result = (dominant - subdominant) / n_peaks / n_peaks / dominant_width
    result = dominant
    results['results'] = [result]
    
    # plot

    raman_label = 'Raman Shift [cm$^{-1}$]'
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(1.5*6.4, 2*4.8))

    ax1.plot(prescan['wavenumbers'][rbm_rows], prescan['intensities'][rbm_rows] + align.x, label='offset prescan')
    ax1.plot(postscan['wavenumbers'][rbm_rows], postscan['intensities'][rbm_rows], label='postscan')
    ax1.legend()
    ax1.set_xlabel(raman_label)


    ax2.plot(processed['wavenumbers'], processed['subtracted'], label='raw subtraction')
    ax2.plot(processed['wavenumbers'], processed['sub_denoised'], label='smoothed')
    ax2.axhline(5*noise_sdev, color='k', linestyle='--', alpha=0.5, label='significance threshold')
    ax2.legend()
    ax2.set_xlabel(raman_label)


    ax3.plot(processed['wavenumbers'], processed['sub_denoised_clip'], label='processed spectrum')
    ax3.plot(processed['wavenumbers'], processed['gradient'], label='integral smoothed')
    ax3.set_xlabel(raman_label)
    #ax3.legend()

    ax4.plot(processed['wavenumbers'], processed['norm_integral'], label='normalized integral')
    ax4.plot(processed['wavenumbers'], processed['norm_integral_smoothed'], label='smoothed integral')
    ax4.set_xlabel(raman_label)
    

    ax5.plot(processed['wavenumbers'], processed['grad_gradient'])
    ax5.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel(raman_label)
    
    
    for z in zero_crossings:
        ax5.axvline(processed['wavenumbers'].iloc[z], color='k', linestyle='--', alpha=0.5)
        ax4.axhline(processed['norm_integral_smoothed'].iloc[z], color='k', linestyle='--', alpha=0.5)
        ax3.axvline(processed['wavenumbers'].iloc[z], color='k', linestyle='--', alpha=0.5)
    
    ax3.legend(['processed spectrum', 'integral smoothed', 'peak boundaries'])
    ax4.legend(['integral', 'smoothed integral', 'peak boundaries'])
    ax5.legend(['spectrum gradient', 'peak boundaries'])

    x_min, x_max = ax1.get_xlim()
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_xlim(x_min, x_max)

    ax6.axis('off')
    '''
    report_message = [f'Area: {round(rbm_area, 3)}']
    report_message.append(f'Dominant peak share: {round(dominant, 3)}')
    report_message.append(f'Subdominant peak share: {round(subdominant, 3)}')
    report_message.append(f'Dominant peak width: {round(dominant_width, 3)}')
    report_message.append(f'Peaks: {n_peaks}')
    report_message.append(f'Result: {round(result, 3)}')
    '''
    report_message = [f'Result: {round(result, 4)}']
    log.write('\t'.join(report_message))
    
    bbox_dict = dict(boxstyle='round', ec=(0, 0, 0), fc=(0.97, 0.97, 0.97))
    ax6.text(0, 0, '\n'.join(report_message), ha='left', fontsize='x-large', bbox=bbox_dict)
    ax6.text(0, 0.9, r'$\frac{\mathrm{Dominant peak area}}{\mathrm{Total RBM area}}$', ha='left', fontsize='x-large', bbox=bbox_dict)

    fig.tight_layout()
    
    if save_plot:
        try:
            figure_file = os.path.join(os.path.dirname(postscan_file), f'{analyzer_name}_result.png')
            fig.savefig(figure_file, dpi=300)
            log.write(f'png summary figure saved as {figure_file}')
        except:
            log.write(f'could not save png summary figure {figure_file}')
    
        try:
            figure_file = os.path.join(os.path.dirname(postscan_file), f'{analyzer_name}_result.pdf')
            fig.savefig(figure_file)
            log.write(f'pdf summary figure saved as {figure_file}')
        except:
            log.write(f'could not save pdf summary figure {figure_file}')
    
    else:
        plt.show()
    
    return results


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    log_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'diameter_control_v3_logs', f'{starttime.strftime("%Y-%m-%d %H-%M-%S")}.log')
    log = logger.AnalyzerLog(log_file)
    
    spectrum_file = sys.argv[1]
    results = analyze(spectrum_file, log)
    log.write(f'execution time: {datetime.datetime.now()-starttime}')
    print(json.dumps(results))
    
    
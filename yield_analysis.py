import argparse
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

import helpers
import logger




def analyze(spectrum_file, log, save_plot=True):
    sys.stderr = log
    analyzer_name = 'g_band_area'
    logger.log_analyzer(analyzer_name, spectrum_file)
    results = {'results' : [0], 'errors' : [], 'labels' : ['G Band Area Result']}
    experiment_number = os.path.dirname(spectrum_file).split('_')[-1]

    try:
        spectrum = helpers.read_ares_spectrum(spectrum_file)
        log.write(f'opened spectrum file {spectrum_file}')
        
    except FileNotFoundError:
        log.write(f'could not open spectrum file {spectrum_file}')
        results['results'] = [-1]
        results['errors'].append(f'could not open spectrum file {spectrum_file}')
        print(json.dumps(results))
        sys.exit()

    # calculate and subtract noise floor 
    dead_bands = [(600, 800), (1100, 1200), (1800, 2000), (2800, 3000)]
    integral_limits = (1425, 1650)

    spectrum = helpers.subtract_noise_floor(spectrum, dead_bands, poly_degree=0, clip_0=True)

    spectrum = helpers.si_peak_area_normalize(spectrum)

    g_band_area = helpers.integrate(spectrum, integral_limits[0], integral_limits[1])

    results['results'] = [g_band_area]

    log.write(json.dumps(results))


    # plot result in experiment folder
    integral_rows = spectrum.iloc[:, 0].between(integral_limits[0], integral_limits[1])
        
    plt.rc('font', size=13)
    plt.rc('axes', labelsize=13)    
    
    fig, ax = plt.subplots()
    fig.suptitle(f'Experiment {experiment_number}')
    ax.plot(spectrum.iloc[:, 0], spectrum.iloc[:, 1])
    ax.fill_between(spectrum.iloc[:, 0][integral_rows], spectrum.iloc[:, 1][integral_rows], 0, color='tab:orange')
        
    ax.set_xlabel('Raman Shift [cm$^{-1}$]')
    ax.set_ylabel('Intensity [a.u.]')

    si_max = spectrum.loc[spectrum['wavenumbers'].between(500, 540), 'intensities'].max()
    
    ax.set_ylim(ax.get_ylim()[0], 1.5*si_max)
    
    _, xmax = ax.get_xlim()
    _, ymax = ax.get_ylim()
    ax.text(xmax*0.85, ymax*0.85, f'G Band Area: {round(g_band_area, 5)}', ha='right', bbox=dict(boxstyle='round', ec=(0, 0, 0), fc=(0.95, 0.95, 0.95)))

    
    fig.tight_layout()
    if save_plot:
        try:
            figure_file = os.path.join(os.path.dirname(spectrum_file), f'{analyzer_name}_result.png')
            fig.savefig(figure_file, dpi=300)
            log.write(f'png summary figure saved as {figure_file}')
        except:
            log.write(f'could not save png summary figure {figure_file}')
    
        try:
            figure_file = os.path.join(os.path.dirname(spectrum_file), f'{analyzer_name}_result.pdf')
            fig.savefig(figure_file)
            log.write(f'pdf summary figure saved as {figure_file}')
        except:
            log.write(f'could not save pdf summary figure {figure_file}')

    return results


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    log_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'g_band_area_logs', f'{starttime.strftime("%Y-%m-%d %H-%M-%S")}.log')
    log = logger.AnalyzerLog(log_file)

    spectrum_file = sys.argv[1]
    results = analyze(spectrum_file, log)
    log.write(f'execution time: {datetime.datetime.now()-starttime}')
    print(json.dumps(results))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:41:34 2019
Applies lowess-based detrending pipeline to lcs which can come from a variety of sources (TESS, K2 etc)
@author: phrhzn
"""

import batman
import lightkurve
import pickle
import random
import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import statsmodels.api as sm
from astropy.timeseries import LombScargle
#from lightkurve import search_lightcurvefile
#from lc_download_methods_late_sectors import diff_image_lc_download2, two_min_lc_download, eleanor_lc_download, raw_FFI_lc_download
from lc_download_methods import diff_image_lc_download, two_min_lc_download, eleanor_lc_download, raw_FFI_lc_download
from scipy.signal import find_peaks
from astropy.timeseries import BoxLeastSquares
#from astropy.io import ascii
from astropy.table import Table
from scipy import interpolate
#from astropy import constants as const
from remove_tess_systematics import clean_tess_lc
from fits_handling import get_lc_from_fits

############################################################


def ffi_lowess_detrend(save_path = '', sector = 1, target_ID_list = [], pipeline = '2min', 
                       multi_sector = False, use_peak_cut = False, 
                       binned = False, transit_mask = False, injected_planet = 'user_defined', 
                       injected_rp = 0.1, injected_per = 8.0, detrending = 'lowess_partial', 
                       single_target_ID = ['HIP 1113'], n_bins = 30):
    for target_ID in target_ID_list:
        try:
            lc_30min = lightkurve.lightcurve.TessLightCurve(time = [],flux=[])
            if multi_sector != False:
                sap_lc, pdcsap_lc = two_min_lc_download(target_ID, sector = multi_sector[0], 
                                                        from_file = False)
                lc_30min = pdcsap_lc
                nancut = np.isnan(lc_30min.flux) | np.isnan(lc_30min.time)
                lc_30min = lc_30min[~nancut]
                clean_time, clean_flux, clean_flux_err = clean_tess_lc(lc_30min.time, lc_30min.flux, lc_30min.flux_err, target_ID, multi_sector[0], save_path)
                lc_30min.time = clean_time
                lc_30min.flux = clean_flux
                lc_30min.flux_err = clean_flux_err
                for sector_num in multi_sector[1:]:
                    sap_lc_new, pdcsap_lc_new = two_min_lc_download(target_ID, sector_num, from_file = False)
                    lc_30min_new = pdcsap_lc_new
                    nancut = np.isnan(lc_30min_new.flux) | np.isnan(lc_30min_new.time)
                    lc_30min_new = lc_30min_new[~nancut]
                    clean_time, clean_flux, clean_flux_err = clean_tess_lc(lc_30min_new.time, lc_30min_new.flux, lc_30min_new.flux_err, target_ID, sector_num, save_path)
                    lc_30min_new.time = clean_time
                    lc_30min_new.flux = clean_flux
                    lc_30min_new.flux_err = clean_flux_err
                    lc_30min = lc_30min.append(lc_30min_new)
#                    lc_30min.flux = lc_30min.flux.append(lc_30min_new.flux)
#                    lc_30min.time = lc_30min.time.append(lc_30min_new.time)
#                    lc_30min.flux_err = lc_30min.flux_err.append(lc_30min_new.flux_err)
            else:
                try:
                    if pipeline == 'DIA':
                        lc_30min, filename = diff_image_lc_download(target_ID, sector, plot_lc = True, save_path = save_path, from_file = True)
                    elif pipeline == '2min':
                        sap_lc, pdcsap_lc = two_min_lc_download(target_ID, sector = sector, from_file = False)
                        lc_30min = pdcsap_lc
                        nancut = np.isnan(lc_30min.flux) | np.isnan(lc_30min.time)
                        lc_30min = lc_30min[~nancut]
                    elif pipeline == 'eleanor':
                        raw_lc, corr_lc, pca_lc = eleanor_lc_download(target_ID, sector, from_file = False, save_path = save_path, plot_pca = False)
                        lc_30min = pca_lc
                    elif pipeline == 'from_file':
                        lcf = lightkurve.open('tess2019140104343-s0012-0000000212461524-0144-s_lc.fits')
                        lc_30min = lcf.PDCSAP_FLUX
                    elif pipeline == 'from_pickle':
                        with open('Original_time.pkl','rb') as f:
                            original_time = pickle.load(f)
                        with open('Original_flux.pkl','rb') as f:
                            original_flux = pickle.load(f)
                        lc_30min = lightkurve.lightcurve.TessLightCurve(time = original_time,flux=original_flux)
                    elif pipeline == 'raw':
                        lc_30min = raw_FFI_lc_download(target_ID, sector, plot_tpf = False, plot_lc = True, save_path = save_path, from_file = False)
                        pipeline = "raw"
                    else: 
                        print('Invalid pipeline')
    
                except:
                		print('Lightcurve for {} not available'.format(target_ID))
        
            ################### Clean TESS lc pointing systematics ########################
            if multi_sector == False:
                clean_time, clean_flux, clean_flux_err = clean_tess_lc(lc_30min.time, lc_30min.flux, lc_30min.flux_err, target_ID, sector, save_path)
                lc_30min.time = clean_time
                lc_30min.flux = clean_flux
                lc_30min.flux_err = clean_flux_err
            
            ######################### Find rotation period ################################
            normalized_flux = np.array(lc_30min.flux)/np.median(lc_30min.flux)
            
            # From Lomb-Scargle
            freq = np.arange(0.04,4.1,0.00001)
            power = LombScargle(lc_30min.time, normalized_flux).power(freq)
            ls_fig = plt.figure()
            plt.plot(freq, power, c='k', linewidth = 1)
            plt.xlabel('Frequency')
            plt.ylabel('Power')
            plt.title('{} LombScargle Periodogram for original lc'.format(target_ID))
            #ls_plot.show(block=True)
    #        ls_fig.savefig(save_path + '{} - Lomb-Scargle Periodogram for original lc.png'.format(target_ID))
            plt.close(ls_fig)
            i = np.argmax(power)
            freq_rot = freq[i]
            p_rot = 1/freq_rot
            print('Rotation Period = {:.3f}d'.format(p_rot))
            
            # From BLS
            durations = np.linspace(0.05, 1, 22) * u.day
            model = BoxLeastSquares(lc_30min.time*u.day, normalized_flux)
            results = model.autopower(durations, frequency_factor=1.0)
            rot_index = np.argmax(results.power)
            rot_period = results.period[rot_index]
            print("Rotation Period from BLS of original = {}d".format(rot_period))
            
            ########################### batman stuff ######################################
            if injected_planet != False:
                params = batman.TransitParams()       #object to store transit parameters
                params.t0 = -10.0                      #time of inferior conjunction
                params.per = 8.0
                params.rp = 0.1
                table_data = Table.read("BANYAN_XI-III_members_with_TIC.csv" , format='ascii.csv')
                i = list(table_data['main_id']).index(target_ID)
                m_star = table_data['Stellar Mass'][i]*m_Sun
                r_star = table_data['Stellar Radius'][i]*r_Sun*1000
                params.a = (((G*m_star*(params.per*86400.)**2)/(4.*(np.pi**2)))**(1./3))/r_star
                if np.isnan(params.a) == True:
                    params.a = 17. #semi-major axis (in units of stellar radii)
                params.inc = 90.
                params.ecc = 0.
                params.w = 90.                        #longitude of periastron (in degrees)
                params.limb_dark = "nonlinear"        #limb darkening model
                params.u = [0.5, 0.1, 0.1, -0.1]      #limb darkening coefficients [u1, u2, u3, u4]
                
                if injected_planet == 'user_defined':
                    # Build planet from user specified parameters
                    params.per = injected_per                      #orbital period (days)
                    params.rp = injected_rp                       #planet radius (in units of stellar radii)
                    params.a = (((G*m_star*(params.per*86400.)**2)/(4.*(np.pi**2)))**(1./3))/r_star
                    if np.isnan(params.a) == True:
                        params.a =  17                            # Recalculates a if period has changed
                    params.inc = 90.                      #orbital inclination (in degrees)
                    params.ecc = 0.                       #eccentricity
                else:
                    raise NameError('Invalid inputfor injected planet')
    
                # Defines times at which to calculate lc and models batman lc
                t = np.linspace(-13.9165035, 13.9165035, len(lc_30min.time))
                index = int(len(lc_30min.time)//2)
                mid_point = lc_30min.time[index]
                t = lc_30min.time - lc_30min.time[index]
                m = batman.TransitModel(params, t)
                t += lc_30min.time[index]
                batman_flux = m.light_curve(params)
                batman_model_fig = plt.figure()
                plt.scatter(lc_30min.time, batman_flux, s = 2, c = 'k')
                plt.xlabel("Time - 2457000 (BTJD days)")
                plt.ylabel("Relative flux")
                plt.title("batman model transit for {}R ratio".format(params.rp))
                #batman_model_fig.savefig(save_path + "batman model transit for {}d {}R planet.png".format(params.per,params.rp))
                #plt.close(batman_model_fig)
                plt.show()
            
            ################################# Combining ###################################
            if injected_planet != False:
                combined_flux = np.array(lc_30min.flux)/np.median(lc_30min.flux) + batman_flux -1
                
                injected_transit_fig = plt.figure()
                plt.scatter(lc_30min.time, combined_flux, s = 2, c = 'k')
                plt.xlabel("Time - 2457000 (BTJD days)")
                plt.ylabel("Relative flux")
                plt.title("{} with injected transits for a {}R {}d planet to star ratio.".format(target_ID, params.rp, params.per))
                ax = plt.gca()
                for n in range(int(-1*8/params.per),int(2*8/params.per+2)):
                    ax.axvline(params.t0+n*params.per+mid_point, ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
                ax.axvline(params.t0+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
                ax.axvline(params.t0+params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
                ax.axvline(params.t0+2*params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    #            injected_transit_fig.savefig(save_path + "{} - Injected transits fig - Period {} - {}R transit.png".format(target_ID, params.per, params.rp))
    #            plt.close(injected_transit_fig)
                plt.show()
        
        ############################## Removing peaks #################################
            if injected_planet == False:
                combined_flux = np.array(lc_30min.flux)/np.median(lc_30min.flux)
#            combined_flux = lc_30min.flux
            if use_peak_cut == True:
                peaks, peak_info = find_peaks(combined_flux, prominence = 0.001, width = 15)
                troughs, trough_info = find_peaks(-combined_flux, prominence = -0.001, width = 15)
                flux_peaks = combined_flux[peaks]
                flux_troughs = combined_flux[troughs]
                amplitude_peaks = ((flux_peaks[0]-1) + (1-flux_troughs[0]))/2
                print("Absolute amplitude of main variability = {}".format(amplitude_peaks))
                peak_location_fig = plt.figure()
                plt.scatter(lc_30min.time, combined_flux, s = 2, c = 'k')
                plt.plot(lc_30min.time[peaks], combined_flux[peaks], "x")
                plt.plot(lc_30min.time[troughs], combined_flux[troughs], "x", c = 'r')
                #peak_location_fig.savefig(save_path + "{} - Peak location fig.png".format(target_ID))
                peak_location_fig.show()
#                plt.close(peak_location_fig)
                
                near_peak_or_trough = [False]*len(combined_flux)
                
                for i in peaks:
                    for j in range(len(lc_30min.time)):
                        if abs(lc_30min.time[j] - lc_30min.time[i]) < 0.1:
                            near_peak_or_trough[j] = True
                
                for i in troughs:
                    for j in range(len(lc_30min.time)):
                        if abs(lc_30min.time[j] - lc_30min.time[i]) < 0.1:
                            near_peak_or_trough[j] = True
                
                near_peak_or_trough = np.array(near_peak_or_trough)
                
                t_cut = lc_30min.time[~near_peak_or_trough]
                flux_cut = combined_flux[~near_peak_or_trough]
                flux_err_cut = lc_30min.flux_err[~near_peak_or_trough]

                # Plot new cut version
                peak_cut_fig = plt.figure()
                plt.scatter(t_cut,flux_cut, c = 'k', s = 2)
                plt.xlabel('Time - 2457000 [BTJD days]')
                plt.ylabel("Relative flux")
                plt.title('{} lc after removing peaks/troughs'.format(target_ID))
                ax = plt.gca()
                #peak_cut_fig.savefig(save_path + "{} - Peak cut fig.png".format(target_ID))
                peak_cut_fig.show()
#                plt.close(peak_cut_fig)
            else:
                 t_cut = lc_30min.time
                 flux_cut = combined_flux
                 flux_err_cut = lc_30min.flux_err
                 print('Flux cut skipped')
                 
        ############################## Apply transit mask #########################
    
            if transit_mask == True:
                period = 8.138
                epoch = 1332.31
                duration = 0.15
                phase = np.mod(t_cut-epoch-period/2,period)/period
                
                near_transit = [False]*len(flux_cut)
                
                for i in range(len(t_cut)):
                    if abs(phase[i] - 0.5) < duration/period:
                        near_transit[i] = True
                
                near_transit = np.array(near_transit)
                
                t_masked = t_cut[~near_transit]
                flux_masked = flux_cut[~near_transit]
                flux_err_masked = flux_err_cut[~near_transit]
                t_new = t_cut[near_transit]
                
                f = interpolate.interp1d(t_masked,flux_masked, kind = 'quadratic')

                flux_new = f(t_new)
                interpolated_fig = plt.figure()
#                plt.scatter(t_masked, flux_masked, s = 2, c = 'k')
                plt.scatter(t_cut, flux_cut, s = 2, c = 'k')
                plt.scatter(t_new,flux_new, s=2, c = 'r')
                plt.xlabel('Time - 2457000 [BTJD days]')
                plt.ylabel('Relative flux')
#                interpolated_fig.savefig(save_path + "{} - Interpolated over transit mask fig.png".format(target_ID))
                
                t_transit_mask = np.concatenate((t_masked,t_new), axis = None)
                flux_transit_mask = np.concatenate((flux_masked,flux_new), axis = None)
                
                sorted_order = np.argsort(t_transit_mask)
                t_transit_mask = t_transit_mask[sorted_order]
                flux_transit_mask = flux_transit_mask[sorted_order]
        
        
        ############################## LOWESS detrending ##############################
            
            # Full lc
            if detrending == 'lowess_full':
                full_lowess_flux = np.array([])
                if transit_mask == True:
                    lowess = sm.nonparametric.lowess(flux_transit_mask, t_transit_mask, frac=0.03)
                else:
                    lowess = sm.nonparametric.lowess(flux_cut, t_cut, frac=0.03)
                
                overplotted_lowess_full_fig = plt.figure()
                plt.scatter(t_cut,flux_cut, c = 'k', s = 2)
                plt.plot(lowess[:, 0], lowess[:, 1])
                plt.title('{} lc with overplotted lowess full lc detrending'.format(target_ID))
                plt.xlabel('Time - 2457000 [BTJD days]')
                plt.ylabel('Relative flux')
                #overplotted_lowess_full_fig.savefig(save_path + "{} lc with overplotted LOWESS full lc detrending.png".format(target_ID))
                plt.show()
#                plt.close(overplotted_lowess_full_fig)
                
                residual_flux_lowess = flux_cut/lowess[:,1]
                full_lowess_flux = np.concatenate((full_lowess_flux,lowess[:,1]))
                
                lowess_full_residuals_fig = plt.figure()
                plt.scatter(t_cut,residual_flux_lowess, c = 'k', s = 2)
                plt.title('{} lc after lowess full lc detrending'.format(target_ID))
                plt.xlabel('Time - 2457000 [BTJD days]')
                plt.ylabel('Relative flux')
                ax = plt.gca()
                #ax.axvline(params.t0+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
                #ax.axvline(params.t0+params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
                #ax.axvline(params.t0+2*params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
                #ax.axvline(params.t0-params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
                #lowess_full_residuals_fig.savefig(save_path + "{} lc after LOWESS full lc detrending.png".format(target_ID))
                plt.show()
                #plt.close(lowess_full_residuals_fig)
                
                
            # Partial lc
            if detrending == 'lowess_partial':
                time_diff = np.diff(t_cut)
                residual_flux_lowess = np.array([])
                time_from_lowess_detrend = np.array([])
                full_lowess_flux = np.array([])
                
                overplotted_detrending_fig = plt.figure()
                plt.scatter(t_cut,flux_cut, c = 'k', s = 2)
                plt.xlabel('Time - 2457000 [BTJD days]')
                plt.ylabel("Normalized flux")
                plt.title('{} lc with overplotted detrending'.format(target_ID))
                
                low_bound = 0
                if pipeline == '2min':
                    n_bins = 450
                else:
                    n_bins = n_bins
                for i in range(len(t_cut)-1):
                    if time_diff[i] > 0.1:
                        high_bound = i+1
                        
                        t_section = t_cut[low_bound:high_bound]
                        flux_section = flux_cut[low_bound:high_bound]
                        if len(t_section)>=n_bins:
                            if transit_mask == True:
                                lowess = sm.nonparametric.lowess(flux_transit_mask[low_bound:high_bound], t_transit_mask[low_bound:high_bound], frac=n_bins/len(t_section))
                            else:
                                lowess = sm.nonparametric.lowess(flux_section, t_section, frac=n_bins/len(t_section))
                            lowess_flux_section = lowess[:,1]
                            plt.plot(t_section, lowess_flux_section, '-')
                            
                            residuals_section = flux_section/lowess_flux_section
                            residual_flux_lowess = np.concatenate((residual_flux_lowess,residuals_section))
                            time_from_lowess_detrend = np.concatenate((time_from_lowess_detrend,t_section))
                            full_lowess_flux = np.concatenate((full_lowess_flux,lowess_flux_section))
                            low_bound = high_bound
                        else:
                            print('LOWESS skipped one gap at {}'.format(t_section[-1]))
                
                # Carries out same process for final line (up to end of data)        
                high_bound = len(t_cut)
                t_section = t_cut[low_bound:high_bound]
                flux_section = flux_cut[low_bound:high_bound]
                if transit_mask == True:
                    lowess = sm.nonparametric.lowess(flux_transit_mask[low_bound:high_bound], t_transit_mask[low_bound:high_bound], frac=n_bins/len(t_section))
                else:
                    lowess = sm.nonparametric.lowess(flux_section, t_section, frac=n_bins/len(t_section))
                lowess_flux_section = lowess[:,1]
                plt.plot(t_section, lowess_flux_section, '-')
#                if injected_planet != False:
#                    overplotted_detrending_fig.savefig(save_path + "{} - Overplotted lowess detrending - partial lc - {}R {}d injected planet.png".format(target_ID, params.rp, params.per))
#                else:
#                    overplotted_detrending_fig.savefig(save_path + "{} - Overplotted lowess detrending - partial lc".format(target_ID))
                overplotted_detrending_fig.show()
#                plt.close(overplotted_detrending_fig)
                
                residuals_section = flux_section/lowess_flux_section
                residual_flux_lowess = np.concatenate((residual_flux_lowess,residuals_section))
                time_from_lowess_detrend = np.concatenate((time_from_lowess_detrend,t_section))
                full_lowess_flux = np.concatenate((full_lowess_flux,lowess_flux_section))
                
                residuals_after_lowess_fig = plt.figure()
                plt.scatter(time_from_lowess_detrend,residual_flux_lowess, c = 'k', s = 2)
                plt.title('{} lc after LOWESS partial lc detrending'.format(target_ID))
                plt.xlabel('Time - 2457000 [BTJD days]')
                plt.ylabel('Relative flux')
                #ax = plt.gca()
                #ax.axvline(params.t0+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
                #ax.axvline(params.t0+params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
                #ax.axvline(params.t0+2*params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
                #ax.axvline(params.t0-params.per+lc_30min.time[index], ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
#                if injected_planet != False:
#                    residuals_after_lowess_fig.savefig(save_path + "{} lc after LOWESS partial lc detrending - {}R {}d injected planet.png".format(target_ID, params.rp, params.per))
#                else:
#                    residuals_after_lowess_fig.savefig(save_path + "{} lc after LOWESS partial lc detrending".format(target_ID))
                residuals_after_lowess_fig.show()
#                plt.close(residuals_after_lowess_fig)     
            
        #    ###################### Periodogram Construction ##################
        
            # Create periodogram
            durations = np.linspace(0.05, 1, 22) * u.day
            if detrending == 'lowess_full' or detrending == 'lowess_partial':
                BLS_flux = residual_flux_lowess
            else:
                BLS_flux = combined_flux
            model = BoxLeastSquares(t_cut*u.day, BLS_flux)
            results = model.autopower(durations, minimum_n_transit=3,frequency_factor=1.0)
            
            # Find the period and epoch of the peak
            index = np.argmax(results.power)
            period = results.period[index]
            #print(results.period)
            t0 = results.transit_time[index]
            duration = results.duration[index]
            transit_info = model.compute_stats(period, duration, t0)
            print(transit_info)
            
            epoch = transit_info['transit_times'][0]
            
            periodogram_fig, ax = plt.subplots(1, 1)
            
            # Highlight the harmonics of the peak period
            ax.axvline(period.value, alpha=0.4, lw=3)
            for n in range(2, 10):
                ax.axvline(n*period.value, alpha=0.4, lw=1, linestyle="dashed")
                ax.axvline(period.value / n, alpha=0.4, lw=1, linestyle="dashed")
            
            # Plot and save the periodogram
            ax.plot(results.period, results.power, "k", lw=0.5)
            ax.set_xlim(results.period.min().value, results.period.max().value)
            ax.set_xlabel("period [days]")
            ax.set_ylabel("log likelihood")
#            ax.set_title('{} - BLS Periodogram after {} detrending - {}R {}d injected planet'.format(target_ID, detrending, params.rp, params.per))
            ax.set_title('{} - BLS Periodogram after {} detrending'.format(target_ID, detrending))
#            periodogram_fig.savefig(save_path + '{} - BLS Periodogram after lowess partial detrending - {}R {}d injected planet.png'.format(target_ID, params.rp, params.per))
#            periodogram_fig.savefig(save_path + '{} - BLS Periodogram after lowess partial detrending.png'.format(target_ID))
#            plt.close(periodogram_fig)
            periodogram_fig.show()   
        	  
        
    ################################## Phase folding ##########################
            # Find indices of 2nd and 3rd peaks of periodogram
            all_peaks = scipy.signal.find_peaks(results.power, width = 5, distance = 10)[0]
            all_peak_powers = results.power[all_peaks]
            sorted_power_indices = np.argsort(all_peak_powers)
            sorted_peak_powers = all_peak_powers[sorted_power_indices]
    #        sorted_peak_periods = results.period[sorted_power_indices]
            
            # Find info for 2nd largest peak in periodogram
            index_peak_2 = np.where(results.power==sorted_peak_powers[-2])[0]
            period_2 = results.period[index_peak_2[0]]
            t0_2 = results.transit_time[index_peak_2[0]]
            
            # Find info for 3rd largest peak in periodogram
            index_peak_3 = np.where(results.power==sorted_peak_powers[-3])[0]
            period_3 = results.period[index_peak_3[0]]
            t0_3 = results.transit_time[index_peak_3[0]]
                    
            phase_fold_plot(t_cut, BLS_flux, period.value, t0.value, target_ID, save_path, '{} {} residuals folded by Periodogram Max ({:.3f} days)'.format(target_ID, detrending, period.value))
            period_to_test = p_rot
            t0_to_test = 1332
            period_to_test2 = period_2.value
            t0_to_test2 = t0_2.value
            period_to_test3 = period_3.value
            t0_to_test3 = t0_3.value      
            phase_fold_plot(t_cut, BLS_flux, p_rot, t0_to_test, target_ID, save_path, '{} folded by rotation period ({} days)'.format(target_ID,period_to_test))
            phase_fold_plot(t_cut, BLS_flux, period_to_test2, t0_to_test2, target_ID, save_path, '{} detrended lc folded by 2nd largest peak ({:0.4} days)'.format(target_ID,period_to_test2))
            phase_fold_plot(t_cut, BLS_flux, period_to_test3, t0_to_test3, target_ID, save_path, '{} detrended lc folded by 3rd largest peak ({:0.4} days)'.format(target_ID,period_to_test3))
            #variability_table.add_row([target_ID,p_rot,rot_period,amplitude_peaks])
            
            ############################# Eyeballing ##############################
            """
            Generate 2 x 2 eyeballing plot
            """
            eye_balling_fig, axs = plt.subplots(2,2, figsize = (16,10),  dpi = 120)
    
            # Original DIA with injected transits setup
            axs[0,0].scatter(lc_30min.time, combined_flux, s=1, c= 'k')
            axs[0,0].set_ylabel('Normalized Flux')
            axs[0,0].set_xlabel('Time')
            axs[0,0].set_title('{} - {} light curve'.format(target_ID, 'DIA'))
            #for n in range(int(-1*8/params.per),int(2*8/params.per+2)):
            #    axs[0,0].axvline(params.t0+n*params.per+mid_point, ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
            
            # Detrended figure setup
            axs[0,1].scatter(t_cut, BLS_flux, c = 'k', s = 1, label = '{} residuals after {} detrending'.format(target_ID,detrending))
#            axs[0,1].set_title('{} residuals after {} detrending - Sector {}'.format(target_ID, detrending, sector))
            axs[0,1].set_title('{} residuals after {} detrending - Sectors 14-18'.format(target_ID, detrending))
            axs[0,1].set_ylabel('Normalized Flux')
            axs[0,1].set_xlabel('Time - 2457000 [BTJD days]')
#            binned_time, binned_flux = bin(t_cut, BLS_flux, binsize=15, method='mean')
#            axs[0,1].scatter(binned_time, binned_flux, c='r', s=4)
            #for n in range(int(-1*8/params.per),int(2*8/params.per+2)):
            #    axs[0,1].axvline(params.t0+n*params.per+mid_point, ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
            
            # Periodogram setup
            axs[1,0].plot(results.period, results.power, "k", lw=0.5)
            axs[1,0].set_xlim(results.period.min().value, results.period.max().value)
            axs[1,0].set_xlabel("period [days]")
            axs[1,0].set_ylabel("log likelihood")
            axs[1,0].set_title('{} - BLS Periodogram of residuals'.format(target_ID))
            axs[1,0].axvline(period.value, alpha=0.4, lw=3)
            for n in range(2, 10):
                axs[1,0].axvline(n*period.value, alpha=0.4, lw=1, linestyle="dashed")
                axs[1,0].axvline(period.value / n, alpha=0.4, lw=1, linestyle="dashed")
            
            # Folded or zoomed plot setup
            epoch = t0.value
            period = period.value
            phase = np.mod(t_cut-epoch-period/2,period)/period 
            axs[1,1].scatter(phase, BLS_flux, c='k', s=1)
            axs[1,1].set_title('{} Lightcurve folded by {:0.4} days'.format(target_ID, period))
            axs[1,1].set_xlabel('Phase')
            axs[1,1].set_ylabel('Normalized Flux')
#            binned_phase, binned_lc = bin(phase, BLS_flux, binsize=15, method='mean')
#            plt.scatter(binned_phase, binned_lc, c='r', s=4)
        
            eye_balling_fig.tight_layout()
#            eye_balling_fig.savefig(save_path + '{} - Full eyeballing fig.pdf'.format(target_ID))
#            plt.close(eye_balling_fig)
            plt.show()
            
            ########################### ADDING INFO ROWS ######################
#            sensitivity_table.add_row([target_ID,sector,pipeline,params.per,params.a,params.rp,period,np.max(results.power),period_2.value,period_3.value])
            
        except RuntimeError:
            print('No DiffImage lc exists for {}'.format(target_ID))
        except:
            print('Some other error for {}'.format(target_ID))  
    return t_cut, BLS_flux, phase, epoch, period


def phase_fold_plot(t, lc, period, epoch, target_ID, save_path, title, binned = False):
    """
    Phase-folds the lc by the given period, and plots a phase-folded light-curve
    for the object of interest
    """
    phase = np.mod(t-epoch-period/2,period)/period 
    
    phase_fold_fig  = plt.figure()
    plt.scatter(phase, lc, c='k', s=2)
    plt.title(title)
    plt.xlabel('Phase')
    plt.ylabel('Normalized Flux')
    if binned == True:
        binned_phase, binned_lc = bin(phase, lc, binsize=15, method='mean')
        plt.scatter(binned_phase, binned_lc, c='r', s=4)
#    plt.savefig(save_path + '{} - Phase folded by {} days.png'.format(target_ID, period))
    plt.show()
#    plt.close(phase_fold_fig)
 
 
def bin(time, flux, binsize=15, method='mean'):
    """Bins a lightcurve in blocks of size `binsize`.
    n.b. based on the one from eleanor
    """
    available_methods = ['mean', 'median']
    if method not in available_methods:
        raise ValueError("method must be one of: {}".format(available_methods))
    methodf = np.__dict__['nan' + method]

    n_bins = len(flux) // binsize
    indexes = np.array_split(np.arange(len(time)), n_bins)
    binned_time = np.array([methodf(time[i]) for i in indexes])
    binned_flux = np.array([methodf(flux[i]) for i in indexes])

    return binned_time, binned_flux

# Set overall figsize
plt.rcParams['figure.figsize'] = (8.5,4)
plt.rcParams['savefig.dpi'] = 180

########################## Constants ##########################################

G = 6.6743*10**-11 #m^3.kg^-1.s^-2
m_Sun = 1.9891*10**30 #kg
r_Sun = 695510 #km
r_Jup = 69911  #km
r_Nep = 24622  #km
r_Earth = 6371 #km
au = 149597871 #km

########################## INPUTS #####################################################
save_path = ''
sector = 1
pipeline = 'DIA'
multi_sector = False #[14,15,16,17,18]
use_peak_cut = False
binned = False
transit_mask = False
injected_planet = False      # Can be 'user_defined' or False
detrending = 'lowess_partial' # Can be 'lowess_full', 'lowess_partial', OR 'None'
n_bins = 30
single_target_ID = ['J0635-5737']
######################################################################################

# Set up table to collect all info on any periodic main stellar variability
#variability_table = Table({'Name':[],'LS_Period':[],'BLS_Period':[],'Var_Amplitude':[]},names=['Name','LS_Period','BLS_Period','Var_Amplitude'])
#variability_table['Name'] = variability_table['Name'].astype(str)

# Set up table for collection of sensitivity analysis info
#sensitivity_table = Table({'Name':[],'Sector':[],'lc Source':[],'Period':[],'Orbital separation':[], 'Radius Ratio':[], 'Recovered period':[], 'Periodogram log likelihood at period':[],'2nd highest Period':[],'3rd highest period':[]},names=['Name','Sector','lc Source','Period','Orbital separation', 'Radius Ratio', 'Recovered period', 'Periodogram log likelihood at period','2nd highest Period','3rd highest period'])
#sensitivity_table['Name'] = sensitivity_table['Name'].astype(str)
#sensitivity_table['lc Source'] = sensitivity_table['lc Source'].astype(str)

#target_ID = "HIP 22295"

#with open('Target Lists/Sector_{}_targets_from_TIC_list.pkl'.format(sector), 'rb') as f:
#    sector_targets = pickle.load(f)

rp = 0.03

t_cut_f, BLS_flux_f, phase_f, epoch_f, period_f = ffi_lowess_detrend(save_path = save_path, sector = sector, target_ID_list = single_target_ID, pipeline = pipeline, multi_sector = multi_sector, use_peak_cut = use_peak_cut, binned = binned, transit_mask = transit_mask, injected_planet = injected_planet, injected_rp = rp, injected_per = 8.0, detrending = detrending, n_bins = n_bins)
#ascii.write(variability_table, save_path + 'Variability_info_eleanor.csv', format='csv', overwrite = True)
#ascii.write(sensitivity_table, save_path + 'Sensitivity_analysis_changing_period.csv', format='csv', overwrite = True) 
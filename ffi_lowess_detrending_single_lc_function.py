#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:07:57 2021

ffi_lowess_detrend_single_lc.py

Simple implementation of ffi_lowess_detrend for a single user-defined lc

@author: mbattley
"""
import batman
import lightkurve
import pickle
import random
import scipy.fftpack
import scipy
import csv
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import statsmodels.api as sm
from numpy import loadtxt
from astropy.io import fits
from astropy.timeseries import LombScargle
from lc_download_methods import diff_image_lc_download, two_min_lc_download, eleanor_lc_download, raw_FFI_lc_download, get_lc_from_fits
from scipy.signal import find_peaks
from astropy.timeseries import BoxLeastSquares
from scipy import interpolate
from remove_tess_systematics import clean_tess_lc
from utility_belt import find_freqs, planet_size_from_depth, tic_stellar_info, binned 


def phase_fold_plot(t, lc, period, epoch, target_ID, save_path, title, binned = False):
    """
    Phase-folds the lc by the given period, and plots a phase-folded light-curve
    for the object of interest
    """
    phase = np.mod(t-epoch-period/2,period)/period 
#    phase = np.mod(t-epoch-period/4,period)/period 
    
    phase_fold_fig  = plt.figure()
    plt.scatter(phase, lc, c='k', s=2)
    plt.title(title)
    plt.xlabel('Phase')
    plt.ylabel('Normalized Flux')
    if binned == True:
        sorted_phase = np.argsort(phase)
        phase = phase[sorted_phase]
        lc = lc[sorted_phase]
        binned_phase, binned_lc = bin(phase, lc, binsize=15, method='mean')
        plt.scatter(binned_phase, binned_lc, c='r', s=4)
#    plt.savefig(save_path + '{} - Phase folded by {} days.png'.format(target_ID, period))
    plt.show()

################################ SETUP STUFF ##################################
###### Example Inputs: #####
# tic = 410214986
# #target_ID = 'TIC {}'.format(tic)
# target_ID = '410214986'
# pipeline = 'DIA'      # n.b. can be '2min', 'SPOC-FFI', 'QLP', 'DIA', or 'tessFFIextract' 
# detrending = 'lowess_partial'   # Typically 'lowess_full' or 'lowess-partial'
# save_path = '/Users/matthewbattley/Documents/NGTS/TIC-125641541/'
# multi_sector = True
# find_freq = True
# transit_mask = True
# transit_mask2 = False
# use_peak_cut = True
# n_bins = 20
# sector = 1
# planet_per = 8.138
# planet_epoch = 1492.17
# planet_dur = 0.2

def lowess_detrend(target_ID='410214986',pipeline='DIA',detrending='lowess_full',save_path='',multi_sector=False,find_freq=True,transit_mask = False,transit_mask2 = False,use_peak_cut=False,clean_lc=False,n_bins = 20,sector = 1,planet_per=8.138,planet_epoch=1332.30997,planet_dur=0.1):
    """
    Performs lowess detrending for a given single target_ID
    Inputs:
        - target_ID:     String. Name of object, ideally a TIC number e.g. 'TIC 410214986'
        - pipeline:      String. Pipeline that lc is from. Can be '2min', 'SPOC-FFI', 'QLP', 'DIA', or 'tessFFIextract' 
        - detrending:    String. Detrending method of choice. Here 'lowess_full' or 'lowess-partial' are possible
        - save_path:     String. Your local load/save folder. If not supplied will just work from your current directory
        - multi_sector:  True/False. False for a single TESS sector, and True when using multipl
        - find_freq:     True/False. Enables the 'find_freqs' function, which removes the typical 13.5day TESS periodicity and returns the first three strongest periods in the data using a Lomb-Scargle search
        - transit_mask:  True/False. Allows for transit masking while detrending takes place. If True, also requires 'planet_per', 'planet_epoch' and 'planet_dur' inputs
        - transit_mask2: True/False. Allows for a second planet to be masked during detrending in multiplanet systems. Planet parameters here need to be hard-coded as this is reasonably rare
        - use_peak_cut:  True/False. If True, automatically cuts the peaks and troughs in sharply evolving light-curves. May require tuning for prominence/expected length
        - clean_lc:      True_False. If True, automatically cleans known bad times of photometry according to engineering quaternion data
        - n_bins:        Number. Number of bins for lowess detrending. Default of 30 for a 15hr bin. 20 advised if rotation is sharp with a period <5 days or so
        - sector:        Number. TESS Sector number(s)
        - planet_per:    Number. Planet period for masked transit
        - planet_epoch:  Number. Planet epoch for masked transit
        - planet_dur:    Number. Duration of planet transit for masked transit
    Outputs:
        - lc: Original light-curve loaded from file
        - t_cut: Time after peak_cutting step. If use_peak_cut is FALSE, this is identical to lc.time
        - BLS_flux: Detrended flux after lowess detrending, which is used for BLS search 
        - time_from_lowess_detrend: Time after lowess detrending 
        - phase: Phase according to main BLS period
        - p_rot: Main rotation period in light-curve 
        - period: Strongest period from BLS search
        - results: Full results from BLS search
        Assorted plots are also made during the running of this code, but are hopefully self-explanatory
    """
########################### OPEN AND CLEAN DATA ###############################
    tic = target_ID[4:]    
    if pipeline == '2min':
        sap_lc, pdcsap_lc = two_min_lc_download(target_ID, sector = sector, from_file = False)
        lc = pdcsap_lc
        nancut = np.isnan(lc.flux) | np.isnan(lc.time)
        lc = lc[~nancut]
    if pipeline == 'SPOC-FFI':
        target_ID_pad = str(tic).rjust(16,'0')
        sector_pad = str(sector).rjust(4,'0')
        filename = save_path+'hlsp_tess-spoc_tess_phot_{}-s{}_tess_v1_lc.fits'.format(target_ID_pad,sector_pad)
        lc = get_lc_from_fits(filename, source='SPOC-FFI', clean=False)
        if multi_sector == True:
            filename_list = ['hlsp_tess-spoc_tess_phot_0000000125641541-s0033_tess_v1_lc.fits']
            for file in filename_list:
                lc_new = get_lc_from_fits(save_path + file, source='SPOC-FFI', clean=False)
                lc = lc.append(lc_new)
    elif pipeline == 'QLP':
        target_ID_pad = str(tic).rjust(16,'0')
        sector_pad = str(sector).rjust(2,'0')
        filename = save_path + 'hlsp_qlp_tess_ffi_s00{}-{}_tess_v01_llc.fits'.format(sector_pad,target_ID_pad)
        lc = get_lc_from_fits(filename, source='QLP', clean=True)
    elif pipeline == 'DIA':
        #lc, filename = diff_image_lc_download(target_ID, sector, plot_lc = True, save_path = save_path, from_file = True)
        filename = '410214986_sector01_3_2.lc'
        lines = loadtxt(filename,delimiter=' ')
        DIA_lc = list(map(list, zip(*lines)))
        DIA_mag = np.array(DIA_lc[1])
        DIA_flux = 10**(-0.4*(DIA_mag - 20.60654144))
        norm_flux = DIA_flux/np.median(DIA_flux)
        lc = lightkurve.LightCurve(time = DIA_lc[0], flux = norm_flux, flux_err = DIA_lc[2])
    elif pipeline == 'tessFFIextract':
        filename = save_path+'TIC-{}.fits'.format(tic)
        hdul = fits.open(filename)
        data = hdul[1].data
        
        time = data['BJD']
        time = np.array([x - 2457000 for x in time])
        flux = np.array(data['COR_AP2.5_norm'])#/np.median(data['COR_AP2.5'])
        
        mask = np.array([True]*len(time))
        
        for i, item in enumerate(flux):
            if item < 0.95:
                mask[i] = False
                
        cleaned_flux = flux[mask]
        cleaned_time = time[mask]
        
        plot_lc = True
        if plot_lc == True:
            original_lc_fig = plt.figure()
            plt.scatter(cleaned_time, cleaned_flux,s=1,c='k')
            plt.title('Original lc')
            plt.xlabel('Time [BJD -2457000]')
            plt.ylabel('Flux')
        #            original_lc_fig.savefig(save_path + "TIC {} - original CDIPS lc.pdf".format(tic))
            plt.show()
        #            plt.close(original_lc_fig)
        
        lc = lightkurve.LightCurve(time = cleaned_time, flux = cleaned_flux, flux_err = cleaned_flux)
        hdul.close()

    else:
        print('Please enter valid pipeline')
    
    
    nancut = np.isnan(np.array(lc.flux)) | np.isnan(np.array(lc.time.value))
    lc = lc[~nancut]
    
    normalized_flux = lc.flux#np.array(lc.flux)/np.median(lc.flux)
    
    normalized_lc_fig = plt.figure()
    plt.scatter(lc.time.value, normalized_flux,s=1,c='k')
    plt.title('Normalised lc')
    plt.xlabel('Time [BJD -2457000]')
    plt.ylabel('Normaized Flux')
    plt.show()
    
    ################## 
    if clean_lc == True:
        clean_time, clean_flux, clean_flux_err = clean_tess_lc(lc.time.value, lc.flux, lc.flux_err, target_ID, sector, save_path)
        lc = lightkurve.LightCurve(time = clean_time, flux = clean_flux, flux_err = clean_flux_err)
        print('Cleaned it fine')
    
    ######################### Find rotation period(s) ################################
    normalized_flux = lc.flux
    
    # From Lomb-Scargle
    if find_freq == True:
        freq_list = find_freqs(lc.time.value,lc.flux,plot_ls_fig = False,target_ID=target_ID)
        p_rot = 1/freq_list[0]
    #    freq_table[num_done-1] = freq_list
    else:
        freq = np.arange(0.04,4.1,0.00001)
        power = LombScargle(lc.time.value, normalized_flux).power(freq)
        ls_fig = plt.figure()
        plt.plot(freq, power, c='k', linewidth = 1)
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.title('{} LombScargle Periodogram for original lc'.format(target_ID))
        ls_fig.show()
    #        ls_fig.savefig(save_path + '{} - Lomb-Sacrgle Periodogram for original lc.png'.format(target_ID))
        plt.close(ls_fig)
        i = np.argmax(power)
        freq_rot = freq[i]
        p_rot = 1/freq_rot
    print('Rotation Period = {:.3f}d'.format(p_rot))
    
    #if p_rot <8:
    #    n_bins = 20
    
    ################################ Remove Peaks #################################
    
    combined_flux = lc.flux
    if use_peak_cut == True:
        try:
            if pipeline == '2min':
                peaks, peak_info = find_peaks(combined_flux, prominence = 0.001, width = 200)
            else:
                peaks, peak_info = find_peaks(combined_flux, prominence = 0.001, width = 20)
            print(peaks)
            if len(peaks) == 0:
                print('No peaks found')
            if pipeline == '2min':
                troughs, trough_info = find_peaks(-combined_flux, prominence = -0.001, width = 200)
            else:
                troughs, trough_info = find_peaks(-combined_flux, prominence = -0.001, width = 20)
            print(troughs)
            if len(troughs) == 0:
                print('No troughs found')
            flux_peaks = combined_flux[peaks]
            flux_troughs = combined_flux[troughs]
            amplitude_peaks = ((flux_peaks[0]-1) + (1-flux_troughs[0]))/2
            print("Absolute amplitude of main variability = {}".format(amplitude_peaks))
            peak_location_fig = plt.figure()
            plt.scatter(lc.time, combined_flux, s = 2, c = 'k')
            plt.plot(lc.time[peaks], combined_flux[peaks], "x")
            plt.plot(lc.time[troughs], combined_flux[troughs], "x", c = 'r')
            peak_location_fig.savefig(save_path + "{} - Peak location fig.png".format(target_ID))
            peak_location_fig.show()
    #                plt.close(peak_location_fig)
            
            near_peak_or_trough = [False]*len(combined_flux)
            
            for i in peaks:
                for j in range(len(lc.time)):
                    if abs(lc.time[j] - lc.time[i]) < 0.1:
                        near_peak_or_trough[j] = True
            
            for i in troughs:
                for j in range(len(lc.time)):
                    if abs(lc.time[j] - lc.time[i]) < 0.1:
                        near_peak_or_trough[j] = True
            
            near_peak_or_trough = np.array(near_peak_or_trough)
            
            t_cut = lc.time[~near_peak_or_trough]
            flux_cut = combined_flux[~near_peak_or_trough]
            flux_err_cut = lc.flux_err[~near_peak_or_trough]
        #    
        #    
            # Plot new cut version
            peak_cut_fig = plt.figure()
            plt.scatter(t_cut,flux_cut, c = 'k', s = 2)
            plt.xlabel('Time - 2457000 [BTJD days]')
            plt.ylabel("Flux")
            plt.title('{} lc after removing peaks/troughs'.format(target_ID))
            ax = plt.gca()
            peak_cut_fig.savefig(save_path + "{} - Peak cut fig.png".format(target_ID))
    #                    peak_cut_fig.show()
            plt.close(peak_cut_fig)
        except:
            t_cut = lc.time
            flux_cut = combined_flux
            flux_err_cut = lc.flux_err
            print('Flux cut failed')  
    else:
         t_cut = lc.time.value
         flux_cut = combined_flux
         flux_err_cut = lc.flux_err
         print('Flux cut skipped')
         
    ############################## Apply transit mask #########################
    
    if transit_mask == True:
        period = planet_per#731.57
        epoch = planet_epoch#1492.17
        duration = planet_dur #0.17
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
        
        lowess = sm.nonparametric.lowess(flux_masked, t_masked, frac=15/len(lc.time.value))
        smooth_flux = lowess[:,1]
        
        plt.figure()
        plt.scatter(t_masked,smooth_flux,s=1,c='k')
    
        if pipeline == '2min':
            f = interpolate.interp1d(t_masked,flux_masked, kind = 'linear')
        else:
            f = interpolate.interp1d(t_masked,smooth_flux, kind = 'linear')
    
        flux_new = f(t_new)
        interpolated_fig = plt.figure()
        plt.scatter(t_cut, flux_cut, s = 8, c = 'k')
        plt.scatter(t_new,flux_new, s=8, c = 'r')
        plt.xlabel('Time - 2457000 [BTJD days]')
        plt.ylabel('Relative flux')
    #    interpolated_fig.savefig(save_path + "{} - Interpolated over transit mask fig.png".format(target_ID))
        
        t_transit_mask = np.concatenate((t_masked,t_new), axis = None)
        flux_transit_mask = np.concatenate((flux_masked,flux_new), axis = None)
        
        sorted_order = np.argsort(t_transit_mask)
        t_transit_mask = t_transit_mask[sorted_order]
        flux_transit_mask = flux_transit_mask[sorted_order]
        
    if transit_mask2 == True:
        period = 2000#731.57 # n.b. left as user-defined
        epoch = 1492.17
        duration = 0.2
        phase = np.mod(t_transit_mask-epoch-period/2,period)/period
        
        near_transit = [False]*len(flux_transit_mask)
        
        for i in range(len(t_cut)):
            if abs(phase[i] - 0.5) < duration/period:
                near_transit[i] = True
        
        near_transit = np.array(near_transit)
        t_masked = t_transit_mask[~near_transit]
        flux_masked = flux_transit_mask[~near_transit]
    #    flux_err_masked = flux_err_cut[~near_transit]
        t_new = t_transit_mask[near_transit]
        
        lowess = sm.nonparametric.lowess(flux_masked, t_masked, frac=20/len(lc.time.value))
        smooth_flux = lowess[:,1]
        
        plt.figure()
        plt.scatter(t_masked,smooth_flux,s=1,c='k')
    
        if pipeline == '2min':
            f = interpolate.interp1d(t_masked,flux_masked, kind = 'linear')
        else:
            f = interpolate.interp1d(t_masked,smooth_flux, kind = 'linear')
    #                f = interpolate.BarycentricInterpolator(t_masked,flux_masked)
    #    f = scipy.optimize.curvefit(func,)
    
        flux_new = f(t_new)
        interpolated_fig = plt.figure()
    #                plt.scatter(t_masked, flux_masked, s = 2, c = 'k')
        plt.scatter(t_transit_mask, flux_transit_mask, s = 8, c = 'k')
        plt.scatter(t_new,flux_new, s=8, c = 'r')
        plt.xlabel('Time - 2457000 [BTJD days]')
        plt.ylabel('Relative flux')
    #    interpolated_fig.savefig(save_path + "{} - Interpolated over transit mask fig.png".format(target_ID))
        
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
            lowess = sm.nonparametric.lowess(flux_transit_mask, t_transit_mask, frac=0.015)
        else:
            lowess = sm.nonparametric.lowess(flux_cut, t_cut, frac=n_bins/len(lc.time))
    #       number of points = 20 at lowest for TESS 30min, or otherwise frac = 20/len(t_section) 
    #       For TESS 2min: 300-450points
    #       For NGTS:   4200 points (Cadence ~13s)
        print(lowess)
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
        time_from_lowess_detrend = t_cut
        
        lowess_full_residuals_fig = plt.figure()
        plt.scatter(t_cut,residual_flux_lowess, c = 'k', s = 2)
        plt.title('{} lc after lowess full lc detrending'.format(target_ID))
        plt.xlabel('Time - 2457000 [BTJD days]')
        plt.ylabel('Relative flux')
        ax = plt.gca()
    #            lowess_full_residuals_fig.savefig(save_path + "{} lc after LOWESS full lc detrending.png".format(target_ID))
        plt.show()
    #                plt.close(lowess_full_residuals_fig)
        
        
    # Partial lc
    elif detrending == 'lowess_partial':
        expand_final = False
        time_diff = np.diff(t_cut)
        residual_flux_lowess = np.array([])
        time_from_lowess_detrend = np.array([])
        full_lowess_flux = np.array([])
        
        overplotted_detrending_fig = plt.figure()
        plt.scatter(t_cut,flux_cut, c = 'k', s = 2)
        plt.xlabel('Time - 2457000 [BTJD days]')
        plt.ylabel("Flux")
        plt.title('{} lc with overplotted detrending'.format(target_ID))
        
        low_bound = 0
        if pipeline == '2min':
            n_bins = 15*n_bins
        else:
            n_bins = n_bins
        for i in range(len(t_cut)-1):
            if t_cut[i] > 2200:
                n_bins = 3*20
            if time_diff[i] > 0.1:
                high_bound = i+1
                
                t_section = t_cut[low_bound:high_bound]
                flux_section = flux_cut[low_bound:high_bound]
    #                        print(t_section)
                if len(t_section)>=n_bins:
                    if transit_mask == True:
                        lowess = sm.nonparametric.lowess(flux_transit_mask[low_bound:high_bound], t_transit_mask[low_bound:high_bound], frac=n_bins/len(t_section))
                    else:
                        if abs(t_section[10] - t_section[9]) < 0.01:
                            n_bins = 50
                        lowess = sm.nonparametric.lowess(flux_section, t_section, frac=n_bins/len(t_section))
    #                    lowess = sm.nonparametric.lowess(flux_section, t_section, frac=20/len(t_section))
                    lowess_flux_section = lowess[:,1]
                    plt.plot(t_section, lowess_flux_section, '-')
                    
                    residuals_section = flux_section/lowess_flux_section
                    residual_flux_lowess = np.concatenate((residual_flux_lowess,residuals_section))
                    time_from_lowess_detrend = np.concatenate((time_from_lowess_detrend,t_section))
                    full_lowess_flux = np.concatenate((full_lowess_flux,lowess_flux_section))
                    low_bound = high_bound
                else:
                    print('Skipped one gap at {}'.format(high_bound))
        
        # Carries out same process for final line (up to end of data)        
        high_bound = len(t_cut)
        
        if high_bound - low_bound < n_bins:
            old_low_bound = low_bound
            low_bound = high_bound - n_bins
            expand_final = True
        t_section = t_cut[low_bound:high_bound]
        flux_section = flux_cut[low_bound:high_bound]
        if transit_mask == True:
            lowess = sm.nonparametric.lowess(flux_transit_mask[low_bound:high_bound], t_transit_mask[low_bound:high_bound], frac=n_bins/len(t_section))
        else:
            if abs(t_section[10] - t_section[9]) < 0.01:
                n_bins = 90
            lowess = sm.nonparametric.lowess(flux_section, t_section, frac=n_bins/len(t_section))
    #            lowess = sm.nonparametric.lowess(flux_section, t_section, frac=20/len(t_section))
        lowess_flux_section = lowess[:,1]
        plt.plot(t_section, lowess_flux_section, '-')
    #    plt.title('AU Mic - Overplotted LOWESS detrending')
    #    if injected_planet != False:
    #        overplotted_detrending_fig.savefig(save_path + "{} - Overplotted lowess detrending - partial lc - {}R {}d injected planet.png".format(target_ID, params.rp, params.per))
    #    else:
    #        overplotted_detrending_fig.savefig(save_path + "{} - Overplotted lowess detrending - partial lc".format(target_ID))
        overplotted_detrending_fig.show()
    #    plt.close(overplotted_detrending_fig)
        
        residuals_section = flux_section/lowess_flux_section
        if expand_final == True:
            shorten_bound = n_bins - (high_bound-old_low_bound)
            residual_flux_lowess = np.concatenate((residual_flux_lowess,residuals_section[shorten_bound:]))
            time_from_lowess_detrend = np.concatenate((time_from_lowess_detrend,t_section[shorten_bound:]))
            full_lowess_flux = np.concatenate((full_lowess_flux,lowess_flux_section[shorten_bound:]))
        else:
            residual_flux_lowess = np.concatenate((residual_flux_lowess,residuals_section))
            time_from_lowess_detrend = np.concatenate((time_from_lowess_detrend,t_section))
            full_lowess_flux = np.concatenate((full_lowess_flux,lowess_flux_section))
        
    #    t_section = t_cut[83:133]
        residuals_after_lowess_fig = plt.figure()
        plt.scatter(time_from_lowess_detrend,residual_flux_lowess, c = 'k', s = 2)
        plt.title('{} lc after LOWESS partial lc detrending'.format(target_ID))
        plt.xlabel('Time - 2457000 [BTJD days]')
        plt.ylabel('Flux')
        #ax = plt.gca()
    #                if injected_planet != False:
    #                    residuals_after_lowess_fig.savefig(save_path + "{} lc after LOWESS partial lc detrending - {}R {}d injected planet.png".format(target_ID, params.rp, params.per))
    #                else:
    #                    residuals_after_lowess_fig.savefig(save_path + "{} lc after LOWESS partial lc detrending".format(target_ID))
        residuals_after_lowess_fig.show()
    #                plt.close(residuals_after_lowess_fig)
        print('Detrended light-curve')
    
    
    #    ########################## Periodogram Stuff ##################################
    
    # Create periodogram
    durations = np.linspace(0.01, 1., 100) * u.day
    if detrending == 'lowess_full' or detrending == 'lowess_partial':
        BLS_flux = residual_flux_lowess
    else:
        BLS_flux = combined_flux
    # with open('Detrended_time.pkl', 'wb') as f:
    #     pickle.dump(t_cut, f, pickle.HIGHEST_PROTOCOL)
    # with open('Detrended_flux.pkl', 'wb') as f:
    #     pickle.dump(BLS_flux, f, pickle.HIGHEST_PROTOCOL)
    time_mask = [True]*len(t_cut)
    for i in range(len(t_cut)):
        if t_cut[i] > 1800:
            time_mask[i] = False
    short_time = t_cut[time_mask]
    short_flux = BLS_flux[time_mask]
    model = BoxLeastSquares(short_time*u.day, short_flux)
    #model = BoxLeastSquares(t_cut*u.day, BLS_flux)
    #model = BLS(lc_30min.time*u.day,BLS_flux)
    if t_cut[-1] - t_cut[0] > 100:
        freq_factor = 10.0
    else:
        freq_factor = 5.0
    results = model.autopower(durations, minimum_n_transit=3,frequency_factor=freq_factor)
    
    # Find the period and epoch of the peak
    index = np.argmax(results.power)
    period = results.period[index]
    t0 = results.transit_time[index]
    duration = results.duration[index]
    transit_info = model.compute_stats(period, duration, t0)
    print(transit_info)
    
    epoch = transit_info['transit_times'][0]
    
    periodogram_fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    #Highlight the harmonics of the peak period
    ax.axvline(period.value, alpha=0.4, lw=3)
    for n in range(2, 10):
        ax.axvline(n*period.value, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(period.value / n, alpha=0.4, lw=1, linestyle="dashed")
    
    # Plot and save the periodogram
    ax.plot(results.period, results.power, "k", lw=0.5)
    ax.set_xlim(results.period.min().value, results.period.max().value)
    ax.set_xlabel("period [days]")
    ax.set_ylabel("log likelihood")
    #if use_TESSflatten == True:
    #    ax.set_title('{} - BLS Periodogram with TESSflatten'.format(target_ID))
    ##            periodogram_fig.savefig(save_path + '{} - BLS Periodogram with TESSflatten.png'.format(target_ID))
    #else:
    #    periodogram_fig.savefig(save_path + '{} - BLS Periodogram after lowess partial detrending.png'.format(target_ID))
    ##            plt.close(periodogram_fig)
    periodogram_fig.show()
    
    periodogram_data = [results.period.value,results.power]
    #
    #with open(save_path+"{}_periodogram_data.csv".format(target_ID),"w+") as my_csv:
    #    csvWriter = csv.writer(my_csv,delimiter=',')
    #    csvWriter.writerows(periodogram_data)
    #
    ###    ################################## Phase folding ##########################
    # Find indices of 2nd and 3rd peaks of periodogram
    all_peaks = scipy.signal.find_peaks(results.power, width = 5, distance = 10)[0]
    all_peak_powers = results.power[all_peaks]
    sorted_power_indices = np.argsort(all_peak_powers)
    sorted_peak_powers = all_peak_powers[sorted_power_indices]
    sorted_peak_periods = results.period[all_peaks][sorted_power_indices]
    
    # MASKS ALL HARMONICS OF MAXIMUM POWER PERIOD
    # n.b. Uses a similar technique to masking bad times is tess_systematics cleaner:
    # E.g. mask all those close to 1.5, 2, 2.5, 3, 3.5, 4, ..., 10 etc times the max period
    # and then search through rest as normal
    harmonic_mask = [False]*len(sorted_peak_powers)
    harmonics = period.value*np.array([0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,3.5,4,4.5,5,5.5,6,
                                       6.5,7,7.5,8,8.5,9,9.5,10])
    simplified_harmonics = period.value*np.array([0.333,0.5,0.75,1,1,1.5,2,3,4,5,6,7,8,9,10])
    
    for i in simplified_harmonics:
        for j in range(len(sorted_peak_periods)):
    #                    print(sorted_peak_periods[j].value - i)
            if abs(sorted_peak_periods[j].value - i) < 0.1:
                harmonic_mask[j] = True
    #            print('Completed for loop')
    harmonic_mask = np.array(harmonic_mask)
    sorted_peak_powers = sorted_peak_powers[~harmonic_mask]
    sorted_peak_periods = sorted_peak_periods[~harmonic_mask]
    
    # Find info for 2nd largest peak in periodogram
    index_peak_2 = np.where(results.power==sorted_peak_powers[-1])[0]
    period_2 = results.period[index_peak_2[0]]
    t0_2 = results.transit_time[index_peak_2[0]]
    
    # Find info for 3rd largest peak in periodogram
    index_peak_3 = np.where(results.power==sorted_peak_powers[-2])[0]
    period_3 = results.period[index_peak_3[0]]
    t0_3 = results.transit_time[index_peak_3[0]]
    
    #phase_fold_plot(lc_30min.time, BLS_flux, rot_period.value, rot_t0.value, target_ID, save_path, '{} folded by rotation period'.format(target_ID))
    print('Max BLS Period = {} days, t0 = {}'.format(period.value, t0.value))        
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
    #phase_fold_plot(t_cut, BLS_flux, period_to_test4, t0_to_test4, target_ID, save_path, '{} detrended lc folded by {:0.4} days'.format(target_ID,period_to_test4))
    #print("Absolute amplitude of main variability = {}".format(amplitude_peaks))
    #print('Main Variability Period from Lomb-Scargle = {:.3f}d'.format(p_rot))
    #print("Main Variability Period from BLS of original = {}".format(rot_period))
    
    ############################# Eyeballing ##############################
    """
    Generate 2 x 2 eyeballing plot
    """
    eye_balling_fig, axs = plt.subplots(2,2, figsize = (16,10),  dpi = 120)
    
    # Original DIA with injected transits setup
    axs[0,0].scatter(lc.time.value, combined_flux, s=1, c= 'k')
    axs[0,0].set_ylabel('Normalized Flux')
    axs[0,0].set_xlabel('Time- 2457000 [BTJD days]')
    axs[0,0].plot(time_from_lowess_detrend,full_lowess_flux)
    #axs[0,0].set_title('{} - {} light curve'.format(target_ID, 'DIA'))
    #for n in range(int(-1*8/params.per),int(2*8/params.per+2)):
    #    axs[0,0].axvline(params.t0+n*params.per+mid_point, ymin = 0.1, ymax = 0.2, lw=1, c = 'r')
    
    # Detrended figure setup
    axs[0,1].scatter(t_cut, BLS_flux, c = 'k', s = 1, label = '{} residuals after {} detrending'.format(target_ID,detrending))
    #axs[0,1].set_title('{} residuals after {} detrending'.format(target_ID, detrending))
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
    #axs[1,0].set_title('{} - BLS Periodogram of residuals'.format(target_ID))
    axs[1,0].axvline(period.value, alpha=0.4, lw=3)
    for n in range(2, 10):
        axs[1,0].axvline(n*period.value, alpha=0.4, lw=1, linestyle="dashed")
        axs[1,0].axvline(period.value/ n, alpha=0.4, lw=1, linestyle="dashed")
    
    # Folded or zoomed plot setup
    epoch = t0.value
    period = period.value
    test_epoch = 1500.09#1492.18
    test_period = 70.447#73.158
    print('Main epoch is {}'.format(t0.value+lc.time.value[0]))
    phase = np.mod(t_cut-epoch-period/2,period)/period 
    if multi_sector == True:
        axs[1,1].scatter(phase[:741], BLS_flux[:741], c='b', s=1,label='Sector 3')
        axs[1,1].scatter(phase[741:], BLS_flux[741:], c='r', s=1,label='Sector 30')
    else:
        axs[1,1].scatter(phase,BLS_flux,c='k',s=1)
    #axs[1,1].set_title('{} Lightcurve folded by {:0.4} days'.format(target_ID, period))
    axs[1,1].set_xlabel('Phase')
    axs[1,1].set_ylabel('Normalized Flux')
    axs[1,1].legend()
    #axs[1,1].set_xlim(0.4,0.6)
    #binned_phase, binned_lc = bin(phase, BLS_flux, binsize=15, method='mean')
    #plt.scatter(binned_phase, binned_lc, c='r', s=4)
    
    eye_balling_fig.tight_layout()
    #eye_balling_fig.savefig(save_path + '{} - Full eyeballing fig S{}.pdf'.format(target_ID,sector))
    #plt.close(eye_balling_fig)
    plt.show()
    
    
    #2x2 eyeballing plot with shared time axes:
    eyeballing_fig2 = plt.figure(figsize = (16,10),dpi = 120)
    
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222, sharex=ax1)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    
    # Original DIA with injected transits setup
    ax1.scatter(lc.time.value, combined_flux, s=1, c= 'k')
    ax1.set_ylabel('Normalized Flux')
    ax1.set_xlabel('Time- 2457000 [BTJD days]')
    ax1.plot(time_from_lowess_detrend,full_lowess_flux)
    
    # Detrended figure setup
    ax2.scatter(t_cut, BLS_flux, c = 'k', s = 1, label = '{} residuals after {} detrending'.format(target_ID,detrending))
    ax2.set_ylabel('Normalized Flux')
    ax2.set_xlabel('Time - 2457000 [BTJD days]')
    
    # Periodogram setup
    ax3.plot(results.period, results.power, "k", lw=0.5)
    #ax3.set_xlim(results.period.min().value, 25)
    ax3.set_xlim(results.period.min().value, results.period.max().value)
    ax3.set_xlabel("period [days]")
    ax3.set_ylabel("log likelihood")
    ax3.axvline(period, alpha=0.4, lw=3)
    for n in range(2, 10):
        ax3.axvline(n*period, alpha=0.4, lw=1, linestyle="dashed")
        ax3.axvline(period/n, alpha=0.4, lw=1, linestyle="dashed")
    
    # Folded or zoomed plot setup
    if multi_sector == True:
        ax4.scatter(phase[:741], BLS_flux[:741], c='b', s=1,label='Sector 3')
        ax4.scatter(phase[741:], BLS_flux[741:], c='r', s=1,label='Sector 30')
    else:
        ax4.scatter(phase,BLS_flux,c='k',s=1)
    #axs[1,1].set_title('{} Lightcurve folded by {:0.4} days'.format(target_ID, period))
    ax4.set_xlabel('Phase')
    ax4.set_ylabel('Normalized Flux')
    ax4.legend()
    
    eyeballing_fig2.tight_layout()
    #eye_balling_fig.savefig(save_path + '{} - Full eyeballing fig S{}.pdf'.format(target_ID,sector))
    #plt.close(eye_balling_fig)
    plt.show()
    
    return lc,t_cut, BLS_flux, time_from_lowess_detrend, phase, p_rot, period, results

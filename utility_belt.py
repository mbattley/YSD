#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:32:12 2019
utility_belt.py

This script collects a series of functions that are useful oevr many different
codes

@author: mbattley
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astroquery.mast import Catalogs
from scipy import optimize
from astropy.timeseries import LombScargle

def convertRAback(ra):
    ra = float(ra)
    hr = ra / 360. * 24. // 1
    mins = (ra / 360. * 24. - hr) * 60 // 1
    sec = ((ra / 360. * 24. - hr) * 60 - mins) * 60
    return hr, mins, sec

def convertDecback(dec):
    dec = float(dec)
    deg = np.sign(dec) * (np.abs(dec) // 1)
    mins = (np.abs(dec) - np.abs(deg)) * 60 // 1
    sec = ((np.abs(dec) - np.abs(deg)) * 60 - mins) * 60
    return deg, mins, sec

def convert_RA_Dec_array_to_hms(RA_array,Dec_array):
    RA_hms = np.zeros(len(RA_array))
    Dec_dms = np.zeros(len(RA_array))
    for i in np.arange(len(RA_array)):
        RA_h, RA_m, RA_s = convertRAback(RA_array)
        Dec_d, Dec_m, Dec_s = convertDecback(Dec_array)
        if RA_s < 10:
            RA_hms[i] = '{:0>2d}:{:0>2d}:0{:.2f}'.format(int(RA_h),int(RA_m),RA_s)
        else:
            RA_hms[i] = '{:0>2d}:{:0>2d}:{:.2f}'.format(int(RA_h),int(RA_m),RA_s)
        if Dec_s<10:
            Dec_dms[i] = '{:0>2d}:{:0>2d}:0{:.2f}'.format(int(Dec_d),int(Dec_m),Dec_s)
        else:
            Dec_dms[i] = '{:0>2d}:{:0>2d}:{:.2f}'.format(int(Dec_d),int(Dec_m),Dec_s)
    return RA_hms, Dec_dms


def trig_func(t,f,a,b,c):
    return a*np.sin(2*np.pi*f*t) + b*np.cos(2*np.pi*f*t) + c

def next_highest_freq(time, flux, freq, f_remove, plot_ls_fig = False,target_ID = ''):
    popt_maxV, pcov_maxV = optimize.curve_fit(lambda t, a, b, c: trig_func(t,f_remove, a, b, c), time, flux, maxfev=1000)
    max_var = trig_func(time,f_remove,*popt_maxV)
    flux = flux/max_var
    power = LombScargle(time, flux).power(freq)
    if plot_ls_fig == True:
        ls_fig = plt.figure()
        plt.plot(freq, power, c='k', linewidth = 1)
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.title('{} LombScargle Periodogram'.format(target_ID))
        ls_fig.show()
    i = np.argmax(power)
    freq_2 = freq[i]
    return freq_2, flux

def find_freqs(time, flux, plot_ls_fig = True, target_ID = ''):
    
    # Remove frequencies associated with 14d data gap
    print('Made it in to find_freqs')
    f=1/14
    popt_sys, pcov_sys = optimize.curve_fit(lambda t, a, b, c: trig_func(t,f, a, b, c), time, flux, maxfev=1000)
    print('Got fitting parameters')
    sys_var = trig_func(time,f,*popt_sys)
    flux = flux/sys_var
    print('Got rid of 14d period')
    
     #From Lomb-Scargle
    freq = np.arange(0.05,4.1,0.00001)
    power = LombScargle(time, flux).power(freq)
    if plot_ls_fig == True:
        ls_fig = plt.figure()
        plt.plot(freq, power, c='k', linewidth = 1)
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.title('{} LombScargle Periodogram for original lc'.format(target_ID))
        ls_fig.show()
#        ls_fig.savefig(save_path + '{} - Lomb-Scargle Periodogram for original lc.png'.format(target_ID))
#        plt.close(ls_fig)
    i = np.argmax(power)
    freq_rot = freq[i]
    print('Found highest freq')
    
    # Remove highest frequency to get 2nd highest
    f_remove=freq_rot
    freq_2, flux = next_highest_freq(time, flux, freq, f_remove, plot_ls_fig = False)
        
    # Remove 2nd highest frequency to get 3rd highest
    f_remove=freq_2
    freq_3, flux = next_highest_freq(time, flux, freq, f_remove, plot_ls_fig = False)

    freq_list = [freq_rot,freq_2,freq_3]
    print('Compiled freq list')
    
    #final_fig = plt.figure()
    #plt.scatter(time,flux,s=1,c='k')
    #plt.show()
    
    return freq_list

def tic_stellar_info(target_ID, from_file = False, filename = 'BANYAN_XI-III_members_with_TIC.csv'):
    if from_file == True:
        table_data = Table.read(filename , format='ascii.csv')
        
        # Obtains ra and dec for object from target_ID
        i = list(table_data['main_id']).index(target_ID)
        #camera = table_data['S{}'.format(sector)][i]
        tic = table_data['MatchID'][i]
        r_star = table_data['Stellar Radius'][i]
        T_eff = table_data['T_eff'][i]
        
    else:
        TIC_table = Catalogs.query_object(target_ID, catalog = "TIC")
        tic = TIC_table['ID'][0]
#        print(TIC_table[0])
        r_star = TIC_table['rad'][0]
        T_eff = TIC_table['Teff'][0]
    
    return tic, r_star, T_eff

def planet_size_from_depth(target_ID, depth):
    tic, r_star, T_eff = tic_stellar_info(target_ID)
    
    r_Sun = 695510 #km
    r_Jup = 69911  #km
    r_Nep = 24622  #km
    r_Earth = 6371 #km
    
    r_p_solar = r_star*np.sqrt(depth) # Radius in Solar radii
    r_p_km = r_p_solar*r_Sun          # Radius in km
    r_p_Jup = r_p_km/r_Jup            # Radius in Jupiter radii
    r_p_Nep = r_p_km/r_Nep            # Radius in Neptune radii
    r_p_Earth = r_p_km/r_Earth        # Radius in Earth radii
    
    print('Planet size:')
    print('{} R_solar'.format(r_p_solar))
    print('{}km'.format(r_p_km))
    print('{} R_Jup'.format(r_p_Jup))
    print('{} R_Nep'.format(r_p_Nep))
    print('{} R_Earth'.format(r_p_Earth))
    
    return r_p_Jup

def fraction_of_breakup_rotation(r_star,m_star,per):
    """
    Determines omega ratio, the ratio of rotation rate to critical rotation rate
    """
    
    # n.b. Period in hours, r and m in stellar units
    
    omega_ratio = 2*np.pi*np.sqrt(r_star**3/m_star)/per
    
    return omega_ratio

def bin(time, flux, binsize=15, method='mean'):
    """Bins a lightcurve in blocks of size `binsize`.
    n.b. based on the one from eleanor

    The value of the bins will contain the mean (`method='mean'`) or the
    median (`method='median'`) of the original data.  The default is mean.

    Parameters
    ----------
    binsize : int
        Number of cadences to include in every bin.
    method: str, one of 'mean' or 'median'
        The summary statistic to return for each bin. Default: 'mean'.

    Returns
    -------
    binned_lc : LightCurve object
        Binned lightcurve.

    Notes
    -----
    - If the ratio between the lightcurve length and the binsize is not
      a whole number, then the remainder of the data points will be
      ignored.
    """
    available_methods = ['mean', 'median']
    if method not in available_methods:
        raise ValueError("method must be one of: {}".format(available_methods))
    methodf = np.__dict__['nan' + method]

    n_bins = len(flux) // binsize
    indexes = np.array_split(np.arange(len(time)), n_bins)
    binned_time = np.array([methodf(time[a]) for a in indexes])
    binned_flux = np.array([methodf(flux[a]) for a in indexes])

    return binned_time, binned_flux

def phase_fold_plot(t, lc, period, epoch, target_ID, save_path, title, binned = True, n_bins=15):
    """
    Phase-folds the lc by the given period, and plots a phase-folded light-curve
    for the object of interest
    """
    phase = np.mod(t-epoch-period/2,period)/period 
    
    phase_fold_fig  = plt.figure(figsize = (10,4))
    plt.scatter(phase, lc, c='k', s=1)
    if title == True:
        plt.title('{} light-curve folded by {:.4f} days'.format(target_ID,period))
    plt.xlabel('Phase')
    plt.ylabel('Normalized Flux')
    plt.xlim([0,1])
#    plt.ylim([1-1.1*(1-np.min(lc)),1.1*(np.max(lc)-1)+1])
    if binned == True:
        sort_array = np.argsort(phase)
        new_phase = phase[sort_array]
        new_lc = lc[sort_array]
        binned_phase, binned_lc = bin(new_phase, new_lc, binsize=n_bins, method='mean')
        plt.scatter(binned_phase, binned_lc, c='r', s=4)
    else:
        new_phase = phase
        new_lc = lc
    #plt.savefig(save_path + '{} - Phase folded by {} days.png'.format(target_ID, period))
    plt.show()
    #plt.close(phase_fold_fig)
    
    #return new_phase, new_lc

#def phase_fold_plot(t, lc, period, epoch, target_ID, save_path, title, binned = False):
#    """
#    Phase-folds the lc by the given period, and plots a phase-folded light-curve
#    for the object of interest
#    """
#    phase = np.mod(t-epoch-period/2,period)/period 
#    
#    phase_fold_fig  = plt.figure()
#    plt.scatter(phase, lc, c='k', s=2)
#    plt.title(title)
#    plt.xlabel('Phase')
#    plt.ylabel('Normalized Flux')
#    if binned == True:
#        binned_phase, binned_flux = bin(phase, lc, binsize=15, method='mean')
#        plt.scatter(binned_phase, binned_lc, c='r', s=4)
##    plt.savefig(save_path + '{} - Phase folded by {} days.pdf'.format(target_ID, period))
#    plt.show()
##    plt.close(phase_fold_fig)
#    
#    return binned_phase, binned_flux
 
 
def binned(time, flux, binsize=15, method='mean'):
    """Bins a lightcurve in blocks of size `binsize`.
    n.b. based on the one from eleanor
    The value of the bins will contain the mean (`method='mean'`) or the
    median (`method='median'`) of the original data.  The default is mean.
    Parameters
    ----------
    binsize : int
        Number of cadences to include in every bin.
    method: str, one of 'mean' or 'median'
        The summary statistic to return for each bin. Default: 'mean'.
    Returns
    -------
    binned_lc : LightCurve object
        Binned lightcurve.
    Notes
    -----
    - If the ratio between the lightcurve length and the binsize is not
      a whole number, then the remainder of the data points will be
      ignored.
    """
    available_methods = ['mean', 'median']
    if method not in available_methods:
        raise ValueError("method must be one of: {}".format(available_methods))
    methodf = np.__dict__['nan' + method]

    n_bins = len(flux) // binsize
    indexes = np.array_split(np.arange(len(time)), n_bins)
    binned_time = np.array([methodf(time[a]) for a in indexes])
    binned_flux = np.array([methodf(flux[a]) for a in indexes])

    return binned_time, binned_flux

def bin_err(time,err,binsize=15):
    """
    Bins errors in quadrature
    """
    n_bins = len(flux) // binsize
    indexes = np.array_split(np.arange(len(time)), n_bins)
    binned_time = np.array([np.mean(time[a]) for a in indexes])
    binned_flux = np.array([np.sqrt(np.sum(err[a]**2))/len(err[a]) for a in indexes])
    
    return binned_time, binned_flux

def rv_semi_amplitude(P, M_p, M_star,i=90,e=0,M_p_type='Earth'):
    """
    Calculates expected RV semi-amplitude from input period and masses for planet/star
    
    Units:
    P [days]
    M_p [M_Jup]
    M_star [M_Solar]
    
    K [m/s]
    """
    if M_p_type == 'Earth':
        M_p = M_p/317.77 #Convert Earth masses to Jupiter masses
    
    K = 203*(P**(-1/3)) * ((M_p)*np.sin(i/180*np.pi))/((M_star + 9.548e-4*M_p)**(2/3)) * 1/np.sqrt(1-e**2)
    
    print('Expected RV semi-amplitude, K = {}m/s'.format(K))
    return K
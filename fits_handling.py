#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:17:05 2020

Code for opening FITS files

@author: mbattley
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import lightkurve

def get_lc_from_fits(filename, source='QLP', save_path = ''):
    """
    Obtains lc from fits file if not one of the standard TESS pipelines
    n.b. Currently set up to handle QLP and CDIPS files, though easily extended
    """
#    fits.info(filename)
    hdul = fits.open(filename)
    data = hdul[1].data
#    print(hdul[1].header)
    
    if source == 'QLP':
        target_ID = hdul[0].header['OBJECT']
        sector = hdul[0].header['SECTOR']
        
        time = data['TIME']
        sap_flux = data['SAP_FLUX']
#        kspsap_flux = data['KSPSAP_FLUX']
        quality = data['QUALITY']
        quality = quality.astype(bool)
        
        clean_time = time[~quality]
        clean_flux = sap_flux[~quality]
#        clean_kspsap_flux = kspsap_flux[~quality]
        
        original_lc_fig = plt.figure()
        plt.scatter(clean_time, clean_flux,s=1,c='k')
        plt.title('SAP lc for {}'.format(target_ID))
        plt.xlabel('Time - 2457000 [BTJD Days]')
        plt.ylabel('Relative Flux')
        original_lc_fig.savefig(save_path + "{} - original CDIPS lc.pdf".format(target_ID))
#        plt.show()
        plt.close(original_lc_fig)
        
#        plt.figure()
#        plt.scatter(clean_time, clean_kspsap_flux,s=1,c='k')
#        plt.title('KSPSAP lc for {}'.format(target_ID))
#        plt.show()
        
        hdul.close()
        
        lc = lightkurve.LightCurve(time = clean_time, flux = clean_flux, flux_err = quality[~quality], targetid = target_ID)
        return lc, target_ID, sector
    elif source == 'CDIPS':
#        print('Using CDIPS')
        tic = hdul[0].header['TICID']
        sector = hdul[0].header['SECTOR']
        
        time = data['TMID_BJD']
        time = [x - 2457000 for x in time]
        mag = data['IRM3']
#        pca_mag = data['PCA3']
        mag_err = data['IRE3']
        
        original_lc_fig = plt.figure()
        plt.scatter(time, mag,s=1,c='k')
        plt.title('SAP lc for TIC {}'.format(tic))
        plt.xlabel('Time')
        plt.ylabel('Mag')
        original_lc_fig.savefig(save_path + "TIC {} - original CDIPS lc.pdf".format(tic))
#        plt.show()
        plt.close(original_lc_fig)
        
#        plt.figure()
#        plt.scatter(time, pca_mag,s=1,c='k')
#        plt.title('PCA lc for TIC {}'.format(tic))
#        plt.show()
        
        hdul.close()
        lc = lightkurve.LightCurve(time = time, flux = mag, flux_err = mag_err, targetid = tic)
        return lc, tic, sector
    else:
        print('Please enter a valid source')
        return 

#filename1 = "tess2019247000000-0000000146520535-111-cr_llc.fits"
#filename2 = "tess2019247000000-0000000224225541-111-cr_llc.fits"
#
#sector = 2
#filename = filename1
#
#tic = filename[25:34]
#
## Get info
#fits.info(filename)
##fits_table_filename = fits.util.get_testdata_filepath('btable.fits')
#
## Open fits file and get data
#hdul = fits.open(filename)
#data = hdul[1].data
#print(hdul[1].header)
#
#print(data[0])
#print(data[-1])
#
#time = data['TIME']
#sap_flux = data['SAP_FLUX']
#kspsap_flux = data['KSPSAP_FLUX']
#
#plt.figure()
#plt.scatter(time, sap_flux,s=1,c='k')
#plt.title('SAP lc for TIC {}'.format(tic))
#plt.show()
#
#plt.figure()
#plt.scatter(time, kspsap_flux,s=1,c='k')
#plt.title('KSPSAP lc for TIC {}'.format(tic))
#plt.show()
#
#hdul.close()
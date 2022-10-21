Feel free to use these tools for your own work, but please cite Battley, Pollacco & Armstrong (2020).

This repository contains tools to download and detrend young star light-curves for exoplanetary searches, including LOWESS-detrending, peak-cutting and
interpolation of stellar trends over potential transits.

This is a streamlined version of a larger pipeline still under development, so improvements can be expected later. 

There are two main ways to run the code:
  1. Using ffi_lowess_detrending_simplified.py: This is a simplified version of the Python script used in Battley, Pollacco & Armstrong (2020), which also 
  allows for injection of planet signals. Note that this is now slightly outdated, so I reccommend the next option:
  
  2. Using ffi_lowess_detrending_single_lc_function.py: This is a python function, allowing simple adjustments to be made in the control line without
  having to take a deep-dive into the code. Once the script has been run, the function can be called as lowess_detrend() from the command line. 
  Running this without changing any inputs will give an example detrending of DS Tuc A, showing the automatic recovery of DS Tuc A b/ TOI 200.01,
  the first young exoplanet discovered by TESS.
  
  Inputs/outputs are explained as follows:
      Inputs:
        - target_ID:     String. Name of object, ideally a TIC number e.g. 'TIC 410214986'
        - pipeline:      String. Pipeline that lc is from. Can be '2min', 'SPOC-FFI', 'QLP', 'DIA', or 'tessFFIextract' 
        - detrending:    String. Detrending method of choice. Here 'lowess_full' or 'lowess-partial' are possible
        - save_path:     String. Your local load/save folder. If not supplied will just work from your current directory
        - multi_sector:  True/False. False for a single TESS sector, and True when using multipl
        - find_freq:     True/False. Enables the 'find_freqs' function, which removes the typical 13.5day TESS periodicity and returns the first three strongest periods in the data using a Lomb-Scargle search
        - transit_mask:  True/False. Allows for transit masking while detrending takes place. If True, also requires 'planet_per', 'planet_epoch' and 'planet_dur' inputs
        - transit_mask2: True/False. Allows for a second planet to be masked during detrending in multiplanet systems. Planet parameters here need to be hard-coded as this is reasonably rare
        - use_peak_cut:  True/False. If 'TRUE' automatically cuts the peaks and troughs in sharply evolving light-curves. May require tuning for prominence/expected length
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

Note that 'bad_times' files are built from the engineering quaternion data only, which captures data dumps and times of bad pointing very well, 
but doesn't account for some scattered light effects and other nuisance signals. These will need to be treated separately.

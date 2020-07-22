This repository contains tools to download and detrend young star light-curves for exoplanetary searches, including LOWESS-detrending, peak-cutting and interpolation of stellar trends over potential transits.

This is a streamlined version of a larger pipeline still under development, so improvements can be expected later. 

Feel free to use these tools for your own work, but please cite Battley, Pollacco & Armstrong (2020).

For reference, the main code is located in ffi_lowess_detrending_simplified.py

Note that 'bad_times' files are built from the engineering quaternion data only, which captures data dumps and times of bad pointing very well, but doesn't account for some scattered light effects and other nuisance signals. These will need to be treated separately.

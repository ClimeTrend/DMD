"""
Useful routines for post-processing of raw data. 
The data is assumed to be a list of size N_years of ndarrays of shape (N_days, ny, nx), 
where nx and ny are image sizes along x- and y- axes.
"""

import numpy as np
from scipy.signal import fftconvolve

def get_days_before(data, year_0, day_0, T):
    """
    Get T days before [year_0, day_0], not inclusive of day_0. 
    If don't have this many days, return all that have

    

    Parameters:
        data (list): List of N elements, where each element has shape (N_days, nx, ny). 
                    Note: N_days can be different for different list elements
        year_0 (int): Index of the year (0-based) in the data.
        day_0 (int): Index of the day (0-based) in the specified year.
        T (int): Length of the window for retrieving previous days' data.

    Returns:
        previous_days (numpy.ndarray): Array of shape (t, nx, ny) containing previous days' data, where t<=T.
        Note last element is the day before year_0, day_0
    """
    
    # within year_0, inclusive of year_0
    get_days_before_ = lambda data, year_0, day_0, T: data[year_0][max(day_0-T+1, 0):day_0+1]

    out = get_days_before_(data, year_0, day_0-1, T) 
    
    T -= day_0
    year_0 -= 1
    while T>=0 and year_0>=0:

        day_0 = data[year_0].shape[0]-1
        out1 = get_days_before_(data, year_0, day_0, T) 

        out = np.concatenate((out1, out), axis = 0)
        T -= day_0+1
        year_0 -= 1

    return out


def get_days_after(data, year_0, day_0, T):
    """
    Get T days after [year_0, day_0], inclusive of day_0. 
    If don't have this many days, return all that have

    Same params and return as get_days_before, but for after, inclusive of day0
    """

    assert day_0 <= data[year_0].shape[0]-1

    # within year_0, inclusive of year_0 and possibly last day
    get_days_after_ = lambda data, year_0, day_0, T: \
        data[year_0][day_0: min(day_0+T, data[year_0].shape[0])]

    out = get_days_after_(data, year_0, day_0, T)

    # days left in this year, after day_0, inclusive of day_0
    N_days_left = data[year_0].shape[0]-day_0

    T -= N_days_left
    year_0 += 1
    while T>0 and year_0 <= len(data)-1:
        out1 = get_days_after_(data, year_0, 0, T) 

        out = np.concatenate((out, out1), axis = 0)

        T -= data[year_0].shape[0]
        year_0 += 1

    return out


def window_mean(days_array, window, t = None):
    """
    Compute window-mean of days_array over a given number of days.
    The resulting configurations should be aligned in time with the end of time array

    Parameters: 
        days_array (ndarray of shape (N_days, ny, nx)): daily snapshots
        window (int): time window over which to take the mean
        t: optional array of times

    Returns: 
        windowed mean array of shape (N_days-window+1, ny, nx) of window-means
        [because window-1 first elements cannot be averaged]

        If times array given, returns a truncated times array, so that window is before current time

    Note:
        If needed to get K window-meaned configurations, call for K+window-1 snapshots, 
            extending window-1 into past
    """

    ny = days_array.shape[1]
    nx = days_array.shape[2]
    out = fftconvolve(days_array, np.ones((window, ny, nx))/window, mode = 'valid', axes = 0)
    out[out<0] = 0.

    if t is not None:
        t = t[window-1:]
        out = (out, t)

    return out


def get_test_set(DATA, year, day, window, T_test):
    """
    Perform window-averaging on days after day, year. 
    
    """

    true_after_ = get_days_after(DATA, year, day, T_test)
    true_win = get_days_before(DATA, year, day, window-1)

    true_win_after = np.concatenate((true_win, true_after_), axis = 0)
    true_after = window_mean(true_win_after, window = window, t = None)

    return true_after

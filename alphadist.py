# Contains help functions for the task of transforming a multivariate dataset based on EHR data into 
# a numerical dataset using distance functions.

import string
import numpy as np
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize
  
def znorm_paa_sax(time_series, alpha, w = 3, missing = 'z'):
    """Takes an array containing real values, z-normalizes, reduces
    dimensionality to w, and finally returns a sax representation of length alpha
    
    time series:    array holding a time series of one measurement for one patient
    w:              the dimensionality to reduce to using PAA, set to len(time_series) in plain
    alpha:          alpha is the number of discretized segments that the SAX rep will reflect, set to 2, 3 or 5 in plain using RDS algo
    """
    
    # If time_series is a string, make it into list format e.g. 'abc' -> ['a', 'b', 'c']
        # why? because it's the structure we require for below and i CBA to change it
    if(isinstance(time_series, str)):
        time_series = list(time_series)
    
    if(len(time_series) > 0):
        # normalizing one time series, time series as numpy array (np.array([]))
        normalized_time_series = znorm(np.array(time_series))
        # dimensionality reduction of time series according to w
        paa_norm_time_series = paa(normalized_time_series, w)
        # turning a discretized and reduced time series into a sequence of characters
        return ts_to_string(paa_norm_time_series, cuts_for_asize(alpha))
    else:
        return missing

def levdist(s1, s2):
    """
    Returns the levenshtein distance of two character sequences.
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]

        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def alphabetical_diff(c1, c2):
    """
    Returns the alphabetical difference of two characters.
    """
    return abs(string.ascii_lowercase.index(c1) - string.ascii_lowercase.index(c2))  

def alphadist(s1, s2):
    """
    A standard Levenshtein distance with the addition of substitution operations modified according to the alphabetical difference or
    effort required to change a character into another.
    """
    
    # Switch s1 and s2 such that s1 is the shortest
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:

                if(distances_[0] > len(s1)):
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
                else:
                    distances_.append(alphabetical_diff(c1,c2))
        distances = distances_
    return distances[-1]

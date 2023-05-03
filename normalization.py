import numpy as np
from scipy.stats import iqr, trimboth
from scipy.stats.mstats import winsorize
import math
# Define functions for normalizing the data
def mad_normalize(data, scale_min=0, scale_max=5):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    normalized_data = (data - median) / (2 * mad)
    scaled_data = normalized_data * (scale_max - scale_min) + (scale_max + scale_min) / 2
    return np.clip(scaled_data, scale_min, scale_max)

def iqr_normalize(data, scale_min=0, scale_max=5):
    q1, q3 = np.percentile(data, [10, 90])
    iqr_val = iqr(data)
    normalized_data = (data - q1) / iqr_val
    scaled_data = normalized_data * (scale_max - scale_min) + scale_min
    return np.clip(scaled_data, scale_min, scale_max)

def winsorize_normalize(data, scale_min=0, scale_max=5, lower_bound= 0.05, upper_bound= 0.05):
    winsorized_data = winsorize(data, limits=[lower_bound, upper_bound])
    min_value = np.min(winsorized_data)
    max_value = np.max(winsorized_data)
    normalized_data = (winsorized_data - min_value) / (max_value - min_value)
    scaled_data = normalized_data * (scale_max - scale_min) + scale_min
    return np.clip(scaled_data, scale_min, scale_max)

def normalize_data(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) * 5 for x in data]
    return normalized_data

def normalize_data_with_logarithm(data):
    log_data = [math.log10(x) for x in data]
    min_val = min(log_data)
    max_val = max(log_data)
    normalized_data = [(x - min_val) / (max_val - min_val) * 5 for x in log_data]
    return normalized_data
from matplotlib import pyplot as plt
import time
from pathlib import Path
#%config InlineBackend.figure_format='retina'

import pandas as pd
import argparse
import numpy as np
import os
import json
import sys
from scipy.stats import ks_2samp, chisquare, chi2_contingency, gaussian_kde
from sklearn.compose import ColumnTransformer, make_column_selector as selector


def distribution_intersection_area(A, B):
    """
    This function computes the intersection area between two kernel density estimates.

    inputs:
        A = numpy array of the reference distribution
        B = numpy array of the observed distribution
    
    output:
        area = intersection area in percent
        kde1_x = KDE values for A
        kde1_x = KDE values for B
        idx = intersection points of the two KDEs
        x = range of the distribution
    """

    kde1 = gaussian_kde(A, bw_method = "scott")
    kde2 = gaussian_kde(B, bw_method = "scott")

    xmin = min(A.min(), B.min())
    xmax = max(A.max(), B.max())
    dx = 0.2*(xmax - xmin)
    xmin -= dx
    xmax += dx

    x = np.linspace(xmin, xmax, 1000)
    kde1_x = kde1(x)
    kde2_x = kde2(x)
    idx = np.argwhere(np.diff(np.sign(kde1_x - kde2_x))).flatten()

    area = np.trapz(np.minimum(kde1_x, kde2_x), x) # intersection area between of the kde curves, between the two intersection points
    return area, kde1_x, kde2_x, idx, x  


def add_noise_last_index(array, max_value_reference_distribution,confidence_percentage):
    """
    This function re-populates a constant value according to the method described in add_noise_to_constant().

    inputs:
        array = constant value f(x) = k
        max_value_reference_distribution = max value from the distribution to be compared against (A or B)
        confidence_percentage = confidence interval that determines how many samples will be re-populated
    returns:
        array = recoded sample defined as:
            array interval = x ∈ {[k, max(A,B)]} with the probability defined as: 
            array probabilities: p((k, max(A,B)]), p(k) = constant
            array sample size:
                N (array) = n + n*(1-confidence_percentage) with:
                    n of x ∈ {(k, max(A,B)]} = n*(1-confidence_percentage) -> all values but k
                    n of x ∈ {k} = n -> includes k only

    """
    array = np.append(array, # N = original sample + new syntehtic values
                        np.linspace( # create a uniformly distributed sample
                            start=min(array), # start with the minimum value of the constant
                            stop=max_value_reference_distribution, # define the maximum range as max(A|B)
                            num=int((1-confidence_percentage)*len(array)))) #populate n of x ∈ {(k, max(A,B)]} according to confidence interval
    return array

def add_noise_to_constant(A, B):
    """
    This function is a helper function to compute the intersection area between two kernel density estimates.
    Because the derivative cannot be taken on constant values via numpy (singular matrix error), it is required
    to recode at least one value of a constant. This approach assumes a 95% CI (one tail), where non-constant values
    are populated uniformly between the max value of the reference distribution and the constant. All newly populated
    values contain the same probability but do not affect the distribution assuming a 95% CI. Using this logic, 
    we get plots that have similar KDE around the constant values and which align with the methods used in e.g. seaborn kde plots. 

    inputs:
        A = numpy array of the reference distribution
        B = numpy array of the observed distribution
        A_label = label you want to be displayed in the plot for series A
        B_label = label you want to be displayed in the plot for series B
    returns:
        matplotlib plot of intersection percentage of Gaussian KDE between A & B
    """
    if len(np.unique(np.array(B))) == 1:
        #print("Adding noise to constant value in current dataset")
        #A_freq = dict(zip(np.unique(A, return_counts=True)[0], np.unique(A, return_counts=True)[1]))
        confidence_percentage = 0.975 #this is based on a 95% CI with considering one tail only #max(A_freq.values()) / sum(A_freq.values())
        max_value_reference_distribution = max(A)
        B = add_noise_last_index(B, max_value_reference_distribution, confidence_percentage).astype("int")
    return B


def contingency_table(reference_col, current_col):
    index = list(set(reference_col.unique()) | set(current_col.unique()))

    value_counts_df = pd.DataFrame(reference_col.value_counts(), index=index)
    value_counts_df.columns = ['reference']
    value_counts_df['current'] = current_col.value_counts()
    value_counts_df.fillna(0, inplace=True)

    result = np.array([[value_counts_df['reference'].values], [value_counts_df['current'].values]])
    return result


#Enable argparse to pass None values
def none_or_str(value):
    if value == 'None':
        return None
    return value

parser = argparse.ArgumentParser()
parser.add_argument('--drift_db_path', type=str)
parser.add_argument('--tansformed_data_path', type=str)
parser.add_argument('--threshold', type=float)
args = parser.parse_args()

drift_db_path = args.drift_db_path
tansformed_data_path = args.tansformed_data_path
threshold = args.threshold # flag a variable as drifted if p-value is below the threshold


lines = [
    f"Input path: {tansformed_data_path}",
    f"Output path: {drift_db_path}",
]

for line in lines:
    print(line)


##################################
###### LOAD DATASETS FROM PREVIOUS PIPELINE #####
##################################
print("mounted_path files: ")
arr = os.listdir(tansformed_data_path)
print(arr)

df_list = []
for filename in arr:
    if ".csv" in filename:
        print("reading file: %s ..." % filename)
        with open(os.path.join(tansformed_data_path, filename), "r") as handle:
            input_df = pd.read_csv((Path(tansformed_data_path) / filename))
            df_list.append(input_df)

# Retrieve the current and reference datasets, those come from the previous pipeline
reference = df_list[0]
current = df_list[1]

# use shortlist if exists, else all columns from reference
columns = list(reference.columns) #if shortlist == [] else shortlist

# identify numerical and categorical columns as those are needed to determine the drift detection method
numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(reference[columns])
categorical_columns = categorical_columns_selector(reference[columns])

##################################
###### DRIFT DATABASE ############
##################################

drift_db = []
intersection_db = []

for col in columns:
    if col in numerical_columns:
        ref_arr = np.sort(reference[col])
        curr_arr = np.sort(current[col])

        # statistical test:
        statistic, p_value = ks_2samp(reference[col].values, current[col].values)

    if col in categorical_columns:
        ref_arr = np.sort(reference_le[col])
        curr_arr = np.sort(current_le[col])

        # statistical test:
        observations = contingency_table(reference_le[col], current_le[col])
        statstic, p_value, dof, _ = chi2_contingency(observations)
        
    if p_value <= threshold:
        drift_indication = 'Drift'
    if p_value > threshold:
        drift_indication = 'No Drift'


    # add noise to enable plotting when constant values present
    ref_arr = add_noise_to_constant(A=curr_arr, B=ref_arr) #this assumes the reference distribution is a constant
    curr_arr = add_noise_to_constant(A=ref_arr, B=curr_arr) #this assumes the current distribution is a constant

    area, kde1_x, kde2_x, idx, x = distribution_intersection_area(ref_arr,curr_arr)

    # look up interval values and length of KDE array to generate sequences
    array_len = len(kde1_x)

    col_n = pd.Series(np.full(fill_value=col, shape=array_len))
    kde_overlap_n = pd.Series(np.full(fill_value=area, shape=array_len))
    drift_indication_n = pd.Series(np.full(fill_value=drift_indication, shape=array_len))
    threshold_n = pd.Series(np.full(fill_value=threshold, shape=array_len))


    drift_db_n = pd.DataFrame([kde1_x, kde2_x, x]).T
    drift_db_n = pd.concat([col_n, drift_db_n, kde_overlap_n, drift_indication_n, threshold_n], axis=1)
    drift_db_n.columns = [
        "column", "reference_kde_values_y", "current_kde_values_y",
         "x_axis", "kde_overlap", "drift_indication", "p_val_threshold"
        ]
    drift_db.append(drift_db_n)

    col_n = pd.Series(np.full(fill_value=col, shape=len(x[idx])))
    intersection_db_n = pd.DataFrame(x[idx], kde2_x[idx]).reset_index()
    intersection_db_n = pd.concat([col_n, intersection_db_n], axis=1)
    intersection_db_n.columns = ["column", "intersection_x", "intersection_y"]
    intersection_db.append(intersection_db_n)

drift_db = pd.concat(drift_db)
intersection_db = pd.concat(intersection_db)

# -------------------------
# SAVE FILES

print(f"Saving to{drift_db_path}")
date = time.strftime("%Y-%m-%d")
# Log with date
#drift_db = drift_db.to_csv((Path(drift_db_path) / f"drift_db_processed_{date}.csv"), index = False)
#intersection_db = intersection_db.to_csv((Path(drift_db_path) / f"intersection_db_processed_{date}.csv"), index = False)
# Overwrite
drift_db = drift_db.to_csv((Path(drift_db_path) / "drift_db_processed.csv"), index = False)
intersection_db = intersection_db.to_csv((Path(drift_db_path) / "intersection_db_processed.csv"), index = False)


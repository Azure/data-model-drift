from matplotlib import pyplot as plt
#%config InlineBackend.figure_format='retina'

import pandas as pd
import argparse
import numpy as np
import os
import json
import sys
from scipy.stats import ks_2samp, chisquare, chi2_contingency, gaussian_kde

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from azureml.core import Workspace, Dataset, Run

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

def plot_distribution_overlap(A, B, A_label='reference', B_label='current', ax=None):
    """
    This function computes the intersection area between two kernel density estimates.

    inputs:
        A = numpy array of the reference distribution
        B = numpy array of the observed distribution
        A_label = label you want to be displayed in the plot for series A
        B_label = label you want to be displayed in the plot for series B
    returns:
        matplotlib plot of intersection percentage of Gaussian KDE between A & B
    """
    if ax is None:
        ax = plt.gca()
    

    area, kde1_x, kde2_x, idx, x = distribution_intersection_area(A, B) # intersection area between of the kde curves, between the two intersection points

    ax.plot(x, kde1_x, color = 'dodgerblue', label = A_label, linewidth = 2)
    ax.plot(x, kde2_x, color = 'orangered', label = B_label, linewidth = 2)

    ax.fill_between(x, np.minimum(kde1_x, kde2_x), 0, color = 'lime', alpha = 0.3, label = 'intersection')

    ax.plot(x[idx], kde2_x[idx], 'ko')

    handles, labels = ax.get_legend_handles_labels()
    labels[2] += f': {area * 100:.1f}%'
    ax.legend(handles, labels)


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
parser.add_argument('--reference_dataset', type=str)
parser.add_argument('--current_dataset', type=str)
parser.add_argument('--threshold', type=float)
parser.add_argument('--shortlist',nargs='*', type=none_or_str, default=[])
args = parser.parse_args()

reference_dataset = args.reference_dataset
current_dataset = args.current_dataset
shortlist = args.shortlist
threshold = args.threshold # flag a variable as drifted if p-value is below the threshold


##################################
######## LOAD AML DATASETS #######
##################################

run = Run.get_context()
ws = run.experiment.workspace   

reference = Dataset.get_by_name(ws, name=reference_dataset).to_pandas_dataframe() # reference dataset (A)
current = Dataset.get_by_name(ws, name=current_dataset).to_pandas_dataframe() # current dataset (B)

print("PREPROCESS DATASET AND ENCODE CATEGORICAL VARIABLES")

##################################
##### PREPROCESS CATEGORICALS ####
##################################

# -------------------------
# LABEL ENCODE 

# use shortlist if exists, else all columns from reference
columns = reference.columns if shortlist == [None] else shortlist

# identify numerical and categorical columns
numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(reference[columns])
categorical_columns = categorical_columns_selector(reference[columns])
   
# label encoding for plots of categorical columns
categorical_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

reference_le = categorical_transformer.fit_transform(reference[categorical_columns])
reference_le = pd.DataFrame(reference_le)
reference_le.columns = categorical_columns

current_le = categorical_transformer.transform(current[categorical_columns])
current_le = pd.DataFrame(current_le)
current_le.columns = categorical_columns

print(f'numerical columns: {numerical_columns}')
print(f'categorical columns: {categorical_columns}')

# -------------------------
# IMPUTE MISSING VALUES

imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
current_le = imp_mode.fit_transform(current_le[categorical_columns])
current_le = pd.DataFrame(current_le)
current_le.columns = categorical_columns

imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
reference_le = imp_mode.fit_transform(reference_le[categorical_columns])
reference_le = pd.DataFrame(reference_le)
reference_le.columns = categorical_columns


##################################
############### PLOT #############
##################################

# prepare plot
total = len(columns)
display_cols = 4

rows = total // display_cols
rows += total % display_cols
position = range(1, total + 1)

fig = plt.figure(figsize=(30, 4 * rows))
fig.subplots_adjust(hspace= 0.3, wspace=0.2)

drift_cols = 0
for k, col in enumerate(columns):
    
    if col in numerical_columns:
        
        # statistical test:
        statistic, p_value = ks_2samp(reference[col].values, current[col].values)
        if p_value < threshold:
            drift_indication = 'Drift'
            drift_cols += 1
        else:
            drift_indication = 'No drift'
        annot = f'{col}: K-S test p_value = {p_value:.4f} ({drift_indication})'
        
        # plot:
        ref_arr = np.sort(reference[col])
        curr_arr = np.sort(current[col])

        # add noise to enable plotting when constant values present
        ref_arr = add_noise_to_constant(A=curr_arr, B=ref_arr) #this assumes the reference distribution is a constant
        curr_arr = add_noise_to_constant(A=ref_arr, B=curr_arr) #this assumes the current distribution is a constant

        ax = fig.add_subplot(rows, display_cols, position[k])
        try: 
            plot_distribution_overlap(A = ref_arr, B = curr_arr, A_label = 'reference', B_label ='current', ax=ax)
        except Exception as e:
            annot = f'{col} - {e}'
            print(annot)

        ax.set_title(annot)
       
    elif col in categorical_columns:
        
        # statistical test:
        observations = contingency_table(reference_le[col], current_le[col])
        statstic, p_value, dof, _ = chi2_contingency(observations)
        if p_value < threshold:
            drift_indication = 'Drift'
            drift_cols += 1
        else:
            drift_indication = 'No drift'
        annot = f'{col}: Chi-square test p_value = {p_value:.4f} ({drift_indication})'

        # plot:
        ref_arr = np.sort(reference_le[col])
        curr_arr = np.sort(current_le[col])

        # add noise to enable plotting when constant values present
        ref_arr = add_noise_to_constant(A=curr_arr, B=ref_arr) #this assumes the reference distribution is a constant
        curr_arr = add_noise_to_constant(A=ref_arr, B=curr_arr) #this assumes the current distribution is a constant

        ax = fig.add_subplot(rows, display_cols, position[k])
        
        try:
            plot_distribution_overlap(A = ref_arr, B = curr_arr, A_label = 'reference', B_label ='current', ax=ax)
        except Exception as e:
            annot = f'{col} - {e}'
            print(annot)
        
        ax.set_title(annot)
        
    else:
        print('Columns type not recognized')

drift_stat = f'Drift identified in {drift_cols} of {len(columns)} columns ({drift_cols/len(columns) * 100:.1f} %).'
plt.suptitle(drift_stat, y=0.94, fontsize = 18)

# -------------------------
# SAVE IMAGES

plt.savefig('foo.png', bbox_inches='tight')


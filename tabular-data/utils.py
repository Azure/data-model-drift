from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chisquare, chi2_contingency, gaussian_kde

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


def spilt_data(data, rct_split=(0.4, 0.4, 0.2), shuffle=False):
    if shuffle:
        data = data.sample(frac=1.0)

    rc_split_idx = round(rct_split[0] * data.shape[0])
    ct_split_idx = round((rct_split[0]+ (rct_split[1])) * data.shape[0])

    reference = data[:rc_split_idx]
    current = data[rc_split_idx:ct_split_idx]
    test = data[ct_split_idx:]

    return reference, current, test

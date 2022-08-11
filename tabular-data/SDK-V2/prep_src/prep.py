import pandas as pd
import argparse
import numpy as np
import os
from pathlib import Path

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

#Enable argparse to pass None values
def none_or_str(value):
    if value == 'None':
        return None
    return value

parser = argparse.ArgumentParser("prep")
parser.add_argument('--input_path', type=str)
parser.add_argument('--output_path', type=str)
#parser.add_argument('--shortlist',nargs='*', type=none_or_str, default=[])
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path

lines = [
    f"Input path: {input_path}",
    f"Output path: {output_path}",
]

for line in lines:
    print(line)


##################################
######## LOAD AML DATASETS #######
##################################
print("mounted_path files: ")
arr = os.listdir(input_path)
print(arr)

df_list = []
dataset_names = []
for filename in arr:
    if ".csv" in filename:
        print("reading file: %s ..." % filename)
        with open(os.path.join(input_path, filename), "r") as handle:
            input_df = pd.read_csv((Path(input_path) / filename))
            df_list.append(input_df)
            dataset_names.append(filename)


# Retrieve the current and reference datasets
reference = df_list[0]
current = df_list[1]

reference_name = dataset_names[0][:-4]
current_name = dataset_names[1][:-4]

#reference = Dataset.get_by_name(ws, name=reference_dataset).to_pandas_dataframe() # reference dataset (A)
#current = Dataset.get_by_name(ws, name=current_dataset).to_pandas_dataframe() # current dataset (B)

print("PREPROCESS DATASET AND ENCODE CATEGORICAL VARIABLES")

##################################
##### PREPROCESS CATEGORICALS ####
##################################

# -------------------------
# LABEL ENCODE 

# use shortlist if exists, else all columns from reference
columns = list(reference.columns) #if shortlist == [] else shortlist

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

# impute missing values

if categorical_columns != []:

    imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    current_le = imp_mode.fit_transform(current_le[categorical_columns])
    current_le = pd.DataFrame(current_le)
    current_le.columns = categorical_columns

    imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    reference_le = imp_mode.fit_transform(reference_le[categorical_columns])
    reference_le = pd.DataFrame(reference_le)
    reference_le.columns = categorical_columns

# join categorical and numerical values back
reference_joined = pd.concat([reference[numerical_columns], reference_le], axis=1)
current_joined = pd.concat([current[numerical_columns], current_le], axis=1)


# -------------------------
# SAVE FILES

#create folder if folder does not exist already. We will save the files here
#Path(output_path).mkdir(parents=True, exist_ok=True)
print(f"Saving to{output_path}")

reference_joined = reference_joined.to_csv((Path(output_path) / f"{reference_name}_processed.csv"), index = False)
current_joined = current_joined.to_csv((Path(output_path) / f"{current_name}_processed.csv"), index = False)


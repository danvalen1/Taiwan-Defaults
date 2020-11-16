import pandas as pd
import numpy as np

# global variables
drop_index_list = [18381]

dummy_list =['marital_status',
             'gender',
             'age',
             'education'
            ]

def clean_df(target_csv):
    """Clean data from CSV and convert into df.
        Parameters:
            target_csv (str): String indicating where the CSV is located.
        Returns:
            df (pd.DataFrame): Clean dataframe.
    """
    # read csv
    df = pd.read_csv(target_csv)
    
    # rename the columns
    df = rename_cols(df)
    
    # drop rows for training data set
    if 'training' in target_csv:
        df = drop_row(df)
    
    # converts objs to nums
    df = convert_objs(df)
    
    # transform features
    df = feature_changes(df)
    
    # creates dummy vars
    df = feature_dummies(df, dummy_list, drop_first=True)
    
    return df

def rename_cols(df):  
    df.drop(columns="Unnamed: 0", inplace=True) # drop unncessary columns
    rename_list = ["max_credit", "gender", "education", "marital_status", "age",
                   "pay_status_sep", "pay_status_aug", "pay_status_jul", "pay_status_jun", "pay_status_may", "pay_status_apr",
                   "bill_sep", "bill_aug", "bill_jul", "bill_jun", "bill_may", "bill_apr",
                   "payment_sep", "payment_aug", "payment_jul", "payment_jun", "payment_may", "payment_apr",
                    "default"]
    col_rename = dict(zip(df.columns,rename_list))
    df = df.rename(columns=col_rename)
    return df

def convert_objs(df):
    df = df.apply(pd.to_numeric)
    return df

def drop_row(df):
    for i in drop_index_list:
        df.drop(index=i, inplace=True)
    return df

def feature_dummies(df, list_of_dummyvars, drop_first=True):
    
    # get dummies and drop first for reference category
    df_dummies = pd.get_dummies(df, columns=list_of_dummyvars, prefix=list_of_dummyvars, drop_first=drop_first)
    
    return df_dummies

def feature_changes(df):
    # Binning ages
    df['age'] = df.age.apply(age_binning)
    
    # Engineered features
    df = feat_eng(df)
    return df
    
def age_binning(n):
    if n < 30:
        return 0
    elif n < 40:
        return 1
    elif n < 50:
        return 2
    elif n < 60:
        return 3
    elif n < 70:
        return 4
    else:
        return 5
    
def feat_eng(df):
    mon_list = ['apr', 'may', 'jun', 'jul', 'aug', 'sep']
    
    for m in mon_list:
        # paid bill proportion
        df['payment_level_' + m] = df['payment_' + m] / df['bill_' + m]
        ## exception for 0 bills and infinity bills
        df['payment_level_' + m].replace(np.inf, 1, inplace=True)
        df['payment_level_' + m].fillna(0, inplace = True)
        
        # credit use for each month
        df['cred_level_' + m] = df['bill_' + m] / df['max_credit']

        
#         #drop vars
#         df.drop(columns=['bill_'+m, 'payment_'+m],  inplace=True)
        
    return df
        
        
    
    
    
import pandas as pd

drop_index_list = [18381]

def clean_df(target_csv):
    """Clean data from CSV and convert into df.
        Parameters:
            target_csv (str): String indicating where the CSV is located.
        Returns:
            df (pd.DataFrame): Clean dataframe.
    """
    # read csv
    df = pd.read_csv('../src/data/training_data.csv')
    
    # rename the columns
    df = rename_cols(df)
    
    # converts objs to nums
    df = convert_objs(df)
    
    
    return df

def rename_cols(df):  
    df.drop(columns="Unnamed: 0", inplace=True) # drop unncessary columns
    rename_list = ["max_credit", "gender", "education", "marital_status", "age",
                   "pay_status_sep", "pay_status_aug", "pay_status_jul", "pay_status_jun", "pay_status_may", "pay_status_apr",
                   "bill_sep", "bill_aug", "bill_jul", "bill_jun", "bill_may", "bill_apr",
                   "payment_sep", "payment_aug", "payments_jul", "payment_jun", "payment_may", "payment_apr",
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
    
    
    
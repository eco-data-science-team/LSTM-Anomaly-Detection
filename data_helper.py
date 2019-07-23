import pandas as pd

def create_standard_multivariable_df(df, point_location = 0, shift = 1, rename_OAT = True):
    #this function creates a standard 50 variable DataFrame
    """
    variables generated:
        - CDD: Cooling Degree Days (OAT-65), 0 where CDD < 0 
        - HDD: Heating Degree Days (65-OAT), 0 where HDD < 0
        - CDD2: (Cooling Degree Days)^2
        - HDD2: (Heating Degree Days)^2
        - MONTH (1-12): 1 on month of data point, 0 on all other months
        - TOD (0-23): 1 on "TIME OF DAY" of data point, 0 on all other times
        - DOW (0-6): 1 on "DAY OF WEEK" of data point, 0 on all other days
        - WEEKEND: 1 if data point falls on weekend, 0 for everything else
        - Shift_N: user defined shift, where N is the amount of lookback
        - Rolling24_mean: generates the rolling 24hr mean 
        - Rolling24_max: takes the maximun value of the rolling 24hr
        - Rolling 24_min: takes the minimum value of the rolling 24hr
    """
    
    if rename_OAT:
        df.rename(columns={'aiTIT4045':'OAT'}, inplace=True)
    start_col = len(df.columns)
    df["CDD"] = df.OAT - 65.0
    df.loc[df.CDD < 0 , "CDD"] = 0
    df["HDD"] = 65.0 - df.OAT
    df.loc[df.HDD < 0, "HDD" ] = 0
    df["CDD2"] = df.CDD ** 2
    df["HDD2"] = df.HDD ** 2
    
    month = [str("MONTH_" + str(x+1)) for x in range(12)]
    df["MONTH"] = df.index.month
    df.MONTH = df.MONTH.astype('category')
    month_df = pd.get_dummies(data = df, columns = ["MONTH"])
    month_df = month_df.T.reindex(month).T.fillna(0)
    month_df = month_df.drop(month_df.columns[0], axis = 1)

    tod = [str("TOD_" + str(x)) for x in range(24)]
    df["TOD"] = df.index.hour
    df.TOD = df.TOD.astype('category')
    tod_df = pd.get_dummies(data = df, columns = ["TOD"])
    tod_df = tod_df.T.reindex(tod).T.fillna(0)
    tod_df = tod_df.drop(tod_df.columns[0], axis = 1)

    dow = [str('DOW_' + str(x)) for x in range(7)]
    df["DOW"] = df.index.weekday
    df.DOW = df.DOW.astype('category')
    dow_df = pd.get_dummies(data = df, columns = ["DOW"])
    dow_df = dow_df.T.reindex(dow).T.fillna(0)
    dow_df = dow_df.drop(dow_df.columns[0], axis = 1)

    df["WEEKEND"] = 0
    df.loc[(dow_df.DOW_5 == 1) | (dow_df.DOW_6 == 1), 'WEEKEND'] = 1

    shift_col = "SHIFT_" + str(shift)
    df[shift_col] = df.iloc[ : , point_location].shift(shift)

    df["Rolling24_mean"] = df.iloc[ : , point_location].rolling("24h").mean()
    df["Rolling24_max"] = df.iloc[ : , point_location].rolling("24h").max()
    df["Rolling24_min"] = df.iloc[ : , point_location].rolling("24h").min()

    df = pd.concat([df, month_df, tod_df, dow_df], axis = 1)
    df.drop(['MONTH', 'TOD', 'DOW'], axis = 1, inplace = True)
    df.dropna(inplace = True)

    del month_df
    del tod_df
    del dow_df
    print(f'Generated: {len(df.columns) - start_col} columns')
    return df


def add_variable_to_df(df,fnc,col_name, kwargs):
    
    df[col_name] = fnc(df, **kwargs)
    
    return df
    
def clean_train_data(df, eval_expression = None):
    """
    Used to extract certain values from the DatFrame
    """
    
    if eval_expression:
        evaluated = []
        for exp in eval_expression:
            print(f"Evaluating: {exp}")
            evaluated.append(pd.eval(exp))
        if len(eval_expression) > 1 :
            return tuple(evaluated)
        else:
            return evaluated[0]
    else:
        print("No expression passed")
        return df
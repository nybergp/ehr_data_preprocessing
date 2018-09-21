
# Used to filter a dataset for a set of ICD-10 diagnosis codes into positive and negative examples 
# within a 90 day window of a code related to an Adverse Drug Event (ADE) occuring.

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def label_ADE(row, code):
   if row['patientnr'] in negative[code]:
      return 0
   else:
      return 1

def diff(row):    
    ad = datetime(int('20'+row[1][2:4]),int(row[1][5:7]),int(row[1][8:10]))
    ev = datetime(int('20'+row[2][2:4]),int(row[2][5:7]),int(row[2][8:10]))
    timedelta(7)
    diff = (ev-ad).days
    return diff

def within(pid, event):
    ADE_date = l270_d_comp_dates.loc[pid]
    
    ad = datetime(int('20'+ADE_date[2:4]),int(ADE_date[5:7]),int(ADE_date[8:10]))
    ev = datetime(int('20'+event[2:4]),int(event[5:7]),int(event[8:10]))
    timedelta(7)
    diff = (ev-ad).days
    if(diff > -1 or diff < -90):
        return False
    return True

AT_datasets = True

pos_diag_codes = [
        'L271',
        'O355',
        'T783',
        'T784',
        'T808',
        'T887',        
        'D611',
        'D642',
        'D695',
        'L270']

neg_diag_codes = [
        'L27',
        'O35',
        'T78',
        'T78',
        'T80',
        'T88',        
        'D61',
        'D64',
        'D69',
        'L27']


data = pd.read_csv('datasets/X-file/X.csv', skipinitialspace=False, skiprows=range(1,10000))

data = data.rename(columns={'evalue': 'value', 'ecode': 'code', 'pid': 'patientnr'})
# filtering for rows that contain drug prescriptions or where and ICD-code corresponding to a neg_diag_code exists
data = data.loc[(data['etype'] == 'M') | ((data['etype'] == 'D') & (data['code'].str.contains(('|'.join(neg_diag_codes)))))]
data = data[pd.notnull(data['date1'])]

# if using AT representation, we drop all rows that contain null date2 since it is required for the calculation
# otherwise we want to keep rows that have null date2 since this does not matter for other representations
if(AT_datasets):
    data = data[pd.notnull(data['date2'])]

negative = {}
positive = {}
combined = {}

for code in pos_diag_codes:
    positive[code] = data[data['code'] == code].patientnr.unique()
    combined[code] = data[data['code'].str.contains(code[:-1], na=False)]
    negative[code] = combined[code][~combined[code]['patientnr'].isin(positive[code])].patientnr.unique()

for code in pos_diag_codes:
    raw_dataset_name = 'datasets/10_raw_drug_pres_AT/' + code + '-90-raw-measurements-run2.csv'
    l270_d = data[(data['patientnr'].isin(combined[code].patientnr.unique()) & (data['code'].str.contains(code[:-1], na=False)))]
    l270_d_comp_dates = l270_d.groupby('patientnr')['date1'].max()
    l270 = data[(data['patientnr'].isin(combined[code].patientnr.unique()) & (data['etype'] == 'M'))]
    l270['ADE'] = l270.apply(lambda row: label_ADE (row, code),axis=1)
    l270['within_date'] = l270.apply(lambda row: 'within' if within(row['patientnr'], row['date1']) else 'outside', axis=1)
    within_dates = l270[l270.within_date != 'outside']
    
    # update value column
    within_dates['value'] = within_dates.apply(diff, axis=1)
    
    within_dates = within_dates.drop(['date2', 'etype', 'within_date'], axis=1)
    within_dates.rename(columns={'date1': 'time'}, inplace=True)   
    within_dates.to_csv(raw_dataset_name, encoding='utf-8', index=False)
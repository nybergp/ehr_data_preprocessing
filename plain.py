
import numpy as np
import pandas as pd
import alphadist as ad
import utils as ut

rdm = np.random.RandomState(seed=123)

# Change these functions to adjust PAA/Raw time series to input, e.g w = series.count() means no PAA
def paa_sax_znorm_agg_2(series):
    tr = ad.znorm_paa_sax(series.tolist(), 2, w = 3, missing = 'z')
    return tr

def paa_sax_znorm_agg_3(series):
    tr = ad.znorm_paa_sax(series.tolist(), 3, w = 3, missing = 'z')
    return tr

def paa_sax_znorm_agg_5(series):
    tr = ad.znorm_paa_sax(series.tolist(), 5, w = 3, missing = 'z')
    return tr


raw_datasets = [
        'L271-90',
        'O355-90',
        'T783-90',
        'T784-90',
        'T808-90',
        'T887-90',        
        'D611-90',
        'D642-90',
        'D695-90',
        'L270-90']

distance_measures = {'editdist': [ut.sliding_ed, 'ed'], 'alphadist': [ad.alphadist, 'ad']}

def plain(distance_measures):
    
    for distance_measure in distance_measures:    
        for raw_dataset_name in raw_datasets:
            print(distance_measure, ' : ', raw_dataset_name)
            #save_path = 'results/drug_pres/' + distance_measure + '/' + raw_dataset_name + '-' + distance_measures[distance_measure][1] + '.csv'
            save_path = 'results/PAA5/clin_meas/' + distance_measure + '/' + raw_dataset_name + '-' + distance_measures[distance_measure][1] + '.csv'
            # Change this line to switch between clin_meas and drug_pres
            raw_dataset_name = 'datasets/10_raw_clin_meas/' + raw_dataset_name + '-raw-measurements.csv'
            
            # raw data frame
            data = pd.read_csv(raw_dataset_name)

            # time axis not used
            data = data.drop('time', axis=1)
            
            # getting a list of class labels to be applied to pivot dataframe later
            class_labels = []
            for pid in data.patientnr.unique():
                class_labels.append(data[(data.patientnr == pid)].iloc[0]['ADE'])
            
            data = data.drop('ADE', axis = 1)
                
            # Create datasets for 3 alpha levels            
            df_alpha_2 = data.pivot_table(data, index=('patientnr'),columns=('code'),aggfunc=paa_sax_znorm_agg_2).fillna('z')
            df_alpha_3 = data.pivot_table(data, index=('patientnr'),columns=('code'),aggfunc=paa_sax_znorm_agg_3).fillna('z')
            df_alpha_5 = data.pivot_table(data, index=('patientnr'),columns=('code'),aggfunc=paa_sax_znorm_agg_5).fillna('z')
            
            df_alpha_2.columns = ["".join((j)) for i,j in df_alpha_2.columns]
            df_alpha_2.reset_index()
            
            df_alpha_3.columns = ["".join((j)) for i,j in df_alpha_3.columns]
            df_alpha_3.reset_index()
            
            df_alpha_5.columns = ["".join((j)) for i,j in df_alpha_5.columns]
            df_alpha_5.reset_index()
            
            # 1. NOW, we compare all features in the three dataframes
                # with each other, producing a fourth dataset where every feature may have a different alpha
            # 2. The comparator used is the info gain, alpha with highest IG is best and is used to pick the 
                    # alpha for each feature
            best_alphas = df_alpha_2.copy()
            # basically RDS-algo loop
            for i, code in enumerate(data.code.unique()):
                print(i,': ',code)
                if pd.notnull(code):
                    alpha_2_list = df_alpha_2[code].tolist()
                    alpha_3_list = df_alpha_3[code].tolist()
                    alpha_5_list = df_alpha_5[code].tolist()
                
                    # candidates alpha 2
                    candidates_2 = set([ut.get_random_subsequence(alpha_2_list, rdm) for _ in range(100)])
                    candidates_2 = list(sorted(candidates_2))        
                    # candidates alpha 3
                    candidates_3 = set([ut.get_random_subsequence(alpha_3_list, rdm) for _ in range(100)])
                    candidates_3 = list(sorted(candidates_3))        
                    # candidates alpha 5
                    candidates_5 = set([ut.get_random_subsequence(alpha_5_list, rdm) for _ in range(100)])
                    candidates_5 = list(sorted(candidates_5))
                
                    # evaluate candidate lists for three alpha values
                    candidate_evals_2 = [ut.evaluate_candidate(c,
                                                      [distance_measures[distance_measure][0](s, c) for s in alpha_2_list],
                                                      class_labels,
                                                      ut.entropy(class_labels),
                                                      missing='plain') for c in candidates_2]
                
                    candidate_evals_3 = [ut.evaluate_candidate(c,
                                                      [distance_measures[distance_measure][0](s, c) for s in alpha_3_list],
                                                      class_labels,
                                                      ut.entropy(class_labels),
                                                      missing='plain') for c in candidates_3]
                
                    candidate_evals_5 = [ut.evaluate_candidate(c,
                                                      [distance_measures[distance_measure][0](s, c) for s in alpha_5_list],
                                                      class_labels,
                                                      ut.entropy(class_labels),
                                                      missing='plain') for c in candidates_5]    
                    
                    #select candidate (shapelet) yielding maximum information gain (to break ties, max margin and min length)
                    shapelet_2 = sorted(candidate_evals_2, key = lambda e : (-e['ig'],#max ig
                                                                     -e['margin'],#max margin
                                                                     len(e['subseq'])))[0]#min length
                
                    shapelet_3 = sorted(candidate_evals_3, key = lambda e : (-e['ig'],#max ig
                                                                     -e['margin'],#max margin
                                                                     len(e['subseq'])))[0]#min length
                    shapelet_5 = sorted(candidate_evals_5, key = lambda e : (-e['ig'],#max ig
                                                                     -e['margin'],#max margin
                                                                     len(e['subseq'])))[0]#min length
                    
                    # calculate IG of three converted columns according to a = 2, 3, 5
                    # shapelet yielding largest IG will be used for this particular feature
                        # as it's considered the most informative
                    
                    if shapelet_2['ig'] > shapelet_3['ig'] and shapelet_2['ig'] > shapelet_5['ig']:
                        #print("shapelet 2 chosen: ",shapelet_2['subseq'],' for ',code)
                        best_alphas[code] = [distance_measures[distance_measure][0](s, shapelet_2['subseq']) for s in alpha_2_list]
                    elif shapelet_3['ig'] > shapelet_2['ig'] and shapelet_3['ig'] > shapelet_5['ig']:
                        #print("shapelet 3 chosen: ",shapelet_3['subseq'],' for ',code)
                        best_alphas[code] = [distance_measures[distance_measure][0](s, shapelet_3['subseq']) for s in alpha_3_list]
                    else:
                        #print("shapelet 5 chosen: ",shapelet_5['subseq'],' for ',code)
                        best_alphas[code] = [distance_measures[distance_measure][0](s, shapelet_5['subseq']) for s in alpha_5_list]
            
            # best_alphas should now contain only numbers according to the plain method
                # each feature is based on either an alpha of 2, 3, or 5 depending 
                    # on which setting yielded the highest information gain
            
            best_alphas['ADE'] = class_labels
            best_alphas.to_csv(save_path, encoding='utf-8', index=False)
            print(save_path,' saved.')

plain(distance_measures)
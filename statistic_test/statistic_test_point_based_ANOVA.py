import numpy as np
from scipy import stats
import os
import pandas as pd
import pingouin as pg
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

path_aware='/Users/piayu/Dropbox/Mac/Downloads/experiment_data/statistic_test/trust_aware/trust_aware_combined/'
path_free='/Users/piayu/Dropbox/Mac/Downloads/experiment_data/statistic_test/trust_free/trust_free_combined/'
path_aware_trial1='/Users/piayu/Dropbox/Mac/Downloads/experiment_data/statistic_test/trust_aware/trust_aware_trial1/'
path_free_trial1='/Users/piayu/Dropbox/Mac/Downloads/experiment_data/statistic_test/trust_free/trust_free_trial1/'
path_aware_trial2='/Users/piayu/Dropbox/Mac/Downloads/experiment_data/statistic_test/trust_aware/trust_aware_trial2/'
path_free_trial2='/Users/piayu/Dropbox/Mac/Downloads/experiment_data/statistic_test/trust_free/trust_free_trial2/'

#-------trust aware csv file loading---------------
dataframes_aware = []
file_list_aware =  os.listdir(path_aware)

for file in file_list_aware:
    if file.endswith('.csv'):
        file_path = os.path.join(path_aware, file)
        data = pd.read_csv(file_path)
        dataframes_aware.append(data)

#-------trust free csv file loading---------------
dataframes_free = []
file_list_free =  os.listdir(path_free)
num_csv_file = len(dataframes_aware)
print(num_csv_file)
for file in file_list_free:
    if file.endswith('.csv'):
        file_path = os.path.join(path_free, file)
        data = pd.read_csv(file_path)
        dataframes_free.append(data)

#-------trust aware data for comparison---------------
precetage_aware = []
trust_average_aware = []
takeover_probability_aware = []
for i in range(len(dataframes_aware)):
    data = dataframes_aware[i]
    num_rows = data.shape[0]
    num_columns  = data.shape[1]
    num = 0
    num_total = 0
    num_satisfy = 0
    num_notsatisfy = 0
    trust = 0
    for k in range(num_rows):
        if str(data.iloc[k, 3]) == 'True':
            num_total += 1
            trust += data.iloc[k, 1]
            if data.iloc[k, 1] >= 2:
                num_satisfy += 1
            else:
                num_notsatisfy += 1
        else:
            num += 1
    
    precetage_aware.append(num_satisfy/num_total)
    trust_average_aware.append(trust/num_total)
    takeover_probability_aware.append(num/(num+num_total))

#-------trust free data for comparison---------------
precetage_free = []
trust_average_free = []
takeover_probability_free = []
for i in range(len(dataframes_free)):
    data = dataframes_free[i]
    num_rows = data.shape[0]
    num = 0
    num_total = 0
    num_satisfy = 0
    num_notsatisfy = 0
    trust = 0
    for k in range(num_rows):
        if str(data.iloc[k, 3]) == 'True':
            num_total += 1
            trust += data.iloc[k, 1]
            if data.iloc[k, 1] >= 2:
                num_satisfy += 1
            else:
                num_notsatisfy += 1
        else:
            num += 1

    precetage_free.append(num_satisfy/num_total)
    trust_average_free.append(trust/num_total)
    takeover_probability_free.append(num/(num+num_total))

# #----------------------
# print('Trust aware satisfaction precentage: %s' %precetage_aware)
# print('Trust free  satisfaction precentage: %s' %precetage_free)
# print('Trust aware average trust: %s' %trust_average_aware)
# print('Trust free  average trust: %s' %trust_average_free)
# print('Trust aware takeover probability: %s' %takeover_probability_aware)
# print('Trust free  takeover probability: %s' %takeover_probability_free)

#---------------Repeated-measures ANOVA------------------
Average_trust = []
Precentage = []
if num_csv_file == 40:
    for i in range(0, 40, 2):
        Average_trust.append(trust_average_aware[i])
        Average_trust.append(trust_average_aware[i+1])
        Average_trust.append(trust_average_free[i])
        Average_trust.append(trust_average_free[i+1])
        Precentage.append(precetage_aware[i])
        Precentage.append(precetage_aware[i+1])
        Precentage.append(precetage_free[i])
        Precentage.append(precetage_free[i+1])
    
    dataframe1 = pd.DataFrame({'Participants': np.repeat([i for i in range(1, 21)], 4),
                            'Policy': np.tile([1, 2, 3, 4], 20), 
                            'Average_trust': Average_trust,
                            })
    
    aovrm1 = AnovaRM(data=dataframe1, depvar='Average_trust',
                subject='Participants', within=['Policy'])
    aovrm_results1 = aovrm1.fit()
    print(aovrm_results1)
    tukey_results1 = pairwise_tukeyhsd(dataframe1['Average_trust'], dataframe1['Policy'])
    print(tukey_results1)

    dataframe2 = pd.DataFrame({'Participants': np.repeat([i for i in range(1, 21)], 4),
                            'Policy': np.tile([1, 2, 3, 4], 20), 
                            'Precentage': Precentage,
                            })
    aovrm2 = AnovaRM(data=dataframe2, depvar='Precentage',
                subject='Participants', within=['Policy'])
    aovrm_results2 = aovrm2.fit()
    print(aovrm_results2)
    tukey_results2 = pairwise_tukeyhsd(dataframe2['Precentage'], dataframe2['Policy'])
    print(tukey_results2)

else:
    for i in range(20):
        Average_trust.append(trust_average_aware[i])
        Average_trust.append(trust_average_free[i])
        Precentage.append(precetage_aware[i])
        Precentage.append(precetage_free[i])

    dataframe1 = pd.DataFrame({'Participants': np.repeat([i for i in range(1, 21)], 2),
                            'Policy': np.tile([1, 2], 20), 
                            'Average_trust': Average_trust,
                            })
    
    print(AnovaRM(data=dataframe1, depvar='Average_trust',
                subject='Participants', within=['Policy']).fit())


    dataframe2 = pd.DataFrame({'Participants': np.repeat([i for i in range(1, 21)], 2),
                            'Policy': np.tile([1, 2], 20), 
                            'Precentage': Precentage,
                            })
    
    print(AnovaRM(data=dataframe2, depvar='Precentage',
                subject='Participants', within=['Policy']).fit())





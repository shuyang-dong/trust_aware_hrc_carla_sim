import numpy as np
from scipy import stats
import os
import pandas as pd

path_aware='/Users/piayu/Dropbox/Mac/Downloads/experiment_data/statistic_test/trust_aware/trust_aware_combined/'
path_free='/Users/piayu/Dropbox/Mac/Downloads/experiment_data/statistic_test/trust_free/trust_free_combined/'
path_aware_trial1='/Users/piayu/Dropbox/Mac/Downloads/experiment_data/statistic_test/trust_aware/trust_aware_trial1/'
path_free_trial1='/Users/piayu/Dropbox/Mac/Downloads/experiment_data/statistic_test/trust_free/trust_free_trial1/'
path_aware_trial2='/Users/piayu/Dropbox/Mac/Downloads/experiment_data/statistic_test/trust_aware/trust_aware_trial2/'
path_free_trial2='/Users/piayu/Dropbox/Mac/Downloads/experiment_data/statistic_test/trust_free/trust_free_trial2/'

#-------trust aware csv file loading---------------
dataframes_aware = []
file_list_aware =  os.listdir(path_aware_trial1)

for file in file_list_aware:
    if file.endswith('.csv'):
        file_path = os.path.join(path_aware_trial1, file)
        data = pd.read_csv(file_path)
        dataframes_aware.append(data)

#-------trust free csv file loading---------------
dataframes_free = []
file_list_free =  os.listdir(path_free_trial1)

for file in file_list_free:
    if file.endswith('.csv'):
        file_path = os.path.join(path_free_trial1, file)
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

#----------------------
print('Trust aware satisfaction precentage: %s' %precetage_aware)
print('Trust free  satisfaction precentage: %s' %precetage_free)
print('Trust aware average trust: %s' %trust_average_aware)
print('Trust free  average trust: %s' %trust_average_free)
print('Trust aware takeover probability: %s' %takeover_probability_aware)
print('Trust free  takeover probability: %s' %takeover_probability_free)

LDTL_satisfaction_aware = []
LDTL_satisfaction_free = []
for i in range(len(precetage_aware)):
    if precetage_aware[i] >= 0.99:
        LDTL_satisfaction_aware.append(1)
    else:
        LDTL_satisfaction_aware.append(0)

for i in range(len(precetage_free)):
    if precetage_free[i] >= 0.99:
        LDTL_satisfaction_free.append(1)
    else:
        LDTL_satisfaction_free.append(0)

#------------------
count_aware_precetage = 0
count_free_precentage = 0
for i in range(len(precetage_aware)):
    if precetage_aware[i]>precetage_free[i]:
        count_aware_precetage += 1
    if precetage_aware[i]<precetage_free[i]:
        count_free_precentage += 1

print('Number of times that trust aware has higher point-to-point satisfaction precentage: %s' %count_aware_precetage)
print('Number of times that trust free has higher point-to-point satisfaction precentage: %s' %count_free_precentage)

#------------------
count_aware_average_trust = 0
count_free_average_trust = 0
for i in range(len(trust_average_aware)):
    if trust_average_aware[i]>trust_average_free[i]:
        count_aware_average_trust += 1
    else:
        count_free_average_trust += 1

print('Number of times that trust aware has higher average trust: %s' %count_aware_average_trust)
print('Number of times that trust free has higher average trust: %s' %count_free_average_trust)

#------------------
count_aware_takeover = 0
count_free_takeover = 0
for i in range(len(takeover_probability_aware)):
    if takeover_probability_aware[i]<takeover_probability_free[i]:
        count_aware_takeover += 1
    else:
        count_free_takeover += 1

print('Number of times that trust aware has lower takeover probability: %s' %count_aware_takeover)
print('Number of times that trust free has lower takeover probability: %s' %count_free_takeover)

#-------paired t-test---------------
t_statistic, p_value = stats.ttest_rel(precetage_aware, precetage_free)
print("T-statistic_point-to-point_precentage: %s" %t_statistic)
print("p-value_point-to-point_precentage: %s" %p_value)

t_statistic1, p_value1 = stats.ttest_rel(trust_average_aware, trust_average_free)
print("T-statistic_average_trust: %s" %t_statistic1)
print("p-value_average_trust: %s" %p_value1)

t_statistic2, p_value2 = stats.ttest_rel(takeover_probability_free, takeover_probability_aware)
print("T-statistic_takeover_probability: %s" %t_statistic2)
print("p-value_takeover_probabiity: %s" %p_value2)

t_statistic3, p_value3 = stats.ttest_rel(LDTL_satisfaction_aware, LDTL_satisfaction_free)
print("T-statistic_satisfaction: %s" %t_statistic3)
print("p-value_satisfaction: %s" %p_value3)

# # Interpret the results
# alpha = 0.05  # Significance level
# if p_value < alpha:
#     print("Reject the null hypothesis. There is a significant difference.")
# else:
#     print("Fail to reject the null hypothesis. No significant difference.")



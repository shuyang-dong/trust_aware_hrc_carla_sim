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


#-------trust aware data post-processing---------------
num_csv_file = len(dataframes_aware)
segment_satisfy_precentage = [] 
travel_road_segment = []
average_trust = [[] for _ in range(num_csv_file)]
max_trust = [[] for _ in range(num_csv_file)]
min_trust = [[] for _ in range(num_csv_file)]
intersection_trust = [[] for _ in range(num_csv_file)]
p2p_precentage = [[] for _ in range(num_csv_file)]

for i in range(num_csv_file):
    data = dataframes_aware[i]
    num_rows = data.shape[0]
    num_segment = 0
    road_segment_last = []
    road_segment_direction_last = []
    num_total = []
    trust = []
    trust_min = []
    trust_max = []
    trust_intersection = []
    num_satisfy = []
    num_notsatisfy = []
    seg_num_satisfy = []
    seg_num_notsatisfy = []
    position = data.iloc[1, 2]
    position_split = position.split(",")
    road_segment_last.append(int(position_split[-2].strip()))
    num_total.append(1)
    trust.append(data.iloc[1, 1])
    trust_min.append(data.iloc[1, 1])
    trust_max.append(data.iloc[1, 1])
    trust_intersection.append(data.iloc[1, 1])
    if data.iloc[1, 1] >= 2:
        num_satisfy.append(1)
        num_notsatisfy.append(0)
    else:
        num_satisfy.append(0)
        num_notsatisfy.append(1)
    num_manual = 0
    index = []
    index.append(1)
    for k in range(2, num_rows):
        if str(data.iloc[k, 3]) == 'True':
            position_current = data.iloc[k, 2]
            position_current_split = position_current.split(",")
            road_segment_current = int(position_current_split[-2].strip())
            if road_segment_last[num_segment] == road_segment_current: #on the same road 
                num_total[num_segment] += 1
                trust[num_segment] += data.iloc[k, 1]
                if data.iloc[k, 1] > trust_max[num_segment]:
                    trust_max[num_segment] = data.iloc[k, 1]
                if data.iloc[k, 1] < trust_min[num_segment]:
                    trust_min[num_segment] = data.iloc[k, 1]
                if data.iloc[k, 1] >= 2:
                    num_satisfy[num_segment] += 1
                else:
                    num_notsatisfy[num_segment] += 1                
            else: 
                index.append(k)
                num_segment += 1
                position_update = data.iloc[k, 2]
                position_update_split =  position_update.split(",")
                road_segment_last.append(int(position_update_split[-2].strip()))
                num_total.append(1)
                trust.append(data.iloc[k, 1])
                trust_min.append(data.iloc[k, 1])
                trust_max.append(data.iloc[k, 1])
                trust_intersection.append(data.iloc[k-1, 1])
                if data.iloc[k, 1] >= 2:
                    num_satisfy.append(1)
                    num_notsatisfy.append(0)
                else:
                    num_satisfy.append(0)
                    num_notsatisfy.append(1)               
        else:
            num_manual += 1
    
    for kk in range(num_segment):
        if num_total[kk]>0:
            average_trust[i].append(trust[kk]/num_total[kk])
            min_trust[i].append(trust_min[kk])
            max_trust[i].append(trust_max[kk])
            intersection_trust[i].append(trust_intersection[kk])
            p2p_precentage[i].append(num_satisfy[kk]/num_total[kk])

    # determine segment satisfaction based on minimum trust
    segment_satisfy = 0
    segment_notsatisfy = 0  
    segment_satisfy_high = 0
    segment_notsatisfy_high = 0 
    for kk in range(num_segment):
        if min_trust[i][kk] >= 2:
            segment_satisfy += 1
        else:
            segment_notsatisfy += 1
        if max_trust[i][kk] >= 6:
            segment_satisfy_high += 1
        else:
            segment_notsatisfy_high += 1

    segment_satisfy_precentage.append(segment_satisfy/num_segment)
    travel_road_segment.append(num_segment)


# print('Average trust for each road segment (trust aware): %s' %average_trust)
# print('Maximim trust for each road segment (trust aware): %s' %max_trust)
# print('Minimum trust for each road segment (trust aware): %s' %min_trust)
# print('Intersection trust for each road segment (trust aware): %s' %intersection_trust)
# print('Point to point satisfaction precentage for each road segment (trust aware): %s' %p2p_precentage)    
        
average_segment_trust = []
max_average_segment_trust = []
min_average_segment_trust = []
intersection_average_segment_trust = []
average_segment_precentage = []

for i in range(num_csv_file):
    total_segment = len(average_trust[i])
    trust_segment = 0
    max_trust_segment = 0
    min_trust_segment = 0
    intersection_trust_segment = 0
    precent = 0
    for k in range(total_segment):
        trust_segment += average_trust[i][k]
        max_trust_segment += max_trust[i][k]
        min_trust_segment += min_trust[i][k]
        intersection_trust_segment += intersection_trust[i][k]
        precent += p2p_precentage[i][k]

    average_segment_trust.append(trust_segment/total_segment)
    max_average_segment_trust.append(max_trust_segment/total_segment)
    min_average_segment_trust.append(min_trust_segment/total_segment)
    intersection_average_segment_trust.append(intersection_trust_segment/total_segment)
    average_segment_precentage.append(precent/total_segment)

print('Average segment trust for complete path (trust aware): %s' %average_segment_trust)
print('Maximum average segment trust for complete path (trust aware): %s' %max_average_segment_trust)
print('Minimum average segment trust for complete path (trust aware): %s' %min_average_segment_trust)
print('Intersection average segment trust for complete path (trust aware): %s' %intersection_average_segment_trust)
print('Segment satisfaction precentage for complete path base on average trust (trust aware): %s' %average_segment_precentage)  
print('Segment satisfaction precentage for complete path base on minimum trust (trust aware): %s' %segment_satisfy_precentage)
print('Number of road segments travelled for each human (trust aware): %s' %travel_road_segment) 

#-------trust free data post-processing---------------
num_csv_file_free = len(dataframes_free)
segment_satisfy_precentage_free = [] 
travel_road_segment_free = []
average_trust_free = [[] for _ in range(num_csv_file_free)]
max_trust_free = [[] for _ in range(num_csv_file_free)]
min_trust_free = [[] for _ in range(num_csv_file_free)]
intersection_trust_free = [[] for _ in range(num_csv_file_free)]
p2p_precentage_free = [[] for _ in range(num_csv_file_free)]

for i in range(num_csv_file_free):
    data_free = dataframes_free[i]
    num_rows_free = data_free.shape[0]
    num_segment_free = 0
    road_segment_last_free = []
    road_segment_direction_last_free = []
    num_total_free = []
    trust_free = []
    trust_min_free = []
    trust_max_free = []
    trust_intersection_free = []
    num_satisfy_free = []
    num_notsatisfy_free = []
    seg_num_satisfy_free = []
    seg_num_notsatisfy_free = []
    position_free = data_free.iloc[1, 2]
    position_split_free = position_free.split(",")
    road_segment_last_free.append(int(position_split_free[-2].strip()))
    num_total_free.append(1)
    trust_free.append(data_free.iloc[1, 1])
    trust_min_free.append(data_free.iloc[1, 1])
    trust_max_free.append(data_free.iloc[1, 1])
    trust_intersection_free.append(data_free.iloc[1, 1])
    if data_free.iloc[1, 1] >= 2:
        num_satisfy_free.append(1)
        num_notsatisfy_free.append(0)
    else:
        num_satisfy_free.append(0)
        num_notsatisfy_free.append(1)
    num_manual_free = 0
    index_free = []
    index_free.append(1)
    for k in range(2, num_rows_free):
        if str(data_free.iloc[k, 3]) == 'True':
            position_current_free = data_free.iloc[k, 2]
            position_current_split_free = position_current_free.split(",")
            road_segment_current_free = int(position_current_split_free[-2].strip())
            if road_segment_last_free[num_segment_free] == road_segment_current_free:
                num_total_free[num_segment_free] += 1
                trust_free[num_segment_free] += data_free.iloc[k, 1]
                if data_free.iloc[k, 1] > trust_max_free[num_segment_free]:
                    trust_max_free[num_segment_free] = data_free.iloc[k, 1]
                if data_free.iloc[k, 1] < trust_min_free[num_segment_free]:
                    trust_min_free[num_segment_free] = data_free.iloc[k, 1]
                if data_free.iloc[k, 1] >= 2:
                    num_satisfy_free[num_segment_free] += 1
                else:
                    num_notsatisfy_free[num_segment_free] += 1                
            else: 
                index_free.append(k)
                num_segment_free += 1
                position_update_free = data_free.iloc[k, 2]
                position_update_split_free =  position_update_free.split(",")
                road_segment_last_free.append(int(position_update_split_free[-2].strip()))
                num_total_free.append(1)
                trust_free.append(data_free.iloc[k, 1])
                trust_min_free.append(data_free.iloc[k, 1])
                trust_max_free.append(data_free.iloc[k, 1])
                trust_intersection_free.append(data_free.iloc[k-1, 1])
                if data_free.iloc[k, 1] >= 2:
                    num_satisfy_free.append(1)
                    num_notsatisfy_free.append(0)
                else:
                    num_satisfy_free.append(0)
                    num_notsatisfy_free.append(1)               
        else:
            num_manual_free += 1
    
    for kk in range(num_segment_free):
        if num_total_free[kk]>0:
            average_trust_free[i].append(trust_free[kk]/num_total_free[kk])
            min_trust_free[i].append(trust_min_free[kk])
            max_trust_free[i].append(trust_max_free[kk])
            intersection_trust_free[i].append(trust_intersection_free[kk])
            p2p_precentage_free[i].append(num_satisfy_free[kk]/num_total_free[kk])

    # determine segment satisfaction based on minimum trust
    segment_satisfy_free = 0
    segment_notsatisfy_free = 0  
    segment_satisfy_high_free = 0
    segment_notsatisfy_high_free = 0 
    for kk in range(num_segment_free):
        if min_trust_free[i][kk] >= 2:
            segment_satisfy_free += 1
        else:
            segment_notsatisfy_free += 1
        if max_trust_free[i][kk] >= 6:
            segment_satisfy_high_free += 1
        else:
            segment_notsatisfy_high_free += 1

    segment_satisfy_precentage_free.append(segment_satisfy_free/num_segment_free)
    travel_road_segment_free.append(num_segment_free)

# print('Average trust for each road segment (trust free): %s' %average_trust_free)
# print('Maximim trust for each road segment (trust free): %s' %max_trust_free)
# print('Minimum trust for each road segment (trust free): %s' %min_trust_free)
# print('Intersection trust for each road segment (trust free): %s' %intersection_trust_free)
# print('Point to point satisfaction precentage for each road segment (trust free): %s' %p2p_precentage_free)    
        
average_segment_trust_free = []
max_average_segment_trust_free = []
min_average_segment_trust_free = []
intersection_average_segment_trust_free = []
average_segment_precentage_free = []

for i in range(num_csv_file_free):
    total_segment_free = len(average_trust_free[i])
    trust_segment_free = 0
    max_trust_segment_free = 0
    min_trust_segment_free = 0
    intersection_trust_segment_free = 0
    precent_free = 0
    for k in range(total_segment_free):
        trust_segment_free += average_trust_free[i][k]
        max_trust_segment_free += max_trust_free[i][k]
        min_trust_segment_free += min_trust_free[i][k]
        intersection_trust_segment_free += intersection_trust_free[i][k]
        precent_free += p2p_precentage_free[i][k]

    average_segment_trust_free.append(trust_segment_free/total_segment_free)
    max_average_segment_trust_free.append(max_trust_segment_free/total_segment_free)
    min_average_segment_trust_free.append(min_trust_segment_free/total_segment_free)
    intersection_average_segment_trust_free.append(intersection_trust_segment_free/total_segment_free)
    average_segment_precentage_free.append(precent_free/total_segment_free)

print('Average segment trust for complete path (trust free): %s' %average_segment_trust_free)
print('Maximum average segment trust for complete path (trust free): %s' %max_average_segment_trust_free)
print('Minimum average segment trust for complete path (trust free): %s' %min_average_segment_trust_free)
print('Intersection average segment trust for complete path (trust free): %s' %intersection_average_segment_trust_free)
print('Segment satisfaction precentage for complete path base on average trust (trust free): %s' %average_segment_precentage_free)  
print('Segment satisfaction precentage for complete path base on minimum trust (trust free): %s' %segment_satisfy_precentage_free) 
print('Number of road segments travelled for each human (trust free): %s' %travel_road_segment_free)


#-------paired t-test---------------
t_statistic, p_value = stats.ttest_rel(average_segment_trust, average_segment_trust_free)
print("T-statistic_average_segment_trust: %s" %t_statistic)
print("p-value_average_segment_trust: %s" %p_value)

t_statistic1, p_value1 = stats.ttest_rel(average_segment_precentage, average_segment_precentage_free)
print("T-statistic_average_segment_precentage: %s" %t_statistic1)
print("p-value_average_segment_precentage: %s" %p_value1)

t_statistic2, p_value2 = stats.ttest_rel(segment_satisfy_precentage, segment_satisfy_precentage_free)
print("T-statistic_segment_precentage_minimum: %s" %t_statistic2)
print("p-value_segment_precentage_minimum: %s" %p_value2)

t_statistic3, p_value3 = stats.ttest_rel(min_average_segment_trust, min_average_segment_trust_free)
print("T-statistic_minumum_segment_trust: %s" %t_statistic3)
print("p-value_minimum_segment_trust: %s" %p_value3)

t_statistic4, p_value4 = stats.ttest_rel(max_average_segment_trust, max_average_segment_trust_free)
print("T-statistic_maximum_segment_trust: %s" %t_statistic4)
print("p-value_maximum_segment_trust: %s" %p_value4)

t_statistic5, p_value5 = stats.ttest_rel(intersection_average_segment_trust, intersection_average_segment_trust_free)
print("T-statistic_intersection_segment_trust: %s" %t_statistic5)
print("p-value_intersection_segment_trust: %s" %p_value5)



# # Interpret the results
# alpha = 0.05  # Significance level
# if p_value < alpha:
#     print("Reject the null hypothesis. There is a significant difference.")
# else:
#     print("Fail to reject the null hypothesis. No significant difference.")



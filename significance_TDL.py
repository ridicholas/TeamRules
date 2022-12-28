import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from sklearn import metrics
from scipy.stats import ttest_ind
from statistics import mean, stdev
import math
from scipy.stats import ttest_rel

numRuns = 10 #adjust this depending on how many runs of results were produced

path = 'gaussian_contradiction_results/'

asym_loss = [1,1]
data = path.split('_')[0]
costs = [0]



tr_conf = 0.5
hyrs_conf = 0

teams = ['team1', 'team2', 'team3']

team_infos = []
datasets = []
tr_results_filtered = {}
hyrs_results_filtered = {}
brs_results = {'team1': [], 'team2': [], 'team3': []}

for team in teams:
    
    tr_results_filtered[team] = {}
    hyrs_results_filtered[team] = {}

for team in teams:
    for cost in costs:
        tr_results_filtered[team][str(cost)] = []
        hyrs_results_filtered[team][str(cost)] = []
        

start_info = pd.read_pickle(path + 'start_info.pkl')
for i in range(0, numRuns):
    datasets.append(pd.read_pickle(path + 'dataset_run{}.pkl'.format(i)))
    for team in teams:
        for cost in costs:
            
            cost = str(cost)
            #tr filtered
            tr_results_filtered[team][cost].append(pd.read_pickle(path + 'cost_{}_'.format(cost) + team + '_tr_filtered_run{}.pkl'.format(i)).sort_values(by='test_error'))
            tr_results_filtered[team][cost][-1] = tr_results_filtered[team][cost][-1][tr_results_filtered[team][cost][-1]['mental_conf'] == tr_conf].reset_index()
            tr_results_filtered[team][cost]

            #hyrs filtered
            hyrs_results_filtered[team][cost].append(pd.read_pickle(path + 'cost_{}_'.format(cost)+ team + '_hyrs_filtered_run{}.pkl'.format(i)).sort_values(by='test_error'))
            hyrs_results_filtered[team][cost][-1] = hyrs_results_filtered[team][cost][-1][hyrs_results_filtered[team][cost][-1]['mental_conf'] == hyrs_conf].reset_index()
        
            if cost == '0':
                brs_results[team].append(pd.read_pickle(path + team + '_brs_run{}.pkl'.format(i)).sort_values(by='test_error_brs'))

settings = ['Rational', 'Neutral', 'Irrational']

teams = ['team1', 'team2', 'team3']

team_names = ['Rational', 'Neutral', 'Irrational']
one_side = pd.DataFrame(columns = ['HyRS', 'BRS'], index = ['Rational', 'Neutral', 'Irrational'])
means = pd.DataFrame(columns = ['Human', 'TeamRules', 'HyRS', 'BRS'], index = ['Rational', 'Neutral', 'Irrational'])
i = 0
for team in teams: 
    TeamRules = []
    BRS = []
    HyRS = []
    Human = []
    for run in range(numRuns):
        Human.append(1-metrics.accuracy_score(datasets[run][team+'_Ytest'], datasets[run][team+'_Ybtest']))
        TeamRules.append(tr_results_filtered[team]['0'][run].loc[0,'test_error'])
        BRS.append(brs_results[team][run].loc[0,'test_error_brs'])
        HyRS.append(hyrs_results_filtered[team]['0'][run].loc[0,'test_error'])
    frame = pd.DataFrame({'TeamRules': TeamRules,
                          'BRS': BRS,
                         'HyRS': HyRS})
    means.loc[team_names[i], 'TeamRules'] = mean(TeamRules)
    means.loc[team_names[i], 'HyRS'] = mean(HyRS)
    means.loc[team_names[i], 'BRS'] = mean(BRS)
    means.loc[team_names[i], 'Human'] = mean(Human)

    one_side.iloc[i, 0] = ttest_rel(frame.TeamRules, frame.HyRS,  alternative='greater')[1]
    one_side.iloc[i, 1] = ttest_rel(frame.TeamRules, frame.BRS,  alternative='greater')[1]
    i+=1

print(means)
print(one_side)



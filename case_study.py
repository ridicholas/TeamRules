import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

folder = '/Users/nicholaswolczynski/Documents/UT Austin/Research Projects/TeamRules/heart_contradiction_results_temp_learned_len4/scenarios/'

with open(f'{folder}/team1.pkl', 'rb') as inp:
    team1 = pickle.load(inp)
inp.close()

with open(f'{folder}/team2.pkl', 'rb') as inp:
    team2 = pickle.load(inp)
inp.close()

with open(f'{folder}/team3.pkl', 'rb') as inp:
    team3 = pickle.load(inp)
inp.close()

with open(f'{folder}/team4.pkl', 'rb') as inp:
    team4 = pickle.load(inp)
inp.close()

teams = [team1,team2,team3,team4]
team_strs = ['team1','team2','team3','team4']

def make_results_tables(method):

    

    opt_dex = {}
    corrects = {}
    accepteds = {}
    covereds = {}
    contradicts = {}
    correct_covereds = {}
    correct_contradicts = {}
    correct_accepteds = {}
    accepted_covereds = {}
    accepted_contradicts = {}
    human_corrects = {}
    human_incorrects = {}
    modelonly_corrects = {}
    team_corrects = {}
    totals = {}
    datasets = {}

    for i in range(len(teams)):
        methods = {'tr': {'train': teams[i].full_tr_results_train, 'val': teams[i].full_tr_results_val, 'test': teams[i].full_tr_results} , 
               'hyrs': {'train': teams[i].full_hyrs_results_train, 'val': teams[i].full_hyrs_results_val, 'test': teams[i].full_hyrs_results}}
        if method == 'tr':
            opt_dex[team_strs[i]] = methods[method]['val'].loc[:, 'objective'].astype(float).idxmin()
        else:
            opt_dex[team_strs[i]] = 0
        corrects[team_strs[i]] = teams[i].data_model_dict['Ytest'] == methods[method]['test'].iloc[opt_dex[team_strs[i]]]['modelonly_test_preds']
        accepteds[team_strs[i]] = methods[method]['test'].iloc[opt_dex[team_strs[i]]]['humanified_test_preds'] == methods[method]['test'].iloc[opt_dex[team_strs[i]]]['modelonly_test_preds']
        covereds[team_strs[i]] = methods[method]['test'].iloc[opt_dex[team_strs[i]]]['test_covereds']
        contradicts[team_strs[i]] = methods[method]['test'].iloc[opt_dex[team_strs[i]]]['modelonly_test_preds'] != teams[i].data_model_dict['Ybtest']
        correct_covereds[team_strs[i]] = corrects[team_strs[i]] & covereds[team_strs[i]]
        correct_contradicts[team_strs[i]] = corrects[team_strs[i]] & contradicts[team_strs[i]]
        correct_accepteds[team_strs[i]] = corrects[team_strs[i]] & accepteds[team_strs[i]]
        accepted_covereds[team_strs[i]] = covereds[team_strs[i]] & accepteds[team_strs[i]]
        accepted_contradicts[team_strs[i]] = contradicts[team_strs[i]] & accepteds[team_strs[i]]
        human_corrects[team_strs[i]] = (teams[i].data_model_dict['Ytest'] == teams[i].data_model_dict['Ybtest']).astype(bool)
        human_incorrects[team_strs[i]] = (teams[i].data_model_dict['Ytest'] != teams[i].data_model_dict['Ybtest']).astype(bool)
        modelonly_corrects[team_strs[i]] = methods[method]['test'].iloc[opt_dex[team_strs[i]]]['modelonly_test_preds'] == teams[i].data_model_dict['Ytest']
        team_corrects[team_strs[i]] = methods[method]['test'].iloc[opt_dex[team_strs[i]]]['humanified_test_preds'] == teams[i].data_model_dict['Ytest']
        totals[team_strs[i]] = human_corrects[team_strs[i]] + human_incorrects[team_strs[i]]

        datasets[team_strs[i]] = pd.DataFrame({'Human Correct': human_corrects[team_strs[i]],
                                'Human Incorrect': human_incorrects[team_strs[i]],
                        'Elderly Male Population': ((teams[i].data_model_dict['Xtest']['age54.0'] == 1) & (teams[i].data_model_dict['Xtest']['sex_Male'] == 1)).astype(bool),
                        'CorrectCovereds': correct_covereds[team_strs[i]], 
                        'Accepted Covereds': accepted_covereds[team_strs[i]],
                        'Accepted Contradicts': accepted_contradicts[team_strs[i]],
                        'Covereds': covereds[team_strs[i]],
                        'Correct Covereds': correct_covereds[team_strs[i]],
                        'Correct Contradictions': correct_contradicts[team_strs[i]],
                            'Contradicts': contradicts[team_strs[i]], 
                            'Team Correct': team_corrects[team_strs[i]],
                            'Model Correct': modelonly_corrects[team_strs[i]] & covereds[team_strs[i]],
                            'Total': totals[team_strs[i]]})
        
    return datasets
    
def make_rates(dataset):

    groupedSums = dataset.groupby('Elderly Male Population').sum()
    rates = pd.DataFrame({'Coverage Rate': groupedSums['Covereds']/groupedSums['Total'],
                                'Contradiction Rate': groupedSums['Contradicts']/groupedSums['Total'],
                                'Contradiction Accuracy': groupedSums['Correct Contradictions']/groupedSums['Contradicts'],
                                'Acceptance Rate': groupedSums['Accepted Contradicts']/groupedSums['Contradicts'],
                                'Human Accuracy': groupedSums['Human Correct']/groupedSums['Total'],
                                'Model Accuracy' :groupedSums['Model Correct']/groupedSums['Covereds'],
                                'Team Accuracy': groupedSums['Team Correct']/groupedSums['Total']})

    return rates
                            





tr_datasets = make_results_tables('tr')
hyrs_datasets = make_results_tables('hyrs')

make_rates(tr_datasets['team1'])

import seaborn as sns

def normalize(x): 
    return (x-np.min(x))/(np.max(x)-np.min(x))

plt.scatter(normalize(curr_team.data_model_dict['test_conf'])[corrects[curr_team_str] & covereds[curr_team_str] ], 
            pd.Series(curr_team.data_model_dict['Ytest'][corrects[curr_team_str] & covereds[curr_team_str] ] == curr_team.data_model_dict['Ybtest'][corrects[curr_team_str] & covereds[curr_team_str] ]).replace({True: 'Human Correct', False: 'Human Incorrect'}), color='green', marker="*", s=100)

plt.scatter(normalize(curr_team.data_model_dict['test_conf'])[~corrects[curr_team_str] & covereds[curr_team_str] ], 
            pd.Series(curr_team.data_model_dict['Ytest'][~corrects[curr_team_str] & covereds[curr_team_str] ] == curr_team.data_model_dict['Ybtest'][~corrects[curr_team_str] & covereds[curr_team_str] ]).replace({True: 'Human Correct', False: 'Human Incorrect'}), color='red', marker="x", s=100)

plt.scatter(normalize(curr_team.data_model_dict['test_conf'])[~covereds[curr_team_str] ], 
            pd.Series(curr_team.data_model_dict['Ytest'][~covereds[curr_team_str] ] == curr_team.data_model_dict['Ybtest'][~covereds[curr_team_str] ]).replace({True: 'Human Correct', False: 'Human Incorrect'}), color='black', marker=".", alpha=0.5)
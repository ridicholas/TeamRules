import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from sklearn import metrics
from scipy.stats import ttest_ind
from statistics import mean, stdev
import math

numRuns = 5 #adjust this depending on how many runs of results were produced

#read in results
path = 'fico_discretion_resultsMONTY/'
data = path.split('_')[0]
discErrors = [0.01, 0.05, 0.25, 0.5, 0.8, 1]
cost = 0.3


teams = ['team1', 'team2', 'team3']
team_infos = {}
datasets = []
tr_results_filtered = {}
hyrs_results_filtered = {}
brs_results = {'team1': [], 'team2': [], 'team3': []}

for run in range(numRuns):
    team_infos[run] = {}

    for discError in discErrors:
        team_infos[run][discError] = pd.read_pickle(path + 'team_info_dataused{}_run{}.pkl'.format(discError, run))


for team in teams:
    
    tr_results_filtered[team] = {}
    hyrs_results_filtered[team] = {}

for team in teams:
    for discError in discErrors:
        tr_results_filtered[team][str(discError)] = []
        hyrs_results_filtered[team][str(discError)] = []

        

start_info = pd.read_pickle(path + 'start_info.pkl')
for i in range(0, numRuns):
    datasets.append(pd.read_pickle(path + 'discError_{}_'.format(discError)+ 'dataset_run{}.pkl'.format(i)))
    for team in teams:
        
        for discError in discErrors:
            
            discError = str(discError)
            
            #tr filtered
            tr_results_filtered[team][discError].append(pd.read_pickle(path + 'discError_{}_'.format(discError) + team + '_tr_filtered_run{}.pkl'.format(i)).sort_values(by='test_error'))
            tr_results_filtered[team][discError][-1] = tr_results_filtered[team][discError][-1][tr_results_filtered[team][discError][-1]['error_conf'] == 0]
            tr_results_filtered[team][discError]

            #hyrs filtered
            hyrs_results_filtered[team][discError].append(pd.read_pickle(path + 'discError_{}_'.format(discError)+ team + '_hyrs_filtered_run{}.pkl'.format(i)).sort_values(by='test_error'))
            hyrs_results_filtered[team][discError][-1] = hyrs_results_filtered[team][discError][-1][hyrs_results_filtered[team][discError][-1]['error_conf'] == 0]
        
            if discError == '0':
                brs_results[team].append(pd.read_pickle(path + team + '_brs_run{}.pkl'.format(i)).sort_values(by='test_error_brs'))

for run in range(numRuns):
    for team in teams:
        for discError in discErrors:
            discError = str(discError)
            if discError == '0':
                brs_results[team][run].loc[0,'error_rate_acceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][datasets[run][team+'TestConf'] < 0.5], brs_results[team][run].loc[0,'team_test_preds'][datasets[run][team+'TestConf'] < 0.5])
            
            
                
                
            
            tr_results_filtered[team][discError][run].loc[0,'error_rate_acceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][datasets[run][team+'TestConf'] < 0.5], tr_results_filtered[team][discError][run].loc[0,'humanified_test_preds'][datasets[run][team+'TestConf'] < 0.5])
            
            
            tr_results_filtered[team][discError][run].loc[0,'error_rate_ModelOnlyAcceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][(datasets[run][team+'TestConf'] < 0.5) & (tr_results_filtered[team][discError][run].loc[0,'test_covereds'])], 
                                                                                                                    tr_results_filtered[team][discError][run].loc[0,'modelonly_test_preds'][(datasets[run][team+'TestConf'] < 0.5) & (tr_results_filtered[team][discError][run].loc[0,'test_covereds'])])

            tr_results_filtered[team][discError][run].loc[0, 'contradictions'] = (tr_results_filtered[team][discError][run].loc[0, 'modelonly_test_preds'] != datasets[run][team+'_Ybtest']).sum()
            hyrs_results_filtered[team][discError][run].loc[0,'error_rate_acceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][datasets[run][team+'TestConf'] < 0.5], hyrs_results_filtered[team][discError][run].loc[0,'humanified_test_preds'][datasets[run][team+'TestConf'] < 0.5])
            hyrs_results_filtered[team][discError][run].loc[0,'error_rate_ModelOnlyAcceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][(datasets[run][team+'TestConf'] < 0.5) & (hyrs_results_filtered[team][discError][run].loc[0,'test_covereds'])], 
                                                                                                                    hyrs_results_filtered[team][discError][run].loc[0,'modelonly_test_preds'][(datasets[run][team+'TestConf'] < 0.5) & (hyrs_results_filtered[team][discError][run].loc[0,'test_covereds'])])
            hyrs_results_filtered[team][discError][run].loc[0, 'contradictions'] = (hyrs_results_filtered[team][discError][run].loc[0, 'modelonly_test_preds'] != datasets[run][team+'_Ybtest']).sum()
            tr_results_filtered[team][discError][run].loc[0,'total_acceptRegion'] = sum(datasets[run][team+'TestConf'] < 0.5)

            hyrs_results_filtered[team][discError][run].loc[0,'total_acceptRegion'] = sum(datasets[run][team+'TestConf'] < 0.5)


teams = []
for t in start_info.sort_values(by='human accept region train acc').index:
    teams.append('team{}'.format(t))

settings = ['Rational', 'Neutral', 'Irrational']

for whichTeam in range(len(settings)):
    plt.clf()
    team = teams[whichTeam]
    setting = settings[whichTeam]
    discErrorFrame = pd.DataFrame(index=[str(x) for x in discErrors], data={'discErrors': [str(x) for x in discErrors], 
                                  'TeamRulesTeamLoss': np.zeros(len(discErrors)), 
                                  'HyRSTeamLoss': np.zeros(len(discErrors)),
                                  'TeamRulesCov': np.zeros(len(discErrors)),
                                  'HyRSCov': np.zeros(len(discErrors)),
                                  'TeamRulesModelOnlyAcceptLoss': np.zeros(len(discErrors)), 
                                  'HyRSModelOnlyAcceptLoss': np.zeros(len(discErrors)),
                                  'TeamRulesRejects': np.zeros(len(discErrors)), 
                                  'HyRSRejects': np.zeros(len(discErrors)), 
                                  'TR_Contradictions': np.zeros(len(discErrors)), 
                                  'HyRS_Contradictions': np.zeros(len(discErrors)),
                                  'TR_Objectives': np.zeros(len(discErrors)),
                                  'HyRS_Objectives': np.zeros(len(discErrors)),})

    for discError in discErrors:

        discError = str(discError)

        TeamRulesLoss = []
        HyRSLoss = []
        TeamRules_modelonly_Loss = []
        HyRS_modelonly_Loss = []
        TeamRulesCov = []
        HyRSCov = []
        TeamRulesRejects = []
        HyRSRejects = []
        TRContradicts = []
        HyRSContradicts = []
        TR_Objectives = []
        HyRS_Objectives = []
        DiscretionErrors = []
        for run in range(numRuns):

            

            TeamRulesLoss.append(tr_results_filtered[team][discError][run].loc[0,'test_error'])
            DiscretionErrors.append(1-team_infos[run][float(discError)].loc[whichTeam+1, 'aversion model test acc'])
            TeamRules_modelonly_Loss.append(tr_results_filtered[team][discError][run].loc[0,'error_rate_ModelOnlyAcceptRegion'])
            TeamRulesRejects.append(tr_results_filtered[team][discError][run].loc[0,'test_rejects'])
            HyRSLoss.append(hyrs_results_filtered[team][discError][run].loc[0,'test_error'])
            HyRS_modelonly_Loss.append(hyrs_results_filtered[team][discError][run].loc[0,'error_rate_ModelOnlyAcceptRegion'])
            HyRSRejects.append(hyrs_results_filtered[team][discError][run].loc[0,'test_rejects'])
            TeamRulesCov.append(tr_results_filtered[team][discError][run].loc[0,'test_coverage'])
            HyRSCov.append(hyrs_results_filtered[team][discError][run].loc[0,'test_coverage'])
            TRContradicts.append(tr_results_filtered[team][discError][run].loc[0,'contradictions'])
            HyRSContradicts.append(hyrs_results_filtered[team][discError][run].loc[0,'contradictions'])
            TR_Objectives.append((TeamRulesLoss[-1]) + (float(cost)*TRContradicts[-1])/(datasets[run].shape[0]))
            HyRS_Objectives.append((HyRSLoss[-1]) + (float(cost)*HyRSContradicts[-1])/(datasets[run].shape[0]))

        

        frame = pd.DataFrame({'TeamRulesLoss': TeamRulesLoss,
                             'HyRSLoss': HyRSLoss,
                             'TeamRulesCov': TeamRulesCov,
                             'HyRSCov': HyRSCov,
                             'TRRej': TeamRulesRejects,
                             'HyRSRej': HyRSRejects})
        discErrorFrame.loc[discError, 'TeamRulesTeamLoss'] = mean(TeamRulesLoss)
        discErrorFrame.loc[discError, 'TeamRulesTeamLoss_std'] = stdev(TeamRulesLoss)
        discErrorFrame.loc[discError, 'HyRSTeamLoss'] = mean(HyRSLoss)
        discErrorFrame.loc[discError, 'HyRSTeamLoss_std'] = stdev(HyRSLoss)
        discErrorFrame.loc[discError, 'TeamRulesCov'] = mean(TeamRulesCov)
        discErrorFrame.loc[discError, 'TeamRulesRejects'] = mean(TeamRulesRejects)
        discErrorFrame.loc[discError, 'HyRSRejects'] = mean(HyRSRejects)
        discErrorFrame.loc[discError, 'HyRSCov'] = mean(HyRSCov)
        discErrorFrame.loc[discError, 'TeamRulesModelOnlyAcceptLoss'] = mean(TeamRules_modelonly_Loss)
        discErrorFrame.loc[discError, 'HyRSModelOnlyAcceptLoss'] = mean(HyRS_modelonly_Loss)
        discErrorFrame.loc[discError, 'TR_Contradictions'] = mean(TRContradicts)
        discErrorFrame.loc[discError, 'TR_Contradictions_std'] = stdev(TRContradicts)
        discErrorFrame.loc[discError, 'HyRS_Contradictions'] = mean(HyRSContradicts)
        discErrorFrame.loc[discError, 'HyRS_Contradictions_std'] = stdev(HyRSContradicts)
        discErrorFrame.loc[discError, 'TR_Objective'] = mean(TR_Objectives)
        discErrorFrame.loc[discError, 'HyRS_Objective'] = mean(HyRS_Objectives)
        discErrorFrame.loc[discError, 'TR_Objective_SE'] = stdev(TR_Objectives)/math.sqrt(numRuns)
        discErrorFrame.loc[discError, 'HyRS_Objective_SE'] = stdev(HyRS_Objectives)/math.sqrt(numRuns)
        discErrorFrame.loc[discError, 'Discretion Error'] = mean(DiscretionErrors)

    

    
    
    discErrorFrame.sort_values(by=['discErrors'], inplace=True)
    plt.plot(discErrorFrame['Discretion Error'], discErrorFrame['HyRS_Objective'], c='red',  label = 'c-HyRS', markersize=6)
    plt.plot(discErrorFrame['Discretion Error'], discErrorFrame['TR_Objective'], c='blue', label='TeamRules', markersize=6)
    plt.fill_between(discErrorFrame['Discretion Error'], 
                discErrorFrame['HyRS_Objective']-(discErrorFrame['HyRS_Objective_SE']),
                discErrorFrame['HyRS_Objective']+(discErrorFrame['HyRS_Objective_SE']) ,
                color='red', alpha=0.7, linewidth=1)
    plt.fill_between(discErrorFrame['Discretion Error'], 
                discErrorFrame['TR_Objective']-(discErrorFrame['TR_Objective_SE']),
                discErrorFrame['TR_Objective']+(discErrorFrame['TR_Objective_SE']) ,
                color='blue', alpha=0.2, linewidth=1)
    plt.xlabel('Mean Discretion Error', fontsize=14)
    plt.ylabel('Team Loss', fontsize=14)
    plt.tick_params(labelrotation=45, labelsize=10)
    #row.set_title('{} Setting'.format(setting), fontsize=15)
    plt.legend(prop={'size': 6})
    plt.savefig('Plots/discretionMONTY_{}_{}_cost{}.png'.format(data,setting, cost), bbox_inches='tight')
    
    
    
    
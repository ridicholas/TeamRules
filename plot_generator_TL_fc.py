import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from sklearn import metrics
from scipy.stats import ttest_ind
from statistics import mean, stdev
import math

numRuns = 10 #adjust this depending on how many runs of results were produced

#read in results
path = 'checkers_contradiction_results/'
data = path.split('_')[0]
costs = [0, 0.01, 0.05, 0.1, 0.2,
                     0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1]



teams = ['team1', 'team2', 'team3']
team_infos = []
datasets = []
tr_results_filtered = {}
hyrs_results_filtered = {}
fc_tr_results_filtered = {}
fc_hyrs_results_filtered = {}
brs_results = {'team1': [], 'team2': [], 'team3': []}

for team in teams:
    
    tr_results_filtered[team] = {}
    hyrs_results_filtered[team] = {}
    fc_tr_results_filtered[team] = {}
    fc_hyrs_results_filtered[team] = {}

for team in teams:
    for cost in costs:
        tr_results_filtered[team][str(cost)] = []
        hyrs_results_filtered[team][str(cost)] = []
        fc_tr_results_filtered[team][str(cost)] = []
        fc_hyrs_results_filtered[team][str(cost)] = []
        

start_info = pd.read_pickle(path + 'start_info.pkl')
for i in range(0, numRuns):
    datasets.append(pd.read_pickle(path + 'dataset_run{}.pkl'.format(i)))
    for team in teams:
        for cost in costs:
            
            cost = str(cost)
            #tr filtered
            tr_results_filtered[team][cost].append(pd.read_pickle(path + 'cost_{}_'.format(cost) + team + '_tr_filtered_run{}.pkl'.format(i)).sort_values(by='test_error'))
            tr_results_filtered[team][cost][-1] = tr_results_filtered[team][cost][-1][tr_results_filtered[team][cost][-1]['error_conf'] == 0]
            

            #hyrs filtered
            hyrs_results_filtered[team][cost].append(pd.read_pickle(path + 'cost_{}_'.format(cost)+ team + '_hyrs_filtered_run{}.pkl'.format(i)).sort_values(by='test_error'))
            hyrs_results_filtered[team][cost][-1] = hyrs_results_filtered[team][cost][-1][hyrs_results_filtered[team][cost][-1]['error_conf'] == 0]
        
            if cost == '0':
                brs_results[team].append(pd.read_pickle(path + team + '_brs_run{}.pkl'.format(i)).sort_values(by='test_error_brs'))
            
            #tr fc filtered
            fc_tr_results_filtered[team][cost].append(pd.read_pickle(path + 'fc_cost_{}_'.format(cost) + team + '_tr_filtered_run{}.pkl'.format(i)).sort_values(by='test_error'))
            fc_tr_results_filtered[team][cost][-1] = fc_tr_results_filtered[team][cost][-1][fc_tr_results_filtered[team][cost][-1]['error_conf'] == 0]
            

            #hyrs fc filtered
            fc_hyrs_results_filtered[team][cost].append(pd.read_pickle(path + 'fc_cost_{}_'.format(cost)+ team + '_hyrs_filtered_run{}.pkl'.format(i)).sort_values(by='test_error'))
            fc_hyrs_results_filtered[team][cost][-1] = fc_hyrs_results_filtered[team][cost][-1][fc_hyrs_results_filtered[team][cost][-1]['error_conf'] == 0]

for run in range(numRuns):
    for team in teams:
        for cost in costs:
            cost = str(cost)
            if cost == '0':
                brs_results[team][run].loc[0,'test_error'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][datasets[run][team+'TestConf'] < 0.5], brs_results[team][run].loc[0,'team_test_preds'][datasets[run][team+'TestConf'] < 0.5])
            
            
                
                
            
            tr_results_filtered[team][cost][run].loc[0,'test_error'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][datasets[run][team+'TestConf'] < 0.5], tr_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'TestConf'] < 0.5])
            
            
            tr_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][(datasets[run][team+'TestConf'] < 0.5) & (tr_results_filtered[team][cost][run].loc[0,'test_covereds'])], 
                                                                                                                    tr_results_filtered[team][cost][run].loc[0,'modelonly_test_preds'][(datasets[run][team+'TestConf'] < 0.5) & (tr_results_filtered[team][cost][run].loc[0,'test_covereds'])])

            tr_results_filtered[team][cost][run].loc[0, 'contradictions'] = (tr_results_filtered[team][cost][run].loc[0, 'modelonly_test_preds'] != datasets[run][team+'_Ybtest']).sum()
            hyrs_results_filtered[team][cost][run].loc[0,'test_error'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][datasets[run][team+'TestConf'] < 0.5], hyrs_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'TestConf'] < 0.5])
            hyrs_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][(datasets[run][team+'TestConf'] < 0.5) & (hyrs_results_filtered[team][cost][run].loc[0,'test_covereds'])], 
                                                                                                                    hyrs_results_filtered[team][cost][run].loc[0,'modelonly_test_preds'][(datasets[run][team+'TestConf'] < 0.5) & (hyrs_results_filtered[team][cost][run].loc[0,'test_covereds'])])
            hyrs_results_filtered[team][cost][run].loc[0, 'contradictions'] = (hyrs_results_filtered[team][cost][run].loc[0, 'modelonly_test_preds'] != datasets[run][team+'_Ybtest']).sum()
            tr_results_filtered[team][cost][run].loc[0,'total_acceptRegion'] = sum(datasets[run][team+'TestConf'] < 0.5)

            hyrs_results_filtered[team][cost][run].loc[0,'total_acceptRegion'] = sum(datasets[run][team+'TestConf'] < 0.5)

            #fc 
            fc_tr_results_filtered[team][cost][run].loc[0,'test_error'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][datasets[run][team+'TestConf'] < 0.5], fc_tr_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'TestConf'] < 0.5])
            
            
            fc_tr_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][(datasets[run][team+'TestConf'] < 0.5) & (fc_tr_results_filtered[team][cost][run].loc[0,'test_covereds'])], 
                                                                                                                    fc_tr_results_filtered[team][cost][run].loc[0,'modelonly_test_preds'][(datasets[run][team+'TestConf'] < 0.5) & (fc_tr_results_filtered[team][cost][run].loc[0,'test_covereds'])])

            fc_tr_results_filtered[team][cost][run].loc[0, 'contradictions'] = (fc_tr_results_filtered[team][cost][run].loc[0, 'modelonly_test_preds'] != datasets[run][team+'_Ybtest']).sum()
            fc_hyrs_results_filtered[team][cost][run].loc[0,'test_error'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][datasets[run][team+'TestConf'] < 0.5], fc_hyrs_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'TestConf'] < 0.5])
            fc_hyrs_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][(datasets[run][team+'TestConf'] < 0.5) & (fc_hyrs_results_filtered[team][cost][run].loc[0,'test_covereds'])], 
                                                                                                                    fc_hyrs_results_filtered[team][cost][run].loc[0,'modelonly_test_preds'][(datasets[run][team+'TestConf'] < 0.5) & (fc_hyrs_results_filtered[team][cost][run].loc[0,'test_covereds'])])
            fc_hyrs_results_filtered[team][cost][run].loc[0, 'contradictions'] = (fc_hyrs_results_filtered[team][cost][run].loc[0, 'modelonly_test_preds'] != datasets[run][team+'_Ybtest']).sum()
            fc_tr_results_filtered[team][cost][run].loc[0,'total_acceptRegion'] = sum(datasets[run][team+'TestConf'] < 0.5)

            fc_hyrs_results_filtered[team][cost][run].loc[0,'total_acceptRegion'] = sum(datasets[run][team+'TestConf'] < 0.5)


teams = []
for t in start_info.sort_values(by='human accept region train acc').index:
    teams.append('team{}'.format(t))

settings = ['Rational', 'Neutral', 'Irrational']
for whichTeam in range(len(settings)):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 2), dpi=200)
    fig.subplots_adjust(bottom=0.15, wspace=.4)
    team = teams[whichTeam]
    setting = settings[whichTeam]
    costFrame = pd.DataFrame(index=[str(x) for x in costs], data={'Costs': [str(x) for x in costs], 
                                  'TeamRulesTeamLoss': np.zeros(len(costs)), 
                                  'HyRSTeamLoss': np.zeros(len(costs)),
                                  'TeamRulesCov': np.zeros(len(costs)),
                                  'HyRSCov': np.zeros(len(costs)),
                                  'TeamRulesModelOnlyAcceptLoss': np.zeros(len(costs)), 
                                  'HyRSModelOnlyAcceptLoss': np.zeros(len(costs)),
                                  'TeamRulesRejects': np.zeros(len(costs)), 
                                  'HyRSRejects': np.zeros(len(costs)), 
                                  'TR_Contradictions': np.zeros(len(costs)), 
                                  'HyRS_Contradictions': np.zeros(len(costs)),
                                  'TR_Objectives': np.zeros(len(costs)),
                                  'HyRS_Objectives': np.zeros(len(costs)), 
                                  'fc_TeamRulesTeamLoss': np.zeros(len(costs)), 
                                  'fc_HyRSTeamLoss': np.zeros(len(costs)),
                                  'fc_TeamRulesCov': np.zeros(len(costs)),
                                  'fc_HyRSCov': np.zeros(len(costs)),
                                  'fc_TeamRulesModelOnlyAcceptLoss': np.zeros(len(costs)), 
                                  'fc_HyRSModelOnlyAcceptLoss': np.zeros(len(costs)),
                                  'fc_TeamRulesRejects': np.zeros(len(costs)), 
                                  'fc_HyRSRejects': np.zeros(len(costs)), 
                                  'fc_TR_Contradictions': np.zeros(len(costs)), 
                                  'fc_HyRS_Contradictions': np.zeros(len(costs)),
                                  'fc_TR_Objectives': np.zeros(len(costs)),
                                  'fc_HyRS_Objectives': np.zeros(len(costs))})

    for cost in costs:

        cost = str(cost)

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

        fc_TeamRulesLoss = []
        fc_HyRSLoss = []
        fc_TeamRules_modelonly_Loss = []
        fc_HyRS_modelonly_Loss = []
        fc_TeamRulesCov = []
        fc_HyRSCov = []
        fc_TeamRulesRejects = []
        fc_HyRSRejects = []
        fc_TRContradicts = []
        fc_HyRSContradicts = []
        fc_TR_Objectives = []
        fc_HyRS_Objectives = []
        for run in range(numRuns):

            

            TeamRulesLoss.append(tr_results_filtered[team][cost][run].loc[0,'test_error'])
            TeamRules_modelonly_Loss.append(tr_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'])
            TeamRulesRejects.append(tr_results_filtered[team][cost][run].loc[0,'test_rejects'])
            HyRSLoss.append(hyrs_results_filtered[team][cost][run].loc[0,'test_error'])
            HyRS_modelonly_Loss.append(hyrs_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'])
            HyRSRejects.append(hyrs_results_filtered[team][cost][run].loc[0,'test_rejects'])
            TeamRulesCov.append(tr_results_filtered[team][cost][run].loc[0,'test_coverage'])
            HyRSCov.append(hyrs_results_filtered[team][cost][run].loc[0,'test_coverage'])
            TRContradicts.append(tr_results_filtered[team][cost][run].loc[0,'contradictions'])
            HyRSContradicts.append(hyrs_results_filtered[team][cost][run].loc[0,'contradictions'])
            TR_Objectives.append((TeamRulesLoss[-1]) + (float(cost)*TRContradicts[-1])/(datasets[run].shape[0]))
            HyRS_Objectives.append((HyRSLoss[-1]) + (float(cost)*HyRSContradicts[-1])/(datasets[run].shape[0]))

            fc_TeamRulesLoss.append(fc_tr_results_filtered[team][cost][run].loc[0,'test_error'])
            fc_TeamRules_modelonly_Loss.append(fc_tr_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'])
            fc_TeamRulesRejects.append(fc_tr_results_filtered[team][cost][run].loc[0,'test_rejects'])
            fc_HyRSLoss.append(fc_hyrs_results_filtered[team][cost][run].loc[0,'test_error'])
            fc_HyRS_modelonly_Loss.append(fc_hyrs_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'])
            fc_HyRSRejects.append(fc_hyrs_results_filtered[team][cost][run].loc[0,'test_rejects'])
            fc_TeamRulesCov.append(fc_tr_results_filtered[team][cost][run].loc[0,'test_coverage'])
            fc_HyRSCov.append(fc_hyrs_results_filtered[team][cost][run].loc[0,'test_coverage'])
            fc_TRContradicts.append(fc_tr_results_filtered[team][cost][run].loc[0,'contradictions'])
            fc_HyRSContradicts.append(fc_hyrs_results_filtered[team][cost][run].loc[0,'contradictions'])
            fc_TR_Objectives.append((fc_TeamRulesLoss[-1]) + (float(cost)*fc_TRContradicts[-1])/(datasets[run].shape[0]))
            fc_HyRS_Objectives.append((fc_HyRSLoss[-1]) + (float(cost)*fc_HyRSContradicts[-1])/(datasets[run].shape[0]))

        frame = pd.DataFrame({'TeamRulesLoss': TeamRulesLoss,
                             'HyRSLoss': HyRSLoss,
                             'TeamRulesCov': TeamRulesCov,
                             'HyRSCov': HyRSCov,
                             'TRRej': TeamRulesRejects,
                             'HyRSRej': HyRSRejects,
                             'fc_TeamRulesLoss': fc_TeamRulesLoss,
                             'fc_HyRSLoss': fc_HyRSLoss,
                             'fc_TeamRulesCov': fc_TeamRulesCov,
                             'fc_HyRSCov': fc_HyRSCov,
                             'fc_TRRej': fc_TeamRulesRejects,
                             'fc_HyRSRej': fc_HyRSRejects})

        costFrame.loc[cost, 'TeamRulesTeamLoss'] = mean(TeamRulesLoss)
        costFrame.loc[cost, 'TeamRulesTeamLoss_std'] = stdev(TeamRulesLoss)
        costFrame.loc[cost, 'HyRSTeamLoss'] = mean(HyRSLoss)
        costFrame.loc[cost, 'HyRSTeamLoss_std'] = stdev(HyRSLoss)
        costFrame.loc[cost, 'TeamRulesCov'] = mean(TeamRulesCov)
        costFrame.loc[cost, 'TeamRulesRejects'] = mean(TeamRulesRejects)
        costFrame.loc[cost, 'HyRSRejects'] = mean(HyRSRejects)
        costFrame.loc[cost, 'HyRSCov'] = mean(HyRSCov)
        costFrame.loc[cost, 'TeamRulesModelOnlyAcceptLoss'] = mean(TeamRules_modelonly_Loss)
        costFrame.loc[cost, 'HyRSModelOnlyAcceptLoss'] = mean(HyRS_modelonly_Loss)
        costFrame.loc[cost, 'TR_Contradictions'] = mean(TRContradicts)
        costFrame.loc[cost, 'TR_Contradictions_std'] = stdev(TRContradicts)
        costFrame.loc[cost, 'HyRS_Contradictions'] = mean(HyRSContradicts)
        costFrame.loc[cost, 'HyRS_Contradictions_std'] = stdev(HyRSContradicts)
        costFrame.loc[cost, 'TR_Objective'] = mean(TR_Objectives)
        costFrame.loc[cost, 'HyRS_Objective'] = mean(HyRS_Objectives)
        costFrame.loc[cost, 'TR_Objective_SE'] = stdev(TR_Objectives)/math.sqrt(numRuns)
        costFrame.loc[cost, 'HyRS_Objective_SE'] = stdev(HyRS_Objectives)/math.sqrt(numRuns)

        costFrame.loc[cost, 'fc_TeamRulesTeamLoss'] = mean(fc_TeamRulesLoss)
        costFrame.loc[cost, 'fc_TeamRulesTeamLoss_std'] = stdev(fc_TeamRulesLoss)
        costFrame.loc[cost, 'fc_HyRSTeamLoss'] = mean(fc_HyRSLoss)
        costFrame.loc[cost, 'fc_HyRSTeamLoss_std'] = stdev(fc_HyRSLoss)
        costFrame.loc[cost, 'fc_TeamRulesCov'] = mean(fc_TeamRulesCov)
        costFrame.loc[cost, 'fc_TeamRulesRejects'] = mean(fc_TeamRulesRejects)
        costFrame.loc[cost, 'fc_HyRSRejects'] = mean(fc_HyRSRejects)
        costFrame.loc[cost, 'fc_HyRSCov'] = mean(fc_HyRSCov)
        costFrame.loc[cost, 'fc_TeamRulesModelOnlyAcceptLoss'] = mean(fc_TeamRules_modelonly_Loss)
        costFrame.loc[cost, 'fc_HyRSModelOnlyAcceptLoss'] = mean(fc_HyRS_modelonly_Loss)
        costFrame.loc[cost, 'fc_TR_Contradictions'] = mean(fc_TRContradicts)
        costFrame.loc[cost, 'fc_TR_Contradictions_std'] = stdev(fc_TRContradicts)
        costFrame.loc[cost, 'fc_HyRS_Contradictions'] = mean(fc_HyRSContradicts)
        costFrame.loc[cost, 'fc_HyRS_Contradictions_std'] = stdev(fc_HyRSContradicts)
        costFrame.loc[cost, 'fc_TR_Objective'] = mean(fc_TR_Objectives)
        costFrame.loc[cost, 'fc_HyRS_Objective'] = mean(fc_HyRS_Objectives)
        costFrame.loc[cost, 'fc_TR_Objective_SE'] = stdev(fc_TR_Objectives)/math.sqrt(numRuns)
        costFrame.loc[cost, 'fc_HyRS_Objective_SE'] = stdev(fc_HyRS_Objectives)/math.sqrt(numRuns)

    


    TR_loss = []
    TR_con = []
    HyRS_loss = []
    HyRS_con = []

    fc_TR_loss = []
    fc_TR_con = []
    fc_HyRS_loss = []
    fc_HyRS_con = []
    for run in range(numRuns):
        for cost in costs:
            cost = str(cost)
            TR_loss.append(tr_results_filtered[team][cost][run].loc[0,'test_error'])
            TR_con.append(tr_results_filtered[team][cost][run].loc[0,'contradictions'])
            HyRS_loss.append(hyrs_results_filtered[team][cost][run].loc[0,'test_error'])
            HyRS_con.append(hyrs_results_filtered[team][cost][run].loc[0,'contradictions'])

            fc_TR_loss.append(fc_tr_results_filtered[team][cost][run].loc[0,'test_error'])
            fc_TR_con.append(fc_tr_results_filtered[team][cost][run].loc[0,'contradictions'])
            fc_HyRS_loss.append(fc_hyrs_results_filtered[team][cost][run].loc[0,'test_error'])
            fc_HyRS_con.append(fc_hyrs_results_filtered[team][cost][run].loc[0,'contradictions'])
    TR_loss = np.array(TR_loss)
    TR_con = np.array(TR_con)
    HyRS_loss = np.array(HyRS_loss)
    HyRS_con = np.array(HyRS_con)

    fc_TR_loss = np.array(fc_TR_loss)
    fc_TR_con = np.array(fc_TR_con)
    fc_HyRS_loss = np.array(fc_HyRS_loss)
    fc_HyRS_con = np.array(fc_HyRS_con)

    #dedup TRs for scatter
    l = list(zip(TR_loss, TR_con))
    z = list(zip(*[i for n, i in enumerate(l) if i not in l[:n]]))
    TR_loss = np.array(z[0])
    TR_con = np.array(z[1])

    l = list(zip(fc_TR_loss, fc_TR_con))
    z = list(zip(*[i for n, i in enumerate(l) if i not in l[:n]]))
    fc_TR_loss = np.array(z[0])
    fc_TR_con = np.array(z[1])



    l = list(zip(HyRS_loss, HyRS_con))
    z = list(zip(*[i for n, i in enumerate(l) if i not in l[:n]]))
    HyRS_loss = np.array(z[0])
    HyRS_con = np.array(z[1])

    l = list(zip(fc_HyRS_loss, fc_HyRS_con))
    z = list(zip(*[i for n, i in enumerate(l) if i not in l[:n]]))
    fc_HyRS_loss = np.array(z[0])
    fc_HyRS_con = np.array(z[1])
    #fig.suptitle('{} Setting'.format(setting), fontsize=16)
    i=0
    for row in ax:
        if i==0:
            
            
            costFrame.sort_values(by=['Costs'], inplace=True)
            row.plot(costFrame['Costs'], costFrame['HyRS_Objective'], c='red', marker='v', label = 'c-HyRS', markersize=6)
            row.plot(costFrame['Costs'], costFrame['TR_Objective'], c='blue', marker='.', label='TeamRules', markersize=6)

            row.plot(costFrame['Costs'], costFrame['fc_HyRS_Objective'], c='green', marker='v', label = 'fc_c-HyRS', markersize=6)
            row.plot(costFrame['Costs'], costFrame['fc_TR_Objective'], c='brown', marker='.', label='fc_TeamRules', markersize=6)


            row.vlines(costFrame['Costs'], 
                       costFrame['HyRS_Objective']-(2*costFrame['HyRS_Objective_SE']),
                       costFrame['HyRS_Objective']+(2*costFrame['HyRS_Objective_SE']) ,
                      colors='red', alpha=0.7, linewidth=1)
            row.vlines(costFrame['Costs'], 
                       costFrame['TR_Objective']-(2*costFrame['TR_Objective_SE']),
                       costFrame['TR_Objective']+(2*costFrame['TR_Objective_SE']) ,
                      colors='blue', alpha=0.7, linewidth=1)
            
            row.vlines(costFrame['Costs'], 
                       costFrame['fc_HyRS_Objective']-(2*costFrame['fc_HyRS_Objective_SE']),
                       costFrame['fc_HyRS_Objective']+(2*costFrame['fc_HyRS_Objective_SE']) ,
                      colors='green', alpha=0.7, linewidth=1)
            row.vlines(costFrame['Costs'], 
                       costFrame['fc_TR_Objective']-(2*costFrame['fc_TR_Objective_SE']),
                       costFrame['fc_TR_Objective']+(2*costFrame['fc_TR_Objective_SE']) ,
                      colors='brown', alpha=0.7, linewidth=1)


            row.set_xlabel('Contradiction Cost', fontsize=14)
            row.set_ylabel('Team Loss', fontsize=14)
            row.tick_params(labelrotation=45, labelsize=10)
            #row.set_title('{} Setting'.format(setting), fontsize=15)
            row.legend(prop={'size': 6})

        else:
            row.scatter(TR_con, TR_loss, c='blue', marker = '.',  alpha=0.2, label='TeamRules', s=6)
            row.scatter(HyRS_con, HyRS_loss, c='red', marker = 'v',  alpha=0.2, label='c-HyRS', s=6)

            row.scatter(fc_TR_con, fc_TR_loss, c='brown', marker = '.',  alpha=0.2, label='fc_TeamRules', s=6)
            row.scatter(fc_HyRS_con, fc_HyRS_loss, c='green', marker = 'v',  alpha=0.2, label='fc_c-HyRS', s=6)


            leg = row.legend(prop={'size': 6})
            for lh in leg.legendHandles: 
                lh.set_alpha(1)


            #col.plot(x, y)
            costFrame.sort_values(by=['HyRS_Contradictions'], inplace=True)
            row.plot(costFrame['HyRS_Contradictions'], costFrame['HyRSTeamLoss'], marker = 'v', markersize=6, c='red', label = 'c-HyRS')
            costFrame.sort_values(by=['TR_Contradictions'], inplace=True)
            row.plot(costFrame['TR_Contradictions'], costFrame['TeamRulesTeamLoss'], marker='.', markersize=6, c='blue', label='TeamRules')

            costFrame.sort_values(by=['fc_HyRS_Contradictions'], inplace=True)
            row.plot(costFrame['fc_HyRS_Contradictions'], costFrame['fc_HyRSTeamLoss'], marker = 'v', markersize=6, c='green', label = 'fc_c-HyRS')
            costFrame.sort_values(by=['fc_TR_Contradictions'], inplace=True)
            row.plot(costFrame['fc_TR_Contradictions'], costFrame['fc_TeamRulesTeamLoss'], marker='.', markersize=6, c='brown', label='fc_TeamRules')
            row.set_xlabel('# of Contradictions', fontsize=14)
            row.set_ylabel('Team Loss', fontsize=14)
            row.tick_params(labelrotation=45, labelsize=10)
            #row.set_title('{} Setting'.format(setting), fontsize=15)
            
        
            
        
        i+=1
    fig.savefig('Plots/{}_{}.png'.format(data,setting), bbox_inches='tight')
    
    


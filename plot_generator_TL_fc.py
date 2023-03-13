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
path = 'fico_contradiction_results/'

asym_loss = [1,1]
data = path.split('_')[0]
costs = [0, 0.01, 0.05, 0.1, 0.2,
                     0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1]



tr_conf = 0.5
fc_TR_conf = 0

teams = ['team1', 'team2', 'team3']
#teams = ['team2']
team_infos = []
datasets = []
tr_results_filtered = {}
fc_TR_results_filtered = {}
brs_results = {'team1': [], 'team2': [], 'team3': []}

for team in teams:  
    tr_results_filtered[team] = {}
    fc_TR_results_filtered[team] = {}

for team in teams:
    for cost in costs:
        tr_results_filtered[team][str(cost)] = []
        fc_TR_results_filtered[team][str(cost)] = []
        

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

            #fc_TR filtered
            fc_TR_results_filtered[team][cost].append(pd.read_pickle(path + 'fc_cost_{}_'.format(cost)+ team + '_tr_filtered_run{}.pkl'.format(i)).sort_values(by='test_error'))
            fc_TR_results_filtered[team][cost][-1] = fc_TR_results_filtered[team][cost][-1][fc_TR_results_filtered[team][cost][-1]['mental_conf'] == fc_TR_conf].reset_index()
        
            #if cost == '0':
                #brs_results[team].append(pd.read_pickle(path + team + '_brs_run{}.pkl'.format(i)).sort_values(by='test_error_brs'))

for run in range(numRuns):
    for team in teams:
            
        for cost in costs:
            cost = str(cost)
            #if cost == '0':
                #brs_results[team][run].loc[0,'error_rate_acceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][datasets[run][team+'TestConf'] < 0.5], brs_results[team][run].loc[0,'team_test_preds'][datasets[run][team+'TestConf'] < 0.5])
            
            
                
                
            
            tr_results_filtered[team][cost][run].loc[0,'error_rate_acceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][datasets[run][team+'TestConf'] < 0.5], tr_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'TestConf'] < 0.5])
            
            
            tr_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][(datasets[run][team+'TestConf'] < 0.5) & (tr_results_filtered[team][cost][run].loc[0,'test_covereds'])], 
                                                                                                                    tr_results_filtered[team][cost][run].loc[0,'modelonly_test_preds'][(datasets[run][team+'TestConf'] < 0.5) & (tr_results_filtered[team][cost][run].loc[0,'test_covereds'])])

            tr_results_filtered[team][cost][run].loc[0, 'contradictions'] = (tr_results_filtered[team][cost][run].loc[0, 'modelonly_test_preds'] != datasets[run][team+'_Ybtest']).sum()
            fc_TR_results_filtered[team][cost][run].loc[0,'error_rate_acceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][datasets[run][team+'TestConf'] < 0.5], fc_TR_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'TestConf'] < 0.5])
            fc_TR_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][(datasets[run][team+'TestConf'] < 0.5) & (fc_TR_results_filtered[team][cost][run].loc[0,'test_covereds'])], 
                                                                                                                    fc_TR_results_filtered[team][cost][run].loc[0,'modelonly_test_preds'][(datasets[run][team+'TestConf'] < 0.5) & (fc_TR_results_filtered[team][cost][run].loc[0,'test_covereds'])])
            fc_TR_results_filtered[team][cost][run].loc[0, 'contradictions'] = (fc_TR_results_filtered[team][cost][run].loc[0, 'modelonly_test_preds'] != datasets[run][team+'_Ybtest']).sum()
            tr_results_filtered[team][cost][run].loc[0,'total_acceptRegion'] = sum(datasets[run][team+'TestConf'] < 0.5)

            fc_TR_results_filtered[team][cost][run].loc[0,'total_acceptRegion'] = sum(datasets[run][team+'TestConf'] < 0.5)


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
                                  'fc_TRTeamLoss': np.zeros(len(costs)),
                                  'TeamRulesCov': np.zeros(len(costs)),
                                  'fc_TRCov': np.zeros(len(costs)),
                                  'TeamRulesModelOnlyAcceptLoss': np.zeros(len(costs)), 
                                  'fc_TRModelOnlyAcceptLoss': np.zeros(len(costs)),
                                  'TeamRulesRejects': np.zeros(len(costs)), 
                                  'fc_TRRejects': np.zeros(len(costs)), 
                                  'TR_Contradictions': np.zeros(len(costs)), 
                                  'fc_TR_Contradictions': np.zeros(len(costs)),
                                  'TR_Objectives': np.zeros(len(costs)),
                                  'fc_TR_Objectives': np.zeros(len(costs)),})

    for cost in costs:

        cost = str(cost)

        TeamRulesLoss = []
        fc_TRLoss = []
        TeamRules_modelonly_Loss = []
        fc_TR_modelonly_Loss = []
        TeamRulesCov = []
        fc_TRCov = []
        TeamRulesRejects = []
        fc_TRRejects = []
        TRContradicts = []
        fc_TRContradicts = []
        TR_Objectives = []
        fc_TR_Objectives = []
        for run in range(numRuns):

            
            newTRLoss = asym_loss[0] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] == 0] != tr_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] == 0]).sum()
            newTRLoss += asym_loss[1] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] != 0] != tr_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] != 0]).sum()
            newTRLoss = newTRLoss/len(datasets[run][team+'_Ytest'])
            TeamRulesLoss.append(newTRLoss)
            TeamRules_modelonly_Loss.append(tr_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'])
            TeamRulesRejects.append(tr_results_filtered[team][cost][run].loc[0,'test_rejects'])
            newfc_TRLoss = asym_loss[0] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] == 0] != fc_TR_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] == 0]).sum()
            newfc_TRLoss += asym_loss[1] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] != 0] != fc_TR_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] != 0]).sum()
            newfc_TRLoss = newfc_TRLoss/len(datasets[run][team+'_Ytest'])
            fc_TRLoss.append(newfc_TRLoss)
            fc_TR_modelonly_Loss.append(fc_TR_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'])
            fc_TRRejects.append(fc_TR_results_filtered[team][cost][run].loc[0,'test_rejects'])
            TeamRulesCov.append(tr_results_filtered[team][cost][run].loc[0,'test_coverage'])
            fc_TRCov.append(fc_TR_results_filtered[team][cost][run].loc[0,'test_coverage'])
            TRContradicts.append(tr_results_filtered[team][cost][run].loc[0,'contradictions'])
            fc_TRContradicts.append(fc_TR_results_filtered[team][cost][run].loc[0,'contradictions'])
            TR_Objectives.append((TeamRulesLoss[-1]) + (float(cost)*TRContradicts[-1])/(datasets[run].shape[0]))
            fc_TR_Objectives.append((fc_TRLoss[-1]) + (float(cost)*fc_TRContradicts[-1])/(datasets[run].shape[0]))

        frame = pd.DataFrame({'TeamRulesLoss': TeamRulesLoss,
                             'fc_TRLoss': fc_TRLoss,
                             'TeamRulesCov': TeamRulesCov,
                             'fc_TRCov': fc_TRCov,
                             'TRRej': TeamRulesRejects,
                             'fc_TRRej': fc_TRRejects})
        costFrame.loc[cost, 'TeamRulesTeamLoss'] = mean(TeamRulesLoss)
        costFrame.loc[cost, 'TeamRulesTeamLoss_std'] = stdev(TeamRulesLoss)
        costFrame.loc[cost, 'fc_TRTeamLoss'] = mean(fc_TRLoss)
        costFrame.loc[cost, 'fc_TRTeamLoss_std'] = stdev(fc_TRLoss)
        costFrame.loc[cost, 'TeamRulesCov'] = mean(TeamRulesCov)
        costFrame.loc[cost, 'TeamRulesRejects'] = mean(TeamRulesRejects)
        costFrame.loc[cost, 'fc_TRRejects'] = mean(fc_TRRejects)
        costFrame.loc[cost, 'fc_TRCov'] = mean(fc_TRCov)
        costFrame.loc[cost, 'TeamRulesModelOnlyAcceptLoss'] = mean(TeamRules_modelonly_Loss)
        costFrame.loc[cost, 'fc_TRModelOnlyAcceptLoss'] = mean(fc_TR_modelonly_Loss)
        costFrame.loc[cost, 'TR_Contradictions'] = mean(TRContradicts)
        costFrame.loc[cost, 'TR_Contradictions_std'] = stdev(TRContradicts)
        costFrame.loc[cost, 'fc_TR_Contradictions'] = mean(fc_TRContradicts)
        costFrame.loc[cost, 'fc_TR_Contradictions_std'] = stdev(fc_TRContradicts)
        costFrame.loc[cost, 'TR_Objective'] = mean(TR_Objectives)
        costFrame.loc[cost, 'fc_TR_Objective'] = mean(fc_TR_Objectives)
        costFrame.loc[cost, 'TR_Objective_SE'] = stdev(TR_Objectives)/math.sqrt(numRuns)
        costFrame.loc[cost, 'fc_TR_Objective_SE'] = stdev(fc_TR_Objectives)/math.sqrt(numRuns)

    


    TR_loss = []
    TR_con = []
    fc_TR_loss = []
    fc_TR_con = []
    for run in range(numRuns):
        for cost in costs:
            cost = str(cost)
            newTRLoss = asym_loss[0] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] == 0] != tr_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] == 0]).sum()
            newTRLoss += asym_loss[1] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] != 0] != tr_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] != 0]).sum()
            newTRLoss = newTRLoss/len(datasets[run][team+'_Ytest'])
            TR_loss.append(newTRLoss)
            TR_con.append(tr_results_filtered[team][cost][run].loc[0,'contradictions'])
            newfc_TRLoss = asym_loss[0] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] == 0] != fc_TR_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] == 0]).sum()
            newfc_TRLoss += asym_loss[1] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] != 0] != fc_TR_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] != 0]).sum()
            newfc_TRLoss = newfc_TRLoss/len(datasets[run][team+'_Ytest'])
            fc_TR_loss.append(newfc_TRLoss)
            fc_TR_con.append(fc_TR_results_filtered[team][cost][run].loc[0,'contradictions'])
    TR_loss = np.array(TR_loss)
    TR_con = np.array(TR_con)
    fc_TR_loss = np.array(fc_TR_loss)
    fc_TR_con = np.array(fc_TR_con)

    #dedup TRs for scatter
    l = list(zip(TR_loss, TR_con))
    z = list(zip(*[i for n, i in enumerate(l) if i not in l[:n]]))
    TR_loss = np.array(z[0])
    TR_con = np.array(z[1])

    l = list(zip(fc_TR_loss, fc_TR_con))
    z = list(zip(*[i for n, i in enumerate(l) if i not in l[:n]]))
    fc_TR_loss = np.array(z[0])
    fc_TR_con = np.array(z[1])
    i=0
    for row in ax:
        if i==0:
            
            
            costFrame.sort_values(by=['Costs'], inplace=True)
            row.plot(costFrame['Costs'], costFrame['fc_TR_Objective'], c='brown', marker='v', label = 'fc_TeamRules', markersize=1)
            row.plot(costFrame['Costs'], costFrame['TR_Objective'], c='blue', marker='.', label='TeamRules', markersize=1)
            row.fill_between(costFrame['Costs'], 
                       costFrame['fc_TR_Objective']-(costFrame['fc_TR_Objective_SE']),
                       costFrame['fc_TR_Objective']+(costFrame['fc_TR_Objective_SE']) ,
                      color='brown', alpha=0.2)
            row.fill_between(costFrame['Costs'], 
                       costFrame['TR_Objective']-(costFrame['TR_Objective_SE']),
                       costFrame['TR_Objective']+(costFrame['TR_Objective_SE']) ,
                      color='blue', alpha=0.2)
            row.set_xlabel('Reconciliation Cost', fontsize=14)
            row.set_ylabel('Team Loss', fontsize=14)
            row.tick_params(labelrotation=45, labelsize=10)
            row.legend(prop={'size': 6})

        else:
            row.scatter(TR_con, TR_loss, c='blue', marker = '.',  alpha=0.2, label='TeamRules', s=6)
            row.scatter(fc_TR_con, fc_TR_loss, c='brown', marker = 'v',  alpha=0.2, label='fc_TeamRules', s=6)
            leg = row.legend(prop={'size': 6})
            for lh in leg.legendHandles: 
                lh.set_alpha(1)

            costFrame.sort_values(by=['fc_TR_Contradictions'], inplace=True)
            row.plot(costFrame['fc_TR_Contradictions'], costFrame['fc_TRTeamLoss'], marker = 'v', markersize=1, c='brown', label = 'fc_TeamRules')
            costFrame.sort_values(by=['TR_Contradictions'], inplace=True)
            row.plot(costFrame['TR_Contradictions'], costFrame['TeamRulesTeamLoss'], marker='.', markersize=1, c='blue', label='TeamRules')
            row.set_xlabel('# of Contradictions', fontsize=14)
            row.set_ylabel('Team Decision Loss', fontsize=14)
            row.tick_params(labelrotation=45, labelsize=10)
            
            
        
            
        
        i+=1
    #fig.savefig('Plots/asym_2_1_{}_{}.png'.format(data,setting), bbox_inches='tight')
    fig.savefig('Plots/{}_{}_fc.png'.format(data,setting), bbox_inches='tight')


    
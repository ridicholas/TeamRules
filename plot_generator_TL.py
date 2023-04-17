import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from sklearn import metrics
from scipy.stats import ttest_ind
from statistics import mean, stdev
import math

numRuns = 10 #adjust this depending on how many runs of results were produced

path = 'gaussian_contradiction_results_bias/'

asym_loss = [1,1]
data = path.split('_')[0]
costs = [0, 0.01, 0.05, 0.1, 0.2,
                     0.3, 0.4, 0.5, 0.6, 0.7, 0.8]



tr_conf = 0.5
hyrs_conf = 0

teams = ['team1', 'team2', 'team3']
#teams = ['team2']
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
            hyrs_results_filtered[team][cost].append(pd.read_pickle(path + 'cost_{}_'.format('0')+ team + '_hyrs_filtered_run{}.pkl'.format(i)).sort_values(by='test_error'))
            hyrs_results_filtered[team][cost][-1] = hyrs_results_filtered[team][cost][-1][hyrs_results_filtered[team][cost][-1]['mental_conf'] == hyrs_conf].reset_index()
        
            if cost == '0':
                brs_results[team].append(pd.read_pickle(path + team + '_brs_run{}.pkl'.format(i)).sort_values(by='test_error_brs'))

for run in range(numRuns):
    for team in teams:
            
        for cost in costs:
            cost = str(cost)
            if cost == '0':
                brs_results[team][run].loc[0,'error_rate_acceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][datasets[run][team+'TestConf'] < 0.5], brs_results[team][run].loc[0,'team_test_preds'][datasets[run][team+'TestConf'] < 0.5])
            
            
                
                
            
            tr_results_filtered[team][cost][run].loc[0,'error_rate_acceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][datasets[run][team+'TestConf'] < 0.5], tr_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'TestConf'] < 0.5])
            
            
            tr_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][(datasets[run][team+'TestConf'] < 0.5) & (tr_results_filtered[team][cost][run].loc[0,'test_covereds'])], 
                                                                                                                    tr_results_filtered[team][cost][run].loc[0,'modelonly_test_preds'][(datasets[run][team+'TestConf'] < 0.5) & (tr_results_filtered[team][cost][run].loc[0,'test_covereds'])])

            tr_results_filtered[team][cost][run].loc[0, 'contradictions'] = (tr_results_filtered[team][cost][run].loc[0, 'modelonly_test_preds'] != datasets[run][team+'_Ybtest']).sum()
            hyrs_results_filtered[team][cost][run].loc[0,'error_rate_acceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][datasets[run][team+'TestConf'] < 0.5], hyrs_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'TestConf'] < 0.5])
            hyrs_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][(datasets[run][team+'TestConf'] < 0.5) & (hyrs_results_filtered[team][cost][run].loc[0,'test_covereds'])], 
                                                                                                                    hyrs_results_filtered[team][cost][run].loc[0,'modelonly_test_preds'][(datasets[run][team+'TestConf'] < 0.5) & (hyrs_results_filtered[team][cost][run].loc[0,'test_covereds'])])
            hyrs_results_filtered[team][cost][run].loc[0, 'contradictions'] = (hyrs_results_filtered[team][cost][run].loc[0, 'modelonly_test_preds'] != datasets[run][team+'_Ybtest']).sum()
            tr_results_filtered[team][cost][run].loc[0,'total_acceptRegion'] = sum(datasets[run][team+'TestConf'] < 0.5)

            hyrs_results_filtered[team][cost][run].loc[0,'total_acceptRegion'] = sum(datasets[run][team+'TestConf'] < 0.5)

            brs_results[team][run].loc[0, 'contradictions'] = (brs_results[team][run].loc[0, 'modelonly_test_preds'] != datasets[run][team+'_Ybtest']).sum()


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
                                  'BRSTeamLoss': np.zeros(len(costs)), 
                                  'BRSCov': np.zeros(len(costs)), 
                                  'BRS_Contradictions': np.zeros(len(costs)),
                                  'BRS_Objectives': np.zeros(len(costs))})

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
        BRS_Objectives = []
        BRSContradicts = []
        BRSLoss = []
        BRSCov = []
        for run in range(numRuns):

            
            newTRLoss = asym_loss[0] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] == 0] != tr_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] == 0]).sum()
            newTRLoss += asym_loss[1] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] != 0] != tr_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] != 0]).sum()
            newTRLoss = newTRLoss/len(datasets[run][team+'_Ytest'])
            TeamRulesLoss.append(newTRLoss)
            TeamRules_modelonly_Loss.append(tr_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'])
            TeamRulesRejects.append(tr_results_filtered[team][cost][run].loc[0,'test_rejects'])
            TeamRulesCov.append(tr_results_filtered[team][cost][run].loc[0,'test_coverage'])
            TRContradicts.append(tr_results_filtered[team][cost][run].loc[0,'contradictions'])
            TR_Objectives.append((TeamRulesLoss[-1]) + (float(cost)*TRContradicts[-1])/(datasets[run].shape[0]))


            newHYRSLoss = asym_loss[0] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] == 0] != hyrs_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] == 0]).sum()
            newHYRSLoss += asym_loss[1] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] != 0] != hyrs_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] != 0]).sum()
            newHYRSLoss = newHYRSLoss/len(datasets[run][team+'_Ytest'])
            HyRSLoss.append(newHYRSLoss)
            HyRS_modelonly_Loss.append(hyrs_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'])
            HyRSRejects.append(hyrs_results_filtered[team][cost][run].loc[0,'test_rejects'])
            HyRSCov.append(hyrs_results_filtered[team][cost][run].loc[0,'test_coverage'])
            HyRSContradicts.append(hyrs_results_filtered[team][cost][run].loc[0,'contradictions'])
            HyRS_Objectives.append((HyRSLoss[-1]) + (float(cost)*HyRSContradicts[-1])/(datasets[run].shape[0]))
            BRSLoss.append((datasets[run][team+'_Ytest'] != brs_results[team][run].loc[0, 'team_test_preds']).sum()/len(datasets[run][team+'_Ytest']))
            BRSContradicts.append(brs_results[team][run].loc[0,'contradictions'])
            BRS_Objectives.append((BRSLoss[-1]) + (float(cost)*BRSContradicts[-1])/(datasets[run].shape[0]))
            

        frame = pd.DataFrame({'TeamRulesLoss': TeamRulesLoss,
                             'HyRSLoss': HyRSLoss,
                             'TeamRulesCov': TeamRulesCov,
                             'HyRSCov': HyRSCov,
                             'TRRej': TeamRulesRejects,
                             'HyRSRej': HyRSRejects}),
                             #'BRSLoss': BRSLoss})
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
        costFrame.loc[cost, 'BRSTeamLoss'] = mean(BRSLoss)
        costFrame.loc[cost, 'BRSTeamLoss_std'] = stdev(BRSLoss)
        costFrame.loc[cost, 'BRS_Contradictions'] = mean(BRSContradicts)
        costFrame.loc[cost, 'BRS_Objective'] = mean(BRS_Objectives)
        costFrame.loc[cost, 'BRS_Objective_SE'] = stdev(BRS_Objectives)/math.sqrt(numRuns)

    


    TR_loss = []
    TR_con = []
    HyRS_loss = []
    HyRS_con = []
    BRS_loss = []
    BRS_con = []

    for run in range(numRuns):
        for cost in costs:
            cost = str(cost)
            newTRLoss = asym_loss[0] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] == 0] != tr_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] == 0]).sum()
            newTRLoss += asym_loss[1] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] != 0] != tr_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] != 0]).sum()
            newTRLoss = newTRLoss/len(datasets[run][team+'_Ytest'])
            TR_loss.append(newTRLoss)
            TR_con.append(tr_results_filtered[team][cost][run].loc[0,'contradictions'])
            newHYRSLoss = asym_loss[0] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] == 0] != hyrs_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] == 0]).sum()
            newHYRSLoss += asym_loss[1] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] != 0] != hyrs_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] != 0]).sum()
            newHYRSLoss = newHYRSLoss/len(datasets[run][team+'_Ytest'])
            HyRS_loss.append(newHYRSLoss)
            HyRS_con.append(hyrs_results_filtered[team][cost][run].loc[0,'contradictions'])
            BRS_loss.append((datasets[run][team+'_Ytest'] != brs_results[team][run].loc[0, 'team_test_preds']).sum()/len(datasets[run][team+'_Ytest']))
            BRS_con.append(brs_results[team][run].loc[0,'contradictions'])
    TR_loss = np.array(TR_loss)
    TR_con = np.array(TR_con)
    HyRS_loss = np.array(HyRS_loss)
    HyRS_con = np.array(HyRS_con)
    BRS_loss = np.array(BRS_loss)
    BRS_con = np.array(BRS_con)

    #dedup TRs for scatter
    l = list(zip(TR_loss, TR_con))
    z = list(zip(*[i for n, i in enumerate(l) if i not in l[:n]]))
    TR_loss = np.array(z[0])
    TR_con = np.array(z[1])

    l = list(zip(HyRS_loss, HyRS_con))
    z = list(zip(*[i for n, i in enumerate(l) if i not in l[:n]]))
    HyRS_loss = np.array(z[0])
    HyRS_con = np.array(z[1])

    l = list(zip(BRS_loss, BRS_con))
    z = list(zip(*[i for n, i in enumerate(l) if i not in l[:n]]))
    BRS_loss = np.array(z[0])
    BRS_con = np.array(z[1])
    #fig.suptitle('{} Setting'.format(setting), fontsize=16)
    i=0
    for row in ax:
        if i==0:
            
            
            costFrame.sort_values(by=['Costs'], inplace=True)
            row.plot(costFrame['Costs'], costFrame['HyRS_Objective'], c='red', marker='v', label = 'HyRS', markersize=1)
            row.plot(costFrame['Costs'], costFrame['TR_Objective'], c='blue', marker='.', label='TeamRules', markersize=1)
            row.plot(costFrame['Costs'], costFrame['BRS_Objective'], c='gray', marker='x', label='BRS', markersize=1)
            row.fill_between(costFrame['Costs'], 
                       costFrame['HyRS_Objective']-(costFrame['HyRS_Objective_SE']),
                       costFrame['HyRS_Objective']+(costFrame['HyRS_Objective_SE']) ,
                      color='red', alpha=0.2)
            row.fill_between(costFrame['Costs'], 
                       costFrame['BRS_Objective']-(costFrame['BRS_Objective_SE']),
                       costFrame['BRS_Objective']+(costFrame['BRS_Objective_SE']) ,
                      color='gray', alpha=0.2)
            row.fill_between(costFrame['Costs'], 
                       costFrame['TR_Objective']-(costFrame['TR_Objective_SE']),
                       costFrame['TR_Objective']+(costFrame['TR_Objective_SE']) ,
                      color='blue', alpha=0.2)
            row.set_xlabel('Reconciliation Cost', fontsize=14)
            row.set_ylabel('Team Loss', fontsize=14)
            row.tick_params(labelrotation=45, labelsize=10)
            #row.set_title('{} Setting'.format(setting), fontsize=15)
            row.legend(prop={'size': 6})

        else:
            row.scatter(TR_con, TR_loss, c='blue', marker = '.',  alpha=0.2, label='TeamRules', s=6)
            row.scatter(HyRS_con, HyRS_loss, c='red', marker = 'v',  alpha=0.2, label='HyRS', s=6)
            row.scatter(BRS_con, BRS_loss, c='gray', marker = 'x',  alpha=0.2, label='BRS', s=6)
            leg = row.legend(prop={'size': 6})
            for lh in leg.legendHandles: 
                lh.set_alpha(1)


            #col.plot(x, y)
            costFrame.sort_values(by=['HyRS_Contradictions'], inplace=True)
            row.plot(costFrame['HyRS_Contradictions'], costFrame['HyRSTeamLoss'], marker = 'v', markersize=4, c='red', label = 'HyRS')
            costFrame.sort_values(by=['BRS_Contradictions'], inplace=True)
            row.plot(costFrame['BRS_Contradictions'], costFrame['BRSTeamLoss'], marker = 'x', markersize=4, c='gray', label = 'BRS')
            costFrame.sort_values(by=['TR_Contradictions'], inplace=True)
            row.plot(costFrame['TR_Contradictions'], costFrame['TeamRulesTeamLoss'], marker='.', markersize=4, c='blue', label='TeamRules')
            row.set_xlabel('# of Contradictions', fontsize=14)
            row.set_ylabel('Team Decision Loss', fontsize=14)
            row.tick_params(labelrotation=45, labelsize=10)
            #row.set_title('{} Setting'.format(setting), fontsize=15)
            
        
            
        
        i+=1
    #fig.savefig('Plots/asym_2_1_{}_{}.png'.format(data,setting), bbox_inches='tight')
    fig.savefig('Plots/det_{}_{}.png'.format(data,setting), bbox_inches='tight')


    
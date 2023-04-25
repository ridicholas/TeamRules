import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from sklearn import metrics
from scipy.stats import ttest_ind
from statistics import mean, stdev
import math

numRuns = 4 #adjust this depending on how many runs of results were produced

path = 'fico_contradiction_results_det/'

asym_loss = [1,1]
data = path.split('_')[0]
costs = [0, 0.01, 0.05, 0.1, 0.2,
                     0.3, 0.4, 0.5, 0.6, 0.7, 0.8]



tr_conf = 0.5
FILT_conf = 0

teams = ['team1', 'team2', 'team3']
#teams = ['team2']
team_infos = []
datasets = []
tr_both_results_filtered = {}
tr_filter_results_filtered = {}
tr_objective_results_filtered = {'team1': [], 'team2': [], 'team3': []}

for team in teams:
    
    tr_both_results_filtered[team] = {}
    tr_filter_results_filtered[team] = {}
    tr_objective_results_filtered[team] = {}

for team in teams:
    for cost in costs:
        tr_both_results_filtered[team][str(cost)] = []
        tr_filter_results_filtered[team][str(cost)] = []
        tr_objective_results_filtered[team][str(cost)] = []
        

start_info = pd.read_pickle(path + 'start_info.pkl')
for i in range(0, numRuns):
    datasets.append(pd.read_pickle(path + 'dataset_run{}.pkl'.format(i)))
    for team in teams:
        for cost in costs:
            
            cost = str(cost)
            #both filtered and objective
            tr_both_results_filtered[team][cost].append(pd.read_pickle(path + 'cost_{}_'.format(cost) + team + '_tr_filtered_run{}.pkl'.format(i)).sort_values(by='test_error'))
            tr_both_results_filtered[team][cost][-1] = tr_both_results_filtered[team][cost][-1][tr_both_results_filtered[team][cost][-1]['mental_conf'] == 0.5].reset_index()

            #just objective
            tr_objective_results_filtered[team][cost].append(pd.read_pickle(path + 'cost_{}_'.format(cost) + team + '_tr_filtered_run{}.pkl'.format(i)).sort_values(by='test_error'))
            tr_objective_results_filtered[team][cost][-1] = tr_objective_results_filtered[team][cost][-1][tr_objective_results_filtered[team][cost][-1]['mental_conf'] == 0].reset_index()
            
            #just filtering results
            tr_filter_results_filtered[team][cost].append(pd.read_pickle(path + 'cost_{}_'.format(cost)+ team + '_hyrs_filtered_run{}.pkl'.format(i)).sort_values(by='test_error'))
            tr_filter_results_filtered[team][cost][-1] = tr_filter_results_filtered[team][cost][-1][tr_filter_results_filtered[team][cost][-1]['mental_conf'] == 0.5].reset_index()
        
            

for run in range(numRuns):
    for team in teams:
            
        for cost in costs:
            cost = str(cost)
            
            
            tr_both_results_filtered[team][cost][run].loc[0,'error_rate_acceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][datasets[run][team+'TestConf'] < 0.5], tr_both_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'TestConf'] < 0.5])
            
            
            tr_both_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][(datasets[run][team+'TestConf'] < 0.5) & (tr_both_results_filtered[team][cost][run].loc[0,'test_covereds'])], 
                                                                                                                    tr_both_results_filtered[team][cost][run].loc[0,'modelonly_test_preds'][(datasets[run][team+'TestConf'] < 0.5) & (tr_both_results_filtered[team][cost][run].loc[0,'test_covereds'])])

            tr_both_results_filtered[team][cost][run].loc[0, 'contradictions'] = (tr_both_results_filtered[team][cost][run].loc[0, 'modelonly_test_preds'] != datasets[run][team+'_Ybtest']).sum()
            
            tr_filter_results_filtered[team][cost][run].loc[0,'error_rate_acceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][datasets[run][team+'TestConf'] < 0.5], tr_filter_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'TestConf'] < 0.5])
            tr_filter_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][(datasets[run][team+'TestConf'] < 0.5) & (tr_filter_results_filtered[team][cost][run].loc[0,'test_covereds'])], 
                                                                                                                    tr_filter_results_filtered[team][cost][run].loc[0,'modelonly_test_preds'][(datasets[run][team+'TestConf'] < 0.5) & (tr_filter_results_filtered[team][cost][run].loc[0,'test_covereds'])])
            tr_filter_results_filtered[team][cost][run].loc[0, 'contradictions'] = (tr_filter_results_filtered[team][cost][run].loc[0, 'modelonly_test_preds'] != datasets[run][team+'_Ybtest']).sum()
            tr_both_results_filtered[team][cost][run].loc[0,'total_acceptRegion'] = sum(datasets[run][team+'TestConf'] < 0.5)

            tr_filter_results_filtered[team][cost][run].loc[0,'total_acceptRegion'] = sum(datasets[run][team+'TestConf'] < 0.5)

            tr_objective_results_filtered[team][cost][run].loc[0,'error_rate_acceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][datasets[run][team+'TestConf'] < 0.5], tr_objective_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'TestConf'] < 0.5])
            
            
            tr_objective_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'] = 1-metrics.accuracy_score(datasets[run][team+'_Ytest'][(datasets[run][team+'TestConf'] < 0.5) & (tr_objective_results_filtered[team][cost][run].loc[0,'test_covereds'])], 
                                                                                                                    tr_objective_results_filtered[team][cost][run].loc[0,'modelonly_test_preds'][(datasets[run][team+'TestConf'] < 0.5) & (tr_objective_results_filtered[team][cost][run].loc[0,'test_covereds'])])

            tr_objective_results_filtered[team][cost][run].loc[0, 'contradictions'] = (tr_objective_results_filtered[team][cost][run].loc[0, 'modelonly_test_preds'] != datasets[run][team+'_Ybtest']).sum()


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
                                  'FILTTeamLoss': np.zeros(len(costs)),
                                  'TeamRulesCov': np.zeros(len(costs)),
                                  'FILTCov': np.zeros(len(costs)),
                                  'TeamRulesModelOnlyAcceptLoss': np.zeros(len(costs)), 
                                  'FILTModelOnlyAcceptLoss': np.zeros(len(costs)),
                                  'TeamRulesRejects': np.zeros(len(costs)), 
                                  'FILTRejects': np.zeros(len(costs)), 
                                  'TR_Contradictions': np.zeros(len(costs)), 
                                  'FILT_Contradictions': np.zeros(len(costs)),
                                  'TR_Objectives': np.zeros(len(costs)),
                                  'FILT_Objectives': np.zeros(len(costs)),
                                  'OBJTeamLoss': np.zeros(len(costs)), 
                                  'OBJCov': np.zeros(len(costs)), 
                                  'OBJ_Contradictions': np.zeros(len(costs)),
                                  'OBJ_Objectives': np.zeros(len(costs))})

    for cost in costs:

        cost = str(cost)

        TeamRulesLoss = []
        FILTLoss = []
        TeamRules_modelonly_Loss = []
        FILT_modelonly_Loss = []
        TeamRulesCov = []
        FILTCov = []
        TeamRulesRejects = []
        FILTRejects = []
        TRContradicts = []
        FILTContradicts = []
        TR_Objectives = []
        FILT_Objectives = []
        OBJ_Objectives = []
        OBJContradicts = []
        OBJLoss = []
        OBJCov = []
        OBJRejects = []
        OBJ_modelonly_Loss = []

        for run in range(numRuns):

            
            newTRLoss = asym_loss[0] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] == 0] != tr_both_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] == 0]).sum()
            newTRLoss += asym_loss[1] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] != 0] != tr_both_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] != 0]).sum()
            newTRLoss = newTRLoss/len(datasets[run][team+'_Ytest'])
            TeamRulesLoss.append(newTRLoss)
            TeamRules_modelonly_Loss.append(tr_both_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'])
            TeamRulesRejects.append(tr_both_results_filtered[team][cost][run].loc[0,'test_rejects'])
            TeamRulesCov.append(tr_both_results_filtered[team][cost][run].loc[0,'test_coverage'])
            TRContradicts.append(tr_both_results_filtered[team][cost][run].loc[0,'contradictions'])
            TR_Objectives.append((TeamRulesLoss[-1]) + (float(cost)*TRContradicts[-1])/(datasets[run].shape[0]))


            newFILTLoss = asym_loss[0] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] == 0] != tr_filter_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] == 0]).sum()
            newFILTLoss += asym_loss[1] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] != 0] != tr_filter_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] != 0]).sum()
            newFILTLoss = newFILTLoss/len(datasets[run][team+'_Ytest'])
            FILTLoss.append(newFILTLoss)
            FILT_modelonly_Loss.append(tr_filter_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'])
            FILTRejects.append(tr_filter_results_filtered[team][cost][run].loc[0,'test_rejects'])
            FILTCov.append(tr_filter_results_filtered[team][cost][run].loc[0,'test_coverage'])
            FILTContradicts.append(tr_filter_results_filtered[team][cost][run].loc[0,'contradictions'])
            FILT_Objectives.append((FILTLoss[-1]) + (float(cost)*FILTContradicts[-1])/(datasets[run].shape[0]))

            newOBJLoss = asym_loss[0] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] == 0] != tr_objective_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] == 0]).sum()
            newOBJLoss += asym_loss[1] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] != 0] != tr_objective_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] != 0]).sum()
            newOBJLoss = newOBJLoss/len(datasets[run][team+'_Ytest'])
            OBJLoss.append(newOBJLoss)
            OBJ_modelonly_Loss.append(tr_objective_results_filtered[team][cost][run].loc[0,'error_rate_ModelOnlyAcceptRegion'])
            OBJRejects.append(tr_objective_results_filtered[team][cost][run].loc[0,'test_rejects'])
            OBJCov.append(tr_objective_results_filtered[team][cost][run].loc[0,'test_coverage'])
            OBJContradicts.append(tr_objective_results_filtered[team][cost][run].loc[0,'contradictions'])
            OBJ_Objectives.append((OBJLoss[-1]) + (float(cost)*OBJContradicts[-1])/(datasets[run].shape[0]))


            

        frame = pd.DataFrame({'TeamRulesLoss': TeamRulesLoss,
                             'FILTLoss': FILTLoss,
                             'TeamRulesCov': TeamRulesCov,
                             'FILTCov': FILTCov,
                             'TRRej': TeamRulesRejects,
                             'FILTRej': FILTRejects}),
                             #'OBJLoss': OBJLoss})
        costFrame.loc[cost, 'TeamRulesTeamLoss'] = mean(TeamRulesLoss)
        costFrame.loc[cost, 'TeamRulesTeamLoss_std'] = stdev(TeamRulesLoss)
        costFrame.loc[cost, 'FILTTeamLoss'] = mean(FILTLoss)
        costFrame.loc[cost, 'FILTTeamLoss_std'] = stdev(FILTLoss)
        costFrame.loc[cost, 'TeamRulesCov'] = mean(TeamRulesCov)
        costFrame.loc[cost, 'TeamRulesRejects'] = mean(TeamRulesRejects)
        costFrame.loc[cost, 'FILTRejects'] = mean(FILTRejects)
        costFrame.loc[cost, 'FILTCov'] = mean(FILTCov)
        costFrame.loc[cost, 'TeamRulesModelOnlyAcceptLoss'] = mean(TeamRules_modelonly_Loss)
        costFrame.loc[cost, 'FILTModelOnlyAcceptLoss'] = mean(FILT_modelonly_Loss)
        costFrame.loc[cost, 'TR_Contradictions'] = mean(TRContradicts)
        costFrame.loc[cost, 'TR_Contradictions_std'] = stdev(TRContradicts)
        costFrame.loc[cost, 'FILT_Contradictions'] = mean(FILTContradicts)
        costFrame.loc[cost, 'FILT_Contradictions_std'] = stdev(FILTContradicts)
        costFrame.loc[cost, 'TR_Objective'] = mean(TR_Objectives)
        costFrame.loc[cost, 'FILT_Objective'] = mean(FILT_Objectives)
        costFrame.loc[cost, 'TR_Objective_SE'] = stdev(TR_Objectives)/math.sqrt(numRuns)
        costFrame.loc[cost, 'FILT_Objective_SE'] = stdev(FILT_Objectives)/math.sqrt(numRuns)
        costFrame.loc[cost, 'OBJTeamLoss'] = mean(OBJLoss)
        costFrame.loc[cost, 'OBJTeamLoss_std'] = stdev(OBJLoss)
        costFrame.loc[cost, 'OBJ_Contradictions'] = mean(OBJContradicts)
        costFrame.loc[cost, 'OBJ_Objective'] = mean(OBJ_Objectives)
        costFrame.loc[cost, 'OBJ_Objective_SE'] = stdev(OBJ_Objectives)/math.sqrt(numRuns)

    


    TR_loss = []
    TR_con = []
    FILT_loss = []
    FILT_con = []
    OBJ_loss = []
    OBJ_con = []

    for run in range(numRuns):
        for cost in costs:
            cost = str(cost)
            newTRLoss = asym_loss[0] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] == 0] != tr_both_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] == 0]).sum()
            newTRLoss += asym_loss[1] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] != 0] != tr_both_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] != 0]).sum()
            newTRLoss = newTRLoss/len(datasets[run][team+'_Ytest'])
            TR_loss.append(newTRLoss)
            TR_con.append(tr_both_results_filtered[team][cost][run].loc[0,'contradictions'])

            newFILTLoss = asym_loss[0] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] == 0] != tr_filter_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] == 0]).sum()
            newFILTLoss += asym_loss[1] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] != 0] != tr_filter_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] != 0]).sum()
            newFILTLoss = newFILTLoss/len(datasets[run][team+'_Ytest'])
            FILT_loss.append(newFILTLoss)
            FILT_con.append(tr_filter_results_filtered[team][cost][run].loc[0,'contradictions'])

            newOBJLoss = asym_loss[0] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] == 0] != tr_objective_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] == 0]).sum()
            newOBJLoss += asym_loss[1] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] != 0] != tr_objective_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] != 0]).sum()
            newOBJLoss = newOBJLoss/len(datasets[run][team+'_Ytest'])
            OBJ_loss.append(newOBJLoss)
            OBJ_con.append(tr_objective_results_filtered[team][cost][run].loc[0,'contradictions'])
            



    TR_loss = np.array(TR_loss)
    TR_con = np.array(TR_con)
    FILT_loss = np.array(FILT_loss)
    FILT_con = np.array(FILT_con)
    OBJ_loss = np.array(OBJ_loss)
    OBJ_con = np.array(OBJ_con)

    #dedup TRs for scatter
    l = list(zip(TR_loss, TR_con))
    z = list(zip(*[i for n, i in enumerate(l) if i not in l[:n]]))
    TR_loss = np.array(z[0])
    TR_con = np.array(z[1])

    l = list(zip(FILT_loss, FILT_con))
    z = list(zip(*[i for n, i in enumerate(l) if i not in l[:n]]))
    FILT_loss = np.array(z[0])
    FILT_con = np.array(z[1])

    l = list(zip(OBJ_loss, OBJ_con))
    z = list(zip(*[i for n, i in enumerate(l) if i not in l[:n]]))
    OBJ_loss = np.array(z[0])
    OBJ_con = np.array(z[1])
    fig.suptitle('{} Setting'.format(setting), fontsize=16)
    i=0
    for row in ax:
        if i==0:
            
            
            costFrame.sort_values(by=['Costs'], inplace=True)
            row.plot(costFrame['Costs'], costFrame['FILT_Objective'], c='red', marker='v', label = 'FILT', markersize=1)
            row.plot(costFrame['Costs'], costFrame['TR_Objective'], c='blue', marker='.', label='TeamRules', markersize=1)
            row.plot(costFrame['Costs'], costFrame['OBJ_Objective'], c='gray', marker='x', label='OBJ', markersize=1)
            row.fill_between(costFrame['Costs'], 
                       costFrame['FILT_Objective']-(costFrame['FILT_Objective_SE']),
                       costFrame['FILT_Objective']+(costFrame['FILT_Objective_SE']) ,
                      color='red', alpha=0.2)
            row.fill_between(costFrame['Costs'], 
                       costFrame['OBJ_Objective']-(costFrame['OBJ_Objective_SE']),
                       costFrame['OBJ_Objective']+(costFrame['OBJ_Objective_SE']) ,
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
            row.scatter(FILT_con, FILT_loss, c='red', marker = 'v',  alpha=0.2, label='FILT', s=6)
            row.scatter(OBJ_con, OBJ_loss, c='gray', marker = 'x',  alpha=0.2, label='OBJ', s=6)
            leg = row.legend(prop={'size': 6})
            for lh in leg.legendHandles: 
                lh.set_alpha(1)


            #col.plot(x, y)
            costFrame.sort_values(by=['FILT_Contradictions'], inplace=True)
            row.plot(costFrame['FILT_Contradictions'], costFrame['FILTTeamLoss'], marker = 'v', markersize=4, c='red', label = 'FILT')
            costFrame.sort_values(by=['OBJ_Contradictions'], inplace=True)
            row.plot(costFrame['OBJ_Contradictions'], costFrame['OBJTeamLoss'], marker = 'x', markersize=4, c='gray', label = 'OBJ')
            costFrame.sort_values(by=['TR_Contradictions'], inplace=True)
            row.plot(costFrame['TR_Contradictions'], costFrame['TeamRulesTeamLoss'], marker='.', markersize=4, c='blue', label='TeamRules')
            row.set_xlabel('# of Contradictions', fontsize=14)
            row.set_ylabel('Team Decision Loss', fontsize=14)
            row.tick_params(labelrotation=45, labelsize=10)
            #row.set_title('{} Setting'.format(setting), fontsize=15)
            
        
            
        
        i+=1
    #fig.savefig('Plots/asym_2_1_{}_{}.png'.format(data,setting), bbox_inches='tight')
    fig.savefig('Plots/ablation_det_{}_{}.png'.format(data,setting), bbox_inches='tight')


    
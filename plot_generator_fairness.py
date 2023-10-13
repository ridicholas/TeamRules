import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from sklearn import metrics
from scipy.stats import ttest_ind
from statistics import mean, stdev
import math

types = ['TL', 'TDL']

numRuns = 10 #adjust this depending on how many runs of results were produced

rule_len = 4
setting_type = 'learned'
dataset = 'heart'
path = f'{dataset}_contradiction_results_{setting_type}_fair1_len{rule_len}/'

asym_loss = [1,1]
data = path.split('_')[0]
costs = [0, 0.01, 0.05, 0.1, 0.2,
                     0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

sensitive='sex_male'

for whichType in types:
    #costs = [0, 0.01, 0.05]
    hyrs_reconcile = False
    tr_conf = 'opt'
    hyrs_conf = 0
    tr_optimizer = True

    teams = ['team1', 'team2', 'team3']
    #teams = ['team2']
    team_infos = []
    datasets = []
    tr_results_filtered = {}
    val_tr_results_filtered = {}
    hyrs_results_filtered = {}
    val_hyrs_results_filtered = {}
    brs_results = {'team1': [], 'team2': [], 'team3': []}
    hyrs_recon_results_filtered = {}

    for team in teams:
        
        tr_results_filtered[team] = {}
        hyrs_results_filtered[team] = {}
        hyrs_recon_results_filtered[team] = {}
        val_tr_results_filtered[team] = {}
        val_hyrs_results_filtered[team] = {}

    for team in teams:
        for cost in costs:
            tr_results_filtered[team][str(cost)] = []
            hyrs_results_filtered[team][str(cost)] = []
            hyrs_recon_results_filtered[team][str(cost)] = []
            val_tr_results_filtered[team][str(cost)] = []
            val_hyrs_results_filtered[team][str(cost)] = []
            
    replacement_tracker = {'team1': {}, 'team2' : {}, 'team3': {}, 'team4': {}}
    start_info = pd.read_pickle(path + 'start_info.pkl')
    for i in range(0, numRuns):

        datasets.append(pd.read_pickle(path + 'dataset_run{}.pkl'.format(i)))
        for team in teams:
            opt_cost_dict = {}
            tr_conf_dict = {}
            for cost in costs:

                
                cost = str(cost)
                if cost not in replacement_tracker[team].keys():
                    replacement_tracker[team][cost] = 0
                if hyrs_reconcile:
                    hyrs_cost = cost
                else:
                    hyrs_cost = '0'


                if tr_optimizer:
                    #tr filtered validation
                    val_tr_results_filtered[team][cost].append(pd.read_pickle(path + 'val_cost_{}_'.format(cost) + team + '_tr_filtered_run{}.pkl'.format(i)).sort_values(by=['objective', 'contradicts']))
                    val_tr_obj = val_tr_results_filtered[team][cost][-1][val_tr_results_filtered[team][cost][-1]['mental_conf'] == 0].loc[0,'objective']
                    val_tr_mopreds = val_tr_results_filtered[team][cost][-1][val_tr_results_filtered[team][cost][-1]['mental_conf'] == 0].loc[0,'modelonly_val_preds']
                    tr_conf = val_tr_results_filtered[team][cost][-1].reset_index().loc[0, 'mental_conf']

                    
                    best_val_obj = 1000000
                    for temp_cost in costs:
                        temp_cost = str(temp_cost)
                        temp_data = pd.read_pickle(path + 'val_cost_{}_'.format(temp_cost) + team + '_tr_filtered_run{}.pkl'.format(i))
                        temp_data['new_objective'] = temp_data.loc[:,'val_error'] + float(cost)*temp_data.loc[:,'contradicts']/len(temp_data.loc[0,'humanified_val_preds'])
                        temp_data = temp_data.sort_values(by=['new_objective', 'contradicts']).reset_index(drop=True)
                        temp_obj = temp_data.loc[0, 'new_objective']
                        if temp_obj < best_val_obj:
                            best_val_obj = temp_obj
                            opt_cost_dict[cost] = temp_cost
                            tr_conf_dict[cost] = temp_data.loc[0, 'mental_conf']
                    
                if whichType == 'TDL' and False:
                    opt_cost_dict[cost] = cost
                    tr_conf_dict[cost] = tr_conf
                    






                    

                #tr filtered
                tr_results_filtered[team][cost].append(pd.read_pickle(path + 'cost_{}_'.format(opt_cost_dict[cost]) + team + '_tr_filtered_run{}.pkl'.format(i)).sort_values(by='test_error'))
                tr_results_filtered[team][cost][-1] = tr_results_filtered[team][cost][-1][tr_results_filtered[team][cost][-1]['mental_conf'] == tr_conf_dict[cost]].reset_index()
                tr_results_filtered[team][cost][-1]['new_objective'] = tr_results_filtered[team][cost][-1]['test_error'] + float(cost)*tr_results_filtered[team][cost][-1]['contradicts']/datasets[i].shape[0]
                tr_results_filtered[team][cost]

                

                #hyrs filtered
                hyrs_results_filtered[team][cost].append(pd.read_pickle(path + 'cost_{}_'.format(hyrs_cost)+ team + '_hyrs_filtered_run{}.pkl'.format(i)).sort_values(by='test_error'))
                hyrs_results_filtered[team][cost][-1] = hyrs_results_filtered[team][cost][-1][hyrs_results_filtered[team][cost][-1]['mental_conf'] == hyrs_conf].reset_index()

                #hyrs recon filtered
                hyrs_recon_results_filtered[team][cost].append(pd.read_pickle(path + 'cost_{}_'.format(cost)+ team + '_hyrs_filtered_run{}.pkl'.format(i)).sort_values(by='test_error'))
                hyrs_recon_results_filtered[team][cost][-1] = hyrs_recon_results_filtered[team][cost][-1][hyrs_recon_results_filtered[team][cost][-1]['mental_conf'] == hyrs_conf].reset_index()

                #hyrs filtered validation
                val_hyrs_results_filtered[team][cost].append(pd.read_pickle(path + 'val_cost_{}_'.format(hyrs_cost)+ team + '_hyrs_filtered_run{}.pkl'.format(i)).sort_values(by='val_error'))
                val_hyrs_results_filtered[team][cost][-1] = val_hyrs_results_filtered[team][cost][-1][val_hyrs_results_filtered[team][cost][-1]['mental_conf'] == hyrs_conf].reset_index()
                val_hyrs_obj = val_hyrs_results_filtered[team][cost][-1].loc[0,'objective']
                val_hyrs_mopreds = val_hyrs_results_filtered[team][cost][-1].loc[0,'modelonly_val_preds']

                
            
                if cost == '0':
                    brs_results[team].append(pd.read_pickle(path + team + '_brs_run{}.pkl'.format(i)).sort_values(by='test_error_brs'))
                    brs_val_obj = brs_results[team][-1].loc[0, 'val_objective']
                    brs_val_contradicts = brs_results[team][-1].loc[0, 'val_contradicts']
                
                brs_val_obj_w_cost = brs_val_obj + float(cost)*brs_val_contradicts/datasets[i].shape[0]
                
                if ((val_hyrs_obj == val_tr_obj) & ((val_hyrs_mopreds == val_tr_mopreds).sum() == len(val_hyrs_mopreds))) or val_tr_obj == brs_val_obj_w_cost:
                    replacement_tracker[team][cost] += 1



    for t in replacement_tracker.keys():
        for c in replacement_tracker[t].keys():
            replacement_tracker[t][c] = replacement_tracker[t][c]/numRuns

    teams = []
    for t in start_info.sort_values(by='human accept region train acc').index:
        teams.append('team{}'.format(t))

    #teams = ['team2']
    settings = ['Calibrated', 'SlightlyMiscalibrated', 'SignificantlyMiscalibrated']
    #settings = ['Neutral']
    for whichTeam in range(len(settings)):
        fig = plt.figure(figsize=(3, 2), dpi=200)
        #fig.subplots_adjust(bottom=0.15, wspace=.4)
        team = teams[whichTeam]
        setting = settings[whichTeam]
        costFrame = pd.DataFrame(index=[str(x) for x in costs], data={'Costs': [str(x) for x in costs], 
                                    'TeamRulesTeamLoss': np.zeros(len(costs)), 
                                    'HyRSTeamLoss': np.zeros(len(costs)), 
                                    'R_HyRSTeamLoss': np.zeros(len(costs)),                        
                                    'TR_Contradictions': np.zeros(len(costs)), 
                                    'HyRS_Contradictions': np.zeros(len(costs)),
                                    'R_HyRS_Contradictions': np.zeros(len(costs)),
                                    'TR_Objectives': np.zeros(len(costs)),
                                    'HyRS_Objectives': np.zeros(len(costs)),
                                    'R_HyRS_Objectives': np.zeros(len(costs)),
                                    'BRSTeamLoss': np.zeros(len(costs)), 
                                    'BRS_Contradictions': np.zeros(len(costs)),
                                    'BRS_Objectives': np.zeros(len(costs))})

        TR_loss = []
        TR_con = []
        HyRS_loss = []
        HyRS_con = []
        R_HyRS_loss = []
        R_HyRS_con = []
        BRS_loss = []
        BRS_con = []


        for cost in costs:

            cost = str(cost)

            TeamRulesLoss = []
            HyRSLoss = []
            R_HyRSLoss = []
            TeamRules_modelonly_Loss = []
            HyRS_modelonly_Loss = []
            TeamRulesCov = []
            HyRSCov = []
            R_HyRSCov = []
            TeamRulesRejects = []
            HyRSRejects = []
            TRContradicts = []
            HyRSContradicts = []
            R_HyRSContradicts = []
            TR_Objectives = []
            HyRS_Objectives = []
            R_HyRS_Objectives = []
            BRS_Objectives = []
            BRSContradicts = []
            BRSLoss = []
            BRSCov = []
            Human = []
            TR_disparity = []
            BRS_disparity = []
            HyRS_disparity = []
            R_HyRS_disparity = []
            for run in range(numRuns):

                
                TeamRulesLoss.append(tr_results_filtered[team][cost][run].loc[0,'test_error'])
                TRContradicts.append(tr_results_filtered[team][cost][run].loc[0,'contradicts'])
                TR_Objectives.append(TeamRulesLoss[-1] + float(cost)*TRContradicts[-1]/(datasets[run].shape[0]))
                TR_disparity.append(tr_results_filtered[team][cost][run].loc[0,'fairness_score'])
                #TR_Objectives.append(tr_results_filtered[team][cost][run].loc[0,'objective'])


                
                HyRSLoss.append(hyrs_results_filtered[team][cost][run].loc[0,'test_error'])
                HyRSContradicts.append(hyrs_results_filtered[team][cost][run].loc[0,'contradicts'])
                if hyrs_cost == '0':
                    HyRS_Objectives.append(hyrs_results_filtered[team][cost][run].loc[0,'objective'] + (float(cost)*HyRSContradicts[-1])/(datasets[run].shape[0]))
                else:
                    HyRS_Objectives.append(hyrs_results_filtered[team][cost][run].loc[0,'objective'])
                HyRS_disparity.append(hyrs_results_filtered[team][cost][run].loc[0,'fairness_score'])


                R_HyRSLoss.append(hyrs_recon_results_filtered[team][cost][run].loc[0,'test_error'])
                R_HyRSContradicts.append(hyrs_recon_results_filtered[team][cost][run].loc[0,'contradicts'])

                R_HyRS_Objectives.append(hyrs_recon_results_filtered[team][cost][run].loc[0,'objective'])
                R_HyRS_disparity.append(hyrs_recon_results_filtered[team][cost][run].loc[0,'fairness_score'])


                BRSLoss.append(brs_results[team][run].loc[0,'test_objective'])
                BRSContradicts.append(brs_results[team][run].loc[0,'test_contradicts'])
                BRS_Objectives.append(brs_results[team][run].loc[0,'test_objective'] + (float(cost)*BRSContradicts[-1])/(datasets[run].shape[0]))
                BRS_disparity.append(brs_results[team][run].loc[0,'test_fairness'])

                Human.append(1-metrics.accuracy_score(datasets[run][f'{team}_Ybtest'], datasets[run][f'{team}_Ytest']))
                

            
            costFrame.loc[cost, 'TeamRulesTeamLoss'] = mean(TeamRulesLoss)
            costFrame.loc[cost, 'TeamRulesTeamLoss_std'] = stdev(TeamRulesLoss)/math.sqrt(numRuns)
            costFrame.loc[cost, 'HyRSTeamLoss'] = mean(HyRSLoss)
            costFrame.loc[cost, 'HyRSTeamLoss_std'] = stdev(HyRSLoss)/math.sqrt(numRuns)
            costFrame.loc[cost, 'R_HyRSTeamLoss'] = mean(R_HyRSLoss)
            costFrame.loc[cost, 'R_HyRSTeamLoss_std'] = stdev(R_HyRSLoss)/math.sqrt(numRuns)
            costFrame.loc[cost, 'TR_Contradictions'] = mean(TRContradicts)
            costFrame.loc[cost, 'TR_Contradictions_std'] = stdev(TRContradicts)/math.sqrt(numRuns)
            costFrame.loc[cost, 'HyRS_Contradictions'] = mean(HyRSContradicts)
            costFrame.loc[cost, 'HyRS_Contradictions_std'] = stdev(HyRSContradicts)/math.sqrt(numRuns)
            costFrame.loc[cost, 'R_HyRS_Contradictions'] = mean(R_HyRSContradicts)
            costFrame.loc[cost, 'R_HyRS_Contradictions_std'] = stdev(R_HyRSContradicts)/math.sqrt(numRuns)
            costFrame.loc[cost, 'TR_Objective'] = mean(TR_Objectives)
            costFrame.loc[cost, 'HyRS_Objective'] = mean(HyRS_Objectives)
            costFrame.loc[cost, 'R_HyRS_Objective'] = mean(R_HyRS_Objectives)
            costFrame.loc[cost, 'TR_Objective_SE'] = stdev(TR_Objectives)/math.sqrt(numRuns)
            costFrame.loc[cost, 'HyRS_Objective_SE'] = stdev(HyRS_Objectives)/math.sqrt(numRuns)
            costFrame.loc[cost, 'R_HyRS_Objective_SE'] = stdev(R_HyRS_Objectives)/math.sqrt(numRuns)
            costFrame.loc[cost, 'BRSTeamLoss'] = mean(BRSLoss)
            costFrame.loc[cost, 'BRSTeamLoss_std'] = stdev(BRSLoss)/math.sqrt(numRuns)
            costFrame.loc[cost, 'BRS_Contradictions'] = mean(BRSContradicts)
            costFrame.loc[cost, 'BRS_Contradictions_std'] = stdev(BRSContradicts)/math.sqrt(numRuns)
            costFrame.loc[cost, 'BRS_Objective'] = mean(BRS_Objectives)
            costFrame.loc[cost, 'BRS_Disparity'] = mean(BRS_disparity)
            costFrame.loc[cost, 'BRS_Disparity_SE'] = stdev(BRS_disparity)/math.sqrt(numRuns)
            costFrame.loc[cost, 'TR_Disparity'] = mean(TR_disparity)
            costFrame.loc[cost, 'TR_Disparity_SE'] = stdev(TR_disparity)/math.sqrt(numRuns)
            costFrame.loc[cost, 'HyRS_Disparity'] = mean(HyRS_disparity)
            costFrame.loc[cost, 'HyRS_Disparity_SE'] = stdev(HyRS_disparity)/math.sqrt(numRuns)
            costFrame.loc[cost, 'R_HyRS_Disparity'] = mean(R_HyRS_disparity)
            costFrame.loc[cost, 'R_HyRS_Disparity_SE'] = stdev(R_HyRS_disparity)/math.sqrt(numRuns)
            costFrame.loc[cost, 'BRS_Objective_SE'] = stdev(BRS_Objectives)/math.sqrt(numRuns)
            costFrame.loc[cost, 'Human Only'] = mean(Human)
            costFrame.loc[cost, 'Human Only SE'] = stdev(Human)/math.sqrt(numRuns)

        

            '''
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
            '''
        
            TR_loss += (TeamRulesLoss)
            TR_con += (TRContradicts)
            HyRS_loss += (HyRSLoss)
            HyRS_con += (HyRSContradicts)
            R_HyRS_loss += (R_HyRSLoss)
            R_HyRS_con += (R_HyRSContradicts)
            BRS_loss += (BRSLoss)
            BRS_con += (BRSContradicts)

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
        #fig.suptitle('{} {} Setting'.format(data, setting), fontsize=16)
        
        
            
        color_dict = {'TR': '#348ABD', 'HYRS': '#E24A33', 'BRS':'#988ED5', 'Human': 'darkgray', 'HYRSRecon': '#8EBA42'}
        
        if whichType == 'TL':
            costFrame.sort_values(by=['Costs'], inplace=True)
            plt.plot(costFrame['Costs'], costFrame['HyRS_Objective'], marker = 'v', c=color_dict['HYRS'], label = 'HyRS', markersize=1.8, linewidth=0.9)
            plt.plot(costFrame['Costs'], costFrame['R_HyRS_Objective'], marker = 'x', c=color_dict['HYRSRecon'], label = 'Task, Decision & Reconciliation Optimized', markersize=1.8, linewidth=0.9)
            plt.plot(costFrame['Costs'], costFrame['TR_Objective'], marker = '.', c=color_dict['TR'], label='TeamRules', markersize=1.8, linewidth=0.9)
            plt.plot(costFrame['Costs'], costFrame['BRS_Objective'], marker = 's', c=color_dict['BRS'], label='BRS', markersize=1.8, linewidth=0.9)
            plt.axhline(costFrame['Human Only'][0], c = color_dict['Human'], markersize=1, label='Human Alone', ls='--', alpha=0.5)
            plt.fill_between(costFrame['Costs'], 
                        costFrame['Human Only']-(costFrame['Human Only SE']),
                        costFrame['Human Only']+(costFrame['Human Only SE']) ,
                        color=color_dict['Human'], alpha=0.2)
            plt.fill_between(costFrame['Costs'], 
                        costFrame['HyRS_Objective']-(costFrame['HyRS_Objective_SE']),
                        costFrame['HyRS_Objective']+(costFrame['HyRS_Objective_SE']) ,
                        color=color_dict['HYRS'], alpha=0.2)
            plt.fill_between(costFrame['Costs'], 
                        costFrame['BRS_Objective']-(costFrame['BRS_Objective_SE']),
                        costFrame['BRS_Objective']+(costFrame['BRS_Objective_SE']) ,
                        color=color_dict['BRS'], alpha=0.2)
            plt.fill_between(costFrame['Costs'], 
                        costFrame['R_HyRS_Objective']-(costFrame['R_HyRS_Objective_SE']),
                        costFrame['R_HyRS_Objective']+(costFrame['R_HyRS_Objective_SE']) ,
                        color=color_dict['HYRSRecon'], alpha=0.2)
            plt.fill_between(costFrame['Costs'], 
                        costFrame['TR_Objective']-(costFrame['TR_Objective_SE']),
                        costFrame['TR_Objective']+(costFrame['TR_Objective_SE']),
                        color=color_dict['TR'], alpha=0.2)
            plt.xlabel('Reconciliation Cost', fontsize=12)
            plt.ylabel('Total Team Loss', fontsize=12)
            plt.tick_params(labelrotation=45, labelsize=10)
            #plt.title('{} Setting'.format(setting), fontsize=15)
            plt.legend(prop={'size': 5})
            plt.grid('on', linestyle='dotted', linewidth=0.2, color='black')

            fig.savefig(f'Plots/fairness_TL_{setting_type}_len{rule_len}_{data}_{setting}.png', bbox_inches='tight')

            plt.clf()
        else:
            cost='0'
            costFrame = costFrame.loc[costFrame['Costs'] == '0']
            costFrame.sort_values(by=['HyRS_Disparity'], inplace=True)
            plt.plot(costFrame['HyRS_Disparity'], costFrame['HyRSTeamLoss'], marker = 'v', c=color_dict['HYRS'], label = 'HyRS', markersize=1.8, linewidth=0.9)
            costFrame.sort_values(by=['R_HyRS_Disparity'], inplace=True)            
            plt.plot(costFrame['R_HyRS_Disparity'], costFrame['R_HyRSTeamLoss'], marker = 'x', c=color_dict['HYRSRecon'], label = 'Task, Decision & Reconciliation Optimized', markersize=1.8, linewidth=0.9)
            costFrame.sort_values(by=['TR_Disparity'], inplace=True)
            plt.plot(costFrame['TR_Disparity'], costFrame['TeamRulesTeamLoss'],marker = '.', c=color_dict['TR'], label='TeamRules', markersize=1.8, linewidth=0.9)
            costFrame.sort_values(by=['BRS_Disparity'], inplace=True)
            plt.plot(costFrame['BRS_Disparity'], costFrame['BRSTeamLoss'],marker = 's', c=color_dict['BRS'], label='BRS', markersize=1.8, linewidth=0.9)
            plt.fill_between(np.linspace(costFrame.loc[cost, 'BRS_Disparity']-costFrame.loc[cost, 'BRS_Disparity_SE'], 
                                        costFrame.loc[cost, 'BRS_Disparity']+costFrame.loc[cost, 'BRS_Disparity_SE'], 
                                        50), costFrame.loc[cost, 'BRSTeamLoss'] - costFrame.loc[cost, 'BRSTeamLoss_std'], costFrame.loc[cost, 'BRSTeamLoss'] + costFrame.loc[cost, 'BRSTeamLoss_std'], color=color_dict['BRS'], alpha=0.2)
            for cost in costFrame.Costs:
                plt.fill_between(np.linspace(costFrame.loc[cost, 'TR_Disparity']-costFrame.loc[cost, 'TR_Disparity_SE'], 
                                            costFrame.loc[cost, 'TR_Disparity']+costFrame.loc[cost, 'TR_Disparity_SE'], 
                                            50), costFrame.loc[cost, 'TeamRulesTeamLoss'] - costFrame.loc[cost, 'TeamRulesTeamLoss_std'], costFrame.loc[cost, 'TeamRulesTeamLoss'] + costFrame.loc[cost, 'TeamRulesTeamLoss_std'], color=color_dict['TR'], alpha=0.2)         
            plt.fill_between(np.linspace(costFrame.loc[cost, 'HyRS_Disparity']-costFrame.loc[cost, 'HyRS_Disparity_SE'], 
                                        costFrame.loc[cost, 'HyRS_Disparity']+costFrame.loc[cost, 'HyRS_Disparity_SE'], 
                                        50), costFrame.loc[cost, 'HyRSTeamLoss'] - costFrame.loc[cost, 'HyRSTeamLoss_std'], costFrame.loc[cost, 'HyRSTeamLoss'] + costFrame.loc[cost, 'HyRSTeamLoss_std'], color=color_dict['HYRS'], alpha=0.2)            
            plt.fill_between(np.linspace(costFrame.loc[cost, 'R_HyRS_Disparity']-costFrame.loc[cost, 'R_HyRS_Disparity_SE'], 
                                        costFrame.loc[cost, 'R_HyRS_Disparity']+costFrame.loc[cost, 'R_HyRS_Disparity_SE'], 
                                        50), costFrame.loc[cost, 'R_HyRSTeamLoss'] - costFrame.loc[cost, 'R_HyRSTeamLoss_std'], costFrame.loc[cost, 'R_HyRSTeamLoss'] + costFrame.loc[cost, 'R_HyRSTeamLoss_std'], color=color_dict['HYRSrecon'], alpha=0.2)               
            plt.xlabel('Mean Disparity', fontsize=12)
            plt.ylabel('TDL', fontsize=12)
            plt.tick_params(labelrotation=45, labelsize=10)
            #plt.title('{} Setting'.format(setting), fontsize=15)
            plt.legend(prop={'size': 5})
            plt.grid('on', linestyle='dotted', linewidth=0.2, color='black')

            fig.savefig(f'Plots/fairness_TDL_{setting_type}_len{rule_len}_{data}_{setting}.png', bbox_inches='tight')
                    
                
                    
                
                
            #fig.savefig('Plots/asym_2_1_{}_{}.png'.format(data,setting), bbox_inches='tight')
    


    
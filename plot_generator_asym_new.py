import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from sklearn import metrics
from scipy.stats import ttest_ind
from statistics import mean, stdev
import math

types = ['TL', 'TDL']

numRuns = 3 #adjust this depending on how many runs of results were produced

rule_len = 4
setting_type = 'learned'
dataset = 'heart'
asym_loss = [3,1]
path = f'{dataset}_contradiction_results_{setting_type}_asym{asym_loss[0]}{asym_loss[1]}_len{rule_len}/'

data = path.split('_')[0]
costs = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.2, 1.5, 2, 2.5]

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

    for team in teams:
        
        tr_results_filtered[team] = {}
        hyrs_results_filtered[team] = {}
        val_tr_results_filtered[team] = {}
        val_hyrs_results_filtered[team] = {}

    for team in teams:
        for cost in costs:
            tr_results_filtered[team][str(cost)] = []
            hyrs_results_filtered[team][str(cost)] = []
            val_tr_results_filtered[team][str(cost)] = []
            val_hyrs_results_filtered[team][str(cost)] = []
            

    start_info = pd.read_pickle(path + 'start_info.pkl')
    for i in range(0, numRuns):
        datasets.append(pd.read_pickle(path + 'dataset_run{}.pkl'.format(i)))
        for team in teams:
            opt_cost_dict = {}
            tr_conf_dict = {}
            for cost in costs:
                
                cost = str(cost)
                if hyrs_reconcile:
                    hyrs_cost = cost
                else:
                    hyrs_cost = '0'


                if tr_optimizer:
                    #tr filtered validation
                    val_tr_results_filtered[team][cost].append(pd.read_pickle(path + 'val_cost_{}_'.format(cost) + team + '_tr_filtered_run{}.pkl'.format(i)).sort_values(by=['objective', 'contradicts']))
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

                #hyrs filtered validation
                val_hyrs_results_filtered[team][cost].append(pd.read_pickle(path + 'val_cost_{}_'.format(hyrs_cost)+ team + '_hyrs_filtered_run{}.pkl'.format(i)).sort_values(by='val_error'))
                val_hyrs_results_filtered[team][cost][-1] = val_hyrs_results_filtered[team][cost][-1][val_hyrs_results_filtered[team][cost][-1]['mental_conf'] == hyrs_conf].reset_index()
                
                if cost == '0':
                    brs_results[team].append(pd.read_pickle(path + team + '_brs_run{}.pkl'.format(i)).sort_values(by='test_error_brs'))            





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
                                    'TR_Contradictions': np.zeros(len(costs)), 
                                    'HyRS_Contradictions': np.zeros(len(costs)),
                                    'TR_Objectives': np.zeros(len(costs)),
                                    'HyRS_Objectives': np.zeros(len(costs)),
                                    'BRSTeamLoss': np.zeros(len(costs)), 
                                    'BRS_Contradictions': np.zeros(len(costs)),
                                    'BRS_Objectives': np.zeros(len(costs))})

        TR_loss = []
        TR_con = []
        HyRS_loss = []
        HyRS_con = []
        BRS_loss = []
        BRS_con = []

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
            Human = []
            for run in range(numRuns):
                
                whicher = 0
                error = (tr_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run]['team1_Ytest'] == whicher] != datasets[run][f'{team}_Ytest'][datasets[run]['team1_Ytest'] == whicher]).sum() * asym_loss[np.abs(whicher-1)]
                #error += (tr_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run]['team1_Ytest'] == 0] != datasets[run][f'{team}_Ytest'][datasets[run]['team1_Ytest'] == 0]).sum() * asym_loss[1]
                #TeamRulesLoss.append(error/len(datasets[run][f'{team}_Ytest']))
                TeamRulesLoss.append(error/np.array([datasets[run]['team1_Ytest'] == whicher]).sum())
                TRContradicts.append((tr_results_filtered[team][cost][run].loc[0,'modelonly_test_preds'][datasets[run]['team1_Ytest'] == whicher] != datasets[run][f'{team}_Ybtest'][datasets[run]['team1_Ytest'] == whicher]).sum())
                TR_Objectives.append(TeamRulesLoss[-1] + float(cost)*TRContradicts[-1]/np.array([datasets[run]['team1_Ytest'] == whicher]).sum())
                #TR_Objectives.append(tr_results_filtered[team][cost][run].loc[0,'objective'])


                error = (hyrs_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run]['team1_Ytest'] == whicher] != datasets[run][f'{team}_Ytest'][datasets[run]['team1_Ytest'] == whicher]).sum() * asym_loss[np.abs(whicher-1)]
                #error += (hyrs_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run]['team1_Ytest'] == 0] != datasets[run][f'{team}_Ytest'][datasets[run]['team1_Ytest'] == 0]).sum() * asym_loss[1]
                #HyRSLoss.append(error/len(datasets[run][f'{team}_Ytest']))
                HyRSLoss.append(error/np.array([datasets[run]['team1_Ytest'] == whicher]).sum())
                HyRSContradicts.append((hyrs_results_filtered[team][cost][run].loc[0,'modelonly_test_preds'][datasets[run]['team1_Ytest'] == whicher] != datasets[run][f'{team}_Ybtest'][datasets[run]['team1_Ytest'] == whicher]).sum())
                if hyrs_cost == '0':
                    HyRS_Objectives.append(HyRSLoss[-1] + float(cost)*HyRSContradicts[-1]/np.array([datasets[run]['team1_Ytest'] == whicher]).sum())
                else:
                    HyRS_Objectives.append(hyrs_results_filtered[team][cost][run].loc[0,'objective'])
                
                error = (brs_results[team][run].loc[0,'team_test_preds'][datasets[run]['team1_Ytest'] == whicher] != datasets[run][f'{team}_Ytest'][datasets[run]['team1_Ytest'] == whicher]).sum() * asym_loss[np.abs(whicher-1)]
                #error += (brs_results[team][run].loc[0,'team_test_preds'][datasets[run]['team1_Ytest'] == 0] != datasets[run][f'{team}_Ytest'][datasets[run]['team1_Ytest'] == 0]).sum() * asym_loss[1]
                #BRSLoss.append(error/len(datasets[run][f'{team}_Ytest']))
                BRSLoss.append(error/np.array([datasets[run]['team1_Ytest'] == whicher]).sum())
                BRSContradicts.append((brs_results[team][run].loc[0,'modelonly_test_preds'][datasets[run]['team1_Ytest'] == whicher] != datasets[run][f'{team}_Ybtest'][datasets[run]['team1_Ytest'] == whicher]).sum())
                BRS_Objectives.append(BRSLoss[-1] + (float(cost)*BRSContradicts[-1])/np.array([datasets[run]['team1_Ytest'] == whicher]).sum())

                asymCosts = datasets[run][f'{team}_Ytest'].replace({0: asym_loss[1], 1: asym_loss[0]})
                Human.append(((np.abs(datasets[run][f'{team}_Ybtest'][datasets[run]['team1_Ytest'] == whicher] - datasets[run][f'{team}_Ytest'][datasets[run]['team1_Ytest'] == whicher]))*asymCosts[datasets[run]['team1_Ytest'] == whicher].values).sum()/np.array([datasets[run]['team1_Ytest'] == whicher]).sum())
                

            
            costFrame.loc[cost, 'TeamRulesTeamLoss'] = mean(TeamRulesLoss)
            costFrame.loc[cost, 'TeamRulesTeamLoss_std'] = stdev(TeamRulesLoss)/math.sqrt(numRuns)
            costFrame.loc[cost, 'HyRSTeamLoss'] = mean(HyRSLoss)
            costFrame.loc[cost, 'HyRSTeamLoss_std'] = stdev(HyRSLoss)/math.sqrt(numRuns)
            costFrame.loc[cost, 'TR_Contradictions'] = mean(TRContradicts)
            costFrame.loc[cost, 'TR_Contradictions_std'] = stdev(TRContradicts)/math.sqrt(numRuns)
            costFrame.loc[cost, 'HyRS_Contradictions'] = mean(HyRSContradicts)
            costFrame.loc[cost, 'HyRS_Contradictions_std'] = stdev(HyRSContradicts)/math.sqrt(numRuns)
            costFrame.loc[cost, 'TR_Objective'] = mean(TR_Objectives)
            costFrame.loc[cost, 'HyRS_Objective'] = mean(HyRS_Objectives)
            costFrame.loc[cost, 'TR_Objective_SE'] = stdev(TR_Objectives)/math.sqrt(numRuns)
            costFrame.loc[cost, 'HyRS_Objective_SE'] = stdev(HyRS_Objectives)/math.sqrt(numRuns)
            costFrame.loc[cost, 'BRSTeamLoss'] = mean(BRSLoss)
            costFrame.loc[cost, 'BRSTeamLoss_std'] = stdev(BRSLoss)/math.sqrt(numRuns)
            costFrame.loc[cost, 'BRS_Contradictions'] = mean(BRSContradicts)
            costFrame.loc[cost, 'BRS_Contradictions_std'] = stdev(BRSContradicts)/math.sqrt(numRuns)
            costFrame.loc[cost, 'BRS_Objective'] = mean(BRS_Objectives)
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
                newTRLoss = asym_loss[np.abs(whicher-1)] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] == 0] != tr_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] == 0]).sum()
                newTRLoss += asym_loss[1] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] != 0] != tr_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] != 0]).sum()
                newTRLoss = newTRLoss/len(datasets[run][team+'_Ytest'])
                TR_loss.append(newTRLoss)
                TR_con.append(tr_results_filtered[team][cost][run].loc[0,'contradictions'])
                newHYRSLoss = asym_loss[np.abs(whicher-1)] * (datasets[run][team+'_Ytest'][datasets[run][team+'_Ytest'] == 0] != hyrs_results_filtered[team][cost][run].loc[0,'humanified_test_preds'][datasets[run][team+'_Ytest'] == 0]).sum()
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
            #BRS_loss += (BRSLoss)
            #BRS_con += (BRSContradicts)

        #dedup TRs for scatter
        l = list(zip(TR_loss, TR_con))
        z = list(zip(*[i for n, i in enumerate(l) if i not in l[:n]]))
        TR_loss = np.array(z[0])
        TR_con = np.array(z[1])

        l = list(zip(HyRS_loss, HyRS_con))
        z = list(zip(*[i for n, i in enumerate(l) if i not in l[:n]]))
        HyRS_loss = np.array(z[0])
        HyRS_con = np.array(z[1])

        #l = list(zip(BRS_loss, BRS_con))
        #z = list(zip(*[i for n, i in enumerate(l) if i not in l[:n]]))
        #BRS_loss = np.array(z[0])
        #BRS_con = np.array(z[1])
        #fig.suptitle('{} {} Setting'.format(data, setting), fontsize=16)
        
        
            
        color_dict = {'TR': '#348ABD', 'HYRS': '#E24A33', 'BRS':'#988ED5', 'Human': 'darkgray'}
        
        if whichType == 'TL':
            costFrame.sort_values(by=['Costs'], inplace=True)
            plt.plot(costFrame['Costs'], costFrame['HyRS_Objective'], marker = 'v', c=color_dict['HYRS'], label = 'HyRS', markersize=1.8, linewidth=0.9)
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
                        costFrame['TR_Objective']-(costFrame['TR_Objective_SE']),
                        costFrame['TR_Objective']+(costFrame['TR_Objective_SE']) ,
                        color=color_dict['TR'], alpha=0.2)
            plt.xlabel('Reconciliation Cost', fontsize=12)
            plt.ylabel('Total Team Loss', fontsize=12)
            plt.tick_params(labelrotation=45, labelsize=10)
            #plt.title('{} Setting'.format(setting), fontsize=15)
            plt.legend(prop={'size': 5})
            plt.grid('on', linestyle='dotted', linewidth=0.2, color='black')

            fig.savefig(f'Plots/loss{whicher}_TL_{setting_type}_len{rule_len}_{data}_{setting}_asym{asym_loss[0]}{asym_loss[1]}.png', bbox_inches='tight')

            plt.clf()
        else:
            fig = plt.figure(figsize=(3, 2), dpi=200)
            plt.grid('on', linestyle='dotted', linewidth=0.2, color='black')




            #plt.scatter(TR_con, TR_loss, c=color_dict['TR'], marker = '.',  alpha=0.4, label='TeamRules', s=6)
            #plt.scatter(HyRS_con, HyRS_loss, c=color_dict['HYRS'], marker = 'v',  alpha=0.4, label='HyRS', s=6)
            #plt.scatter(BRS_con, BRS_loss, c=color_dict['BRS'], marker = 's',  alpha=0.4, label='BRS', s=6)
            #leg = plt.legend(prop={'size': 6})
            #for lh in leg.legendHandles: 
            #    lh.set_alpha(1)


            #col.plot(x, y)
            
            costFrame.sort_values(by=['HyRS_Contradictions'], inplace=True)
            plt.plot(costFrame['HyRS_Contradictions'], costFrame['HyRSTeamLoss'], marker = 'v', markersize=2, c=color_dict['HYRS'], label = 'HyRS', linewidth=0.9)
            cost = '0'
            plt.fill_between(np.linspace(costFrame.loc[cost, 'HyRS_Contradictions']-costFrame.loc[cost, 'HyRS_Contradictions_std'], 
                                        costFrame.loc[cost, 'HyRS_Contradictions']+costFrame.loc[cost, 'HyRS_Contradictions_std'], 
                                        50), costFrame.loc[cost, 'HyRSTeamLoss'] - costFrame.loc[cost, 'HyRSTeamLoss_std'], costFrame.loc[cost, 'HyRSTeamLoss'] + costFrame.loc[cost, 'HyRSTeamLoss_std'], color=color_dict['HYRS'], alpha=0.2)
            #costFrame.sort_values(by=['BRS_Contradictions'], inplace=True)
            #plt.plot(costFrame['BRS_Contradictions'], costFrame['BRSTeamLoss'], marker = 's', markersize=2, c=color_dict['BRS'], label = 'BRS', linewidth=0.9)
            #plt.fill_between(np.linspace(costFrame.loc[cost, 'BRS_Contradictions']-costFrame.loc[cost, 'BRS_Contradictions_std'], 
            #                            costFrame.loc[cost, 'BRS_Contradictions']+costFrame.loc[cost, 'BRS_Contradictions_std'], 
            #                            50), costFrame.loc[cost, 'BRSTeamLoss'] - costFrame.loc[cost, 'BRSTeamLoss_std'], costFrame.loc[cost, 'BRSTeamLoss'] + costFrame.loc[cost, 'BRSTeamLoss_std'], color=color_dict['BRS'], alpha=0.2)
            costFrame.sort_values(by=['TR_Contradictions'], inplace=True)
            plt.plot(costFrame['TR_Contradictions'], costFrame['TeamRulesTeamLoss'], marker='.', markersize=2, c=color_dict['TR'], label='TeamRules', linewidth=0.9)
            for cost in costFrame.Costs:
                plt.fill_between(np.linspace(costFrame.loc[cost, 'TR_Contradictions']-costFrame.loc[cost, 'TR_Contradictions_std'], 
                                            costFrame.loc[cost, 'TR_Contradictions']+costFrame.loc[cost, 'TR_Contradictions_std'], 
                                            50), costFrame.loc[cost, 'TeamRulesTeamLoss'] - costFrame.loc[cost, 'TeamRulesTeamLoss_std'], costFrame.loc[cost, 'TeamRulesTeamLoss'] + costFrame.loc[cost, 'TeamRulesTeamLoss_std'], color=color_dict['TR'], alpha=0.2)
            plt.xlabel('# of Contradictions', fontsize=12)
            plt.ylabel('Team Decision Loss', fontsize=12)
            plt.tick_params(labelrotation=45, labelsize=10)
            plt.legend(prop={'size': 5})
            #plt.set_title('{} Setting'.format(setting), fontsize=15)

            fig.savefig(f'Plots/loss{whicher}_TDL_{setting_type}_len{rule_len}_{data}_{setting}_asym{asym_loss[0]}{asym_loss[1]}.png', bbox_inches='tight')
                    
                
                    
                
                
            #fig.savefig('Plots/asym_2_1_{}_{}.png'.format(data,setting), bbox_inches='tight')
    


    
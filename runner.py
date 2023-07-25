from experiment_helper import *
from util import *
from sklearn.preprocessing import MinMaxScaler
import pickle
import gc


def basic_ADB_func_det(c_human, c_model=None, agreement=None):
    #returns the probability that the human accepts a recommendation given conf of human and conf of model
    return c_human <= 0.5

def basic_ADB_func_prob(c_human, c_model=None, agreement=None):
    #returns the probability that the human accepts a recommendation given conf of human and conf of model
    return 1-c_human

def basic_ADB_predicted_accept(paccept):
    #returns back the passed in probability that the human accepts a recommendation given feature values
    return paccept

def complex_ADB(c_human, c_model, agreement, delta=5, beta=0.05, k=0.63, gamma=0.95):
    #from will you accept the AI recommendation
    def w(p, k):
        return (p**k)/((p**k)+(1-p)**k)
    

    
    c_human_new = c_human.copy()
    c_human_new[c_human_new <= 0] = 0.0000001
    c_human_new[c_human_new >= 1] = 0.9999999

    if c_model.min() >= 0.5:
        scaler = MinMaxScaler((0,1))
        scaler.fit(np.array(c_model).reshape(-1,1))
        c_model_new = scaler.transform(np.array(c_model).reshape(-1,1)).flatten()
    else:
        c_model_new = c_model.copy()
    c_model_new[c_model_new <= 0] = 0.0000001
    c_model_new[c_model_new >= 1] = 0.9999999
    
    c_human_new[~agreement] = 1-c_human_new[~agreement]
    a = (c_model_new**gamma)/((c_model_new**gamma)+((1-c_model_new)**gamma))
    b = (c_human_new**gamma)/((c_human_new**gamma)+((1-c_human_new)**gamma))
    
    conf = 1/(1+(((1-a)*(1-b))/(a*b)))
    
    util_accept = (1+beta)*w(conf,k)-beta
    util_reject = 1-(1+beta)*w(conf,k)


    prob = np.exp(delta*util_accept)/(np.exp(delta*util_accept)+np.exp(delta*util_reject))
    df = pd.DataFrame({'c_human': c_human, 'c_human_new' : c_human_new, 'conf' : conf, 'c_model':c_model, 'agreement':agreement, 'prob': prob})

    return prob

# Repeat Experiments
def run(team1, team2, team3, folder, team_info):

    setting_type = 'learned'


    if setting_type=='learned':
        folder = folder + '_learned'
    elif setting_type=='perfect':
        folder = folder + '_perfect'



    team1_2_start_threshold = 0.5
    team3_4_start_threshold = 0.5

    #initial hyperparams
    Niteration = 1000
    Nchain = 1
    Nlevel = 1
    Nrules = 10000
    supp = 5
    maxlen = 2
    protected = 'NA'
    budget = 1
    sample_ratio = 1
    alpha = 0
    beta = 0
    iters = Niteration
    contradiction_reg = 0
    fairness_reg = 0
    numRuns = 10
    asym_loss = [1,1]

    if asym_loss != [1,1]:
        folder += '_asym' + str(asym_loss)

    folder = folder + f'_len{maxlen}' 

    team_info.to_pickle('{}/start_info.pkl'.format(folder))

    team1.data_model_dict['Xtrain'].to_pickle('{}/startDataSet.pkl'.format(folder))

    


    fA=complex_ADB

    



    validations = 1
    whichTeams = ['team1', 'team2', 'team3']

    teams = [team1, team2, team3]
    for run in range(0, numRuns):

        team_info = pd.DataFrame(index=[1, 2, 3])

        if asym_loss == [1,1]:
            contradiction_regs = [0, 0.01, 0.05, 0.1, 0.2,
                        0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        else:
            contradiction_regs = [0, 0.2,
                        0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.2, 1.5, 2, 2.5]

        
        if asym_loss == [1,1]:
            alt_mods = ['hyrs', 'brs']
        else:
            alt_mods = ['hyrs']
        
        
        contradiction_reg = 0


        try:
            team1
        except:
            with open(f'{folder}/team1.pkl', 'rb') as inp:
                team1 = pickle.load(inp)
            inp.close()
        
        try:
            team2
        except:
            with open(f'{folder}/team2.pkl', 'rb') as inp:
                team2 = pickle.load(inp)
            inp.close()
        
        try:
            team3
        except:
            with open(f'{folder}/team3.pkl', 'rb') as inp:
                team3 = pickle.load(inp)
            inp.close()

        # split training and test randomly
        team1.makeAdditionalTestSplit(testPercent=0.2, replaceExisting=True, random_state=run, others=[team2, team3])

        startData = pd.DataFrame({'team1_Ytest': team1.data_model_dict['Ytest'],
                                'team1_Ybtest': team1.data_model_dict['Ybtest'],
                                'team2_Ytest': team2.data_model_dict['Ytest'],
                                'team2_Ybtest': team2.data_model_dict['Ybtest'],
                                'team3_Ytest': team3.data_model_dict['Ytest'],
                                'team3_Ybtest': team3.data_model_dict['Ybtest'],
                                'team1TestConf': team1.data_model_dict['test_conf'],
                                'team2TestConf': team2.data_model_dict['test_conf'],
                                'team3TestConf': team3.data_model_dict['test_conf'],
                                'testDex': team1.testDex})

        startData.to_pickle('{}/dataset_run{}.pkl'.format(folder, run))

        for i in range(1, 4):
            team_info.loc[i, 'Human Test Acc'] = metrics.accuracy_score(teams[i - 1].data_model_dict['Ytest'],
                                                                        teams[i - 1].data_model_dict['Ybtest'])


        
        # train aversion and error boundary models
        if 'team1' in whichTeams:
            team1.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio,
                                        alpha,
                                        beta, iters, fairness_reg, contradiction_reg, fA, force_complete_coverage=False, asym_loss = asym_loss, fair_feat=None)
            team1.train_confidence_model(setting_type, 0.2)
            team1.train_mental_error_boundary_model()
            if setting_type=='learned':
                team1.train_ADB_model(0.2)
                team1.set_fA(team1.trained_ADB_model_wrapper)
            
            
            with open('{}/team1.pkl'.format(folder), 'wb') as outp:
                pickle.dump(team1, outp, pickle.HIGHEST_PROTOCOL)
            outp.close()

            del team1

            gc.collect()

        if 'team2' in whichTeams:
            team2.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio,
                                        alpha,
                                        beta, iters, fairness_reg, contradiction_reg, fA, force_complete_coverage=False, asym_loss = asym_loss, fair_feat=None)
            
            team2.train_confidence_model(setting_type, 0.2)
            
            team2.train_mental_error_boundary_model()
            if setting_type=='learned':
                team2.train_ADB_model(0.2)
                team2.set_fA(team2.trained_ADB_model_wrapper)
            
            
            with open('{}/team2.pkl'.format(folder), 'wb') as outp:
                pickle.dump(team2, outp, pickle.HIGHEST_PROTOCOL)
            outp.close()

            del team2

            gc.collect()

        if 'team3' in whichTeams:
            team3.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio,
                                        alpha,
                                        beta, iters, fairness_reg, contradiction_reg, fA, force_complete_coverage=False, asym_loss = asym_loss, fair_feat=None)
            
            team3.train_confidence_model(setting_type, 0.2)
            
            team3.train_mental_error_boundary_model()
            if setting_type=='learned':
                team3.train_ADB_model(0.2)
                team3.set_fA(team3.trained_ADB_model_wrapper)
            
            
            with open('{}/team3.pkl'.format(folder), 'wb') as outp:
                pickle.dump(team3, outp, pickle.HIGHEST_PROTOCOL)
            outp.close()

            del team3     

            gc.collect()    

        team_info.to_pickle('{}/team_info_run{}.pkl'.format(folder, run))

        for reg in contradiction_regs:
            contradiction_reg = reg
            
            
            print('training team1 hyrs model...')
            # train hyrs baseline

            

            if 'team1' in whichTeams:
                try:
                    team1
                except:
                    with open(f'{folder}/team1.pkl', 'rb') as inp:
                        team1 = pickle.load(inp)
                    inp.close()

                team1.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio,
                                        alpha,
                                        beta, iters, fairness_reg, contradiction_reg, fA, force_complete_coverage=False, asym_loss = asym_loss, fair_feat=None)
                
                
                
                t = time.time()
                team1.setup_hyrs()
                print(time.time() - t)

                tempval = 100
                tempTeam = deepcopy(team1)
                for i in range(validations):
                    tempTeam.train_hyrs()
                    if tempTeam.hyrs.val_obj < tempval:
                        team1 = tempTeam
                        tempval = tempTeam.hyrs.val_obj
                del tempTeam
                team1.filter_hyrs_results(mental=True, error=False)
                
            

                if (contradiction_reg == 0):
                    print('training team1 brs model...')
                    team1.setup_brs()

                    tempval = 100
                    tempTeam = deepcopy(team1)
                    for i in range(validations):
                        tempTeam.train_brs()
                        if tempTeam.brs_objective(contradiction_reg, 'val') < tempval:
                            team1 = tempTeam
                            tempval = team1.brs_objective(contradiction_reg, 'val')
                    
                    team1.brs_results.to_pickle('{}/team1_brs_run{}.pkl'.format(folder, run))
                    

                
                print('training team1 tr model...')
                team1.setup_tr()
                tempval = 100
                tempTeam = deepcopy(team1)
                for i in range(validations):
                    tempTeam.train_tr(alt_mods=alt_mods)
                    tempTeam.filter_tr_results(mental=True, error=False)
                    if tempTeam.full_tr_results_val.iloc[2, :]['objective'] < tempval:
                        team1 = tempTeam
                        tempval = tempTeam.full_tr_results_val.iloc[2, :]['objective']
                
                #team1.train_tr(resume=True, resume_with = {'prs': team1.tr.prs_min, 'nrs': team1.tr.nrs_min})
                #team1.filter_tr_results(mental=True, error=False)

                


                
                
                

                team1.full_hyrs_results.to_pickle('{}/cost_{}_team1_hyrs_filtered_run{}.pkl'.format(folder, reg, run))
            
                team1.full_tr_results.to_pickle('{}/cost_{}_team1_tr_filtered_run{}.pkl'.format(folder, reg, run))

                team1.full_hyrs_results_val.to_pickle('{}/val_cost_{}_team1_hyrs_filtered_run{}.pkl'.format(folder, reg, run))
            
                team1.full_tr_results_val.to_pickle('{}/val_cost_{}_team1_tr_filtered_run{}.pkl'.format(folder, reg, run))

                with open('{}/cost_{}_team1_tr_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
                    pickle.dump(team1.tr_results, outp, pickle.HIGHEST_PROTOCOL)
                outp.close()
                
                with open('{}/cost_{}_team1_hyrs_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
                    pickle.dump(team1.hyrs_results, outp, pickle.HIGHEST_PROTOCOL)
                outp.close()

                with open('{}/team1.pkl'.format(folder), 'wb') as outp:
                    pickle.dump(team1, outp, pickle.HIGHEST_PROTOCOL)
                outp.close()

                del team1
                gc.collect()

            
            
            if 'team2' in whichTeams:
                print('training team2 hyrs model...')

                try:
                    team2
                except:
                    with open(f'{folder}/team2.pkl', 'rb') as inp:
                        team2 = pickle.load(inp)
                    inp.close()

                team2.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio,
                                        alpha,
                                        beta, iters, fairness_reg, contradiction_reg, fA, force_complete_coverage=False, asym_loss = asym_loss, fair_feat=None)
                
                team2.setup_hyrs()

                tempval = 100
                tempTeam = deepcopy(team2)
                for i in range(validations):
                    tempTeam.train_hyrs()
                    if tempTeam.hyrs.val_obj < tempval:
                        team2 = tempTeam
                        tempval = tempTeam.hyrs.val_obj
                del tempTeam
                #team2.train_hyrs()
                team2.filter_hyrs_results(mental=True, error=False)
                
            

                if (contradiction_reg == 0):
                    print('training team2 brs model...')
                    team2.setup_brs()

                    tempval = 100
                    tempTeam = deepcopy(team2)
                    for i in range(validations):
                        tempTeam.train_brs()
                        if tempTeam.brs_objective(contradiction_reg, 'val') < tempval:
                            team2 = tempTeam
                            tempval = team2.brs_objective(contradiction_reg, 'val')
                    
                    team2.brs_results.to_pickle('{}/team2_brs_run{}.pkl'.format(folder, run))
                    


                    #team2.train_brs()
                    # print(team2.brs_results['test_error_modelonly'])
                
                
                print('training team2 tr model...')
                team2.setup_tr()
                tempval = 100
                tempTeam = deepcopy(team2)
                for i in range(validations):
                    tempTeam.train_tr(alt_mods=alt_mods)
                    tempTeam.filter_tr_results(mental=True, error=False)
                    if tempTeam.full_tr_results_val.iloc[2, :]['objective'] < tempval:
                        team2 = tempTeam
                        tempval = tempTeam.full_tr_results_val.iloc[2, :]['objective']
                
                
                team2.full_hyrs_results.to_pickle('{}/cost_{}_team2_hyrs_filtered_run{}.pkl'.format(folder, reg, run))
                team2.full_tr_results.to_pickle('{}/cost_{}_team2_tr_filtered_run{}.pkl'.format(folder, reg, run))
                team2.full_hyrs_results_val.to_pickle('{}/val_cost_{}_team2_hyrs_filtered_run{}.pkl'.format(folder, reg, run))
                team2.full_tr_results_val.to_pickle('{}/val_cost_{}_team2_tr_filtered_run{}.pkl'.format(folder, reg, run))

                with open('{}/cost_{}_team2_tr_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
                    pickle.dump(team2.tr_results, outp, pickle.HIGHEST_PROTOCOL)
                outp.close()

                with open('{}/cost_{}_team2_hyrs_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
                    pickle.dump(team2.hyrs_results, outp, pickle.HIGHEST_PROTOCOL)
                outp.close() 

                with open('{}/team2.pkl'.format(folder), 'wb') as outp:
                    pickle.dump(team2, outp, pickle.HIGHEST_PROTOCOL)
                outp.close()

                del team2
                gc.collect()
            
            
            if 'team3' in whichTeams:

                try:
                    team3
                except:
                    with open(f'{folder}/team3.pkl', 'rb') as inp:
                        team3 = pickle.load(inp)
                    inp.close()

                print('training team3 hyrs model...')
                team3.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio,
                                        alpha,
                                        beta, iters, fairness_reg, contradiction_reg, fA, force_complete_coverage=False, asym_loss = asym_loss, fair_feat=None)
                
                team3.setup_hyrs()

                tempval = 100
                tempTeam = deepcopy(team3)
                for i in range(validations):
                    tempTeam.train_hyrs()
                    if tempTeam.hyrs.val_obj < tempval:
                        team3 = tempTeam
                        tempval = tempTeam.hyrs.val_obj
                del tempTeam
                #team3.train_hyrs()
                team3.filter_hyrs_results(mental=True, error=False)
                
            

                if (contradiction_reg == 0):
                    print('training team3 brs model...')
                    team3.setup_brs()

                    tempval = 100
                    tempTeam = deepcopy(team3)
                    for i in range(validations):
                        tempTeam.train_brs()
                        if tempTeam.brs_objective(contradiction_reg, 'val') < tempval:
                            team3 = tempTeam
                            tempval = team3.brs_objective(contradiction_reg, 'val')
                    team3.brs_results.to_pickle('{}/team3_brs_run{}.pkl'.format(folder, run))
                    


                    #team3.train_brs()
                    # print(team3.brs_results['test_error_modelonly'])
                
                
                print('training team3 tr model...')
                team3.setup_tr()
                tempval = 100
                tempTeam = deepcopy(team3)
                for i in range(validations):
                    tempTeam.train_tr(alt_mods=alt_mods)
                    tempTeam.filter_tr_results(mental=True, error=False)
                    if tempTeam.full_tr_results_val.iloc[2, :]['objective'] < tempval:
                        team3 = tempTeam
                        tempval = tempTeam.full_tr_results_val.iloc[2, :]['objective']
                
        
               
                
                team3.full_hyrs_results.to_pickle('{}/cost_{}_team3_hyrs_filtered_run{}.pkl'.format(folder, reg, run))
                team3.full_tr_results.to_pickle('{}/cost_{}_team3_tr_filtered_run{}.pkl'.format(folder, reg, run))

                team3.full_hyrs_results_val.to_pickle('{}/val_cost_{}_team3_hyrs_filtered_run{}.pkl'.format(folder, reg, run))
                team3.full_tr_results_val.to_pickle('{}/val_cost_{}_team3_tr_filtered_run{}.pkl'.format(folder, reg, run))

                with open('{}/cost_{}_team3_tr_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
                    pickle.dump(team3.tr_results, outp, pickle.HIGHEST_PROTOCOL)
                outp.close()

                with open('{}/cost_{}_team3_hyrs_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
                    pickle.dump(team3.hyrs_results, outp, pickle.HIGHEST_PROTOCOL)
                outp.close()

                with open('{}/team3.pkl'.format(folder), 'wb') as outp:
                    pickle.dump(team3, outp, pickle.HIGHEST_PROTOCOL)
                outp.close()
                
                del team3
                gc.collect()
            
            

            '''
            #forced coverage versions
            
            team1.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio,
                                    alpha,
                                    beta, iters, fairness_reg, contradiction_reg, fA, force_complete_coverage=True)
            team1.setup_hyrs()
            team1.train_hyrs()
            team1.filter_hyrs_results(mental=True, error=False)


            print('training team1 tr model...')
            team1.setup_tr()
            team1.train_tr()
            team1.filter_tr_results(mental=True, error=False)

            team2.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio,
                                    alpha,
                                    beta, iters, fairness_reg, contradiction_reg, fA, force_complete_coverage=True)
            team2.setup_hyrs()
            team2.train_hyrs()
            team2.filter_hyrs_results(mental=True, error=False)


            print('training team2 tr model...')
            team2.setup_tr()
            team2.train_tr()
            team2.filter_tr_results(mental=True, error=False)

            team3.set_training_params(Niteration, Nchain, Nlevel, Nrules, supp, maxlen, protected, budget, sample_ratio,
                                    alpha,
                                    beta, iters, fairness_reg, contradiction_reg, fA, force_complete_coverage=True)
            team3.setup_hyrs()
            team3.train_hyrs()
            team3.filter_hyrs_results(mental=True, error=False)


            print('training team3 tr model...')
            team3.setup_tr()
            team3.train_tr()
            team3.filter_tr_results(mental=True, error=False)

            # team1
            
            team1.full_hyrs_results.to_pickle('{}/fc_cost_{}_team1_hyrs_filtered_run{}.pkl'.format(folder, reg, run))
            team1.full_tr_results.to_pickle('{}/fc_cost_{}_team1_tr_filtered_run{}.pkl'.format(folder, reg, run))

            with open('{}/fc_cost_{}_team1_tr_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
                pickle.dump(team1.tr_results, outp, pickle.HIGHEST_PROTOCOL)
            outp.close()

            with open('{}/fc_cost_{}_team1_hyrs_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
                pickle.dump(team1.hyrs_results, outp, pickle.HIGHEST_PROTOCOL)
            outp.close()

            # team2
            team2.full_hyrs_results.to_pickle('{}/fc_cost_{}_team2_hyrs_filtered_run{}.pkl'.format(folder, reg, run))
            team2.full_tr_results.to_pickle('{}/fc_cost_{}_team2_tr_filtered_run{}.pkl'.format(folder, reg, run))

            with open('{}/fc_cost_{}_team2_tr_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
                pickle.dump(team2.tr_results, outp, pickle.HIGHEST_PROTOCOL)
            outp.close()

            with open('{}/fc_cost_{}_team2_hyrs_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
                pickle.dump(team2.hyrs_results, outp, pickle.HIGHEST_PROTOCOL)
            outp.close()

            # team3

            team3.full_hyrs_results.to_pickle('{}/fc_cost_{}_team3_hyrs_filtered_run{}.pkl'.format(folder, reg, run))
            team3.full_tr_results.to_pickle('{}/fc_cost_{}_team3_tr_filtered_run{}.pkl'.format(folder, reg, run))

            with open('{}/fc_cost_{}_team3_tr_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
                pickle.dump(team3.tr_results, outp, pickle.HIGHEST_PROTOCOL)
            outp.close()

            with open('{}/fc_cost_{}_team3_hyrs_results_run{}.pkl'.format(folder, reg, run), 'wb') as outp:
                pickle.dump(team3.hyrs_results, outp, pickle.HIGHEST_PROTOCOL)
            outp.close()
            
            '''
            
    print('finally done.')
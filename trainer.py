from sklearn.externals import joblib
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import gc
from datetime import datetime
import sys
import pickle


def catboost_model(features, test_features ,catboostparams ,idcol,targetcol,testcolhas_target='yes',
                   catthreshold = 100 ,n_folds = 2,filename='catboost.pkl',identifier='catboost'):
    train_ids = features[idcol]
    test_ids = test_features[idcol]
    labels = features[targetcol]

    features = features.drop(columns = [idcol, targetcol])
    if testcolhas_target == 'yes':
        test_features = test_features.drop(columns = [idcol ,targetcol])
    else:
        test_features = test_features.drop(columns=[idcol])
    catlabels = []
    catcols = []
    i = 0
    if catthreshold>0:
        for x in features:
            if x not in [idcol ,targetcol]:
                if features[x].dtype != float :
                    if features[x].nunique() < catthreshold:
                        catlabels.append(i)
                        catcols.append(x)
            i = i+ 1

    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    feature_names = list(features.columns)
    features = np.array(features)
    test_features = np.array(test_features)
    k_fold = KFold(n_splits=n_folds, shuffle=False, random_state=50)
    feature_importance_values = np.zeros(len(feature_names))
    test_predictions = np.zeros(test_features.shape[0])
    out_of_fold = np.zeros(features.shape[0])
    valid_scores = []
    train_scores = []
    for train_indices, valid_indices in k_fold.split(features):
        train_features, train_labels = features[train_indices], labels[train_indices]
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        model = CatBoostRegressor(**catboostparams)
        model.fit(train_features, train_labels,
                  eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                  early_stopping_rounds=500, verbose=200, cat_features=catlabels if catthreshold > 1 else None)
        best_iteration = model.best_iteration_
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        test_predictions += model.predict(test_features) / k_fold.n_splits
        out_of_fold[valid_indices] = model.predict(valid_features)
        valid_score = model.best_score_['validation_0']['RMSE']
        train_score = model.best_score_['validation_1']['RMSE']
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        gc.enable()
        del train_features, valid_features
        gc.collect()
    #joblib.dump(model, 'catboost_1.pkl')
    pickle.dump(model, open(filename, 'wb'))
    listcol = identifier + '_list'+'.pkl'
    pickle.dump(feature_names, open(listcol, 'wb'))
    gc.enable()
    del model
    gc.collect()
    predcol = 'y_pred_'+identifier
    submission = pd.DataFrame({'id': test_ids, predcol: test_predictions})
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    valid_r2 = r2_score(labels, out_of_fold)
    valid_scores.append(valid_r2)
    train_scores.append(np.mean(train_scores))
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})
    return submission, feature_importances, metrics


def lightgbm_model(features, test_features, lgbmparams,idcol,targetcol,testcolhas_target='yes', n_folds=2,filename='lgbm.pkl',
                   identifier='lgbm'):
    train_ids = features[idcol]
    test_ids = test_features[idcol]
    labels = features[targetcol]
    features = features.drop(columns=[idcol, targetcol])
    if testcolhas_target == 'yes':
        test_features = test_features.drop(columns=[idcol, targetcol])
    else:
        test_features = test_features.drop(columns=[idcol])
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    feature_names = list(features.columns)
    features = np.array(features)
    test_features = np.array(test_features)
    k_fold = KFold(n_splits=n_folds, shuffle=False, random_state=50)
    feature_importance_values = np.zeros(len(feature_names))
    test_predictions = np.zeros(test_features.shape[0])
    out_of_fold = np.zeros(features.shape[0])
    valid_scores = []
    train_scores = []
    for train_indices, valid_indices in k_fold.split(features):
        train_features, train_labels = features[train_indices], labels[train_indices]
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        model = lgb.LGBMRegressor(**lgbmparams)
        model.fit(train_features, train_labels,
                  eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                  early_stopping_rounds=500, verbose=200, eval_metric='RMSE')
        best_iteration = model.best_iteration_
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        test_predictions += model.predict(test_features, num_iteration=best_iteration) / k_fold.n_splits
        out_of_fold[valid_indices] = model.predict(valid_features, num_iteration=best_iteration)
        valid_score = model.best_score_['valid_0']['rmse']
        train_score = model.best_score_['training']['rmse']
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        gc.enable()
        del train_features, valid_features
        gc.collect()
    predcol = 'y_pred_'+identifier
    submission = pd.DataFrame({'id': test_ids, predcol: test_predictions})
    #joblib.dump(model, 'lgb_1.pkl')
    pickle.dump(model, open(filename, 'wb'))
    listcol = identifier + '_list'+'.pkl'
    pickle.dump(feature_names, open(listcol, 'wb'))
    gc.enable()
    del model
    gc.collect()
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    valid_r2 = r2_score(labels, out_of_fold)
    valid_scores.append(valid_r2)
    train_scores.append(np.mean(train_scores))
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})

    return submission, feature_importances, metrics

def load_predict(modelfilename_list,maindf,targetcol='y',id_col = 'id',testcolhas_target='yes',modelidentifier=['lgbm','catboost']):

    submission_list = []
    for modelpickle,modelid in zip(modelfilename_list,modelidentifier):
        model = pickle.load(open(modelpickle, 'rb'))

        featurecol = pickle.load(open(modelid + '_list.pkl', 'rb'))
        print(f"INFO : {str(datetime.now())} : Loaded pickle file for model {modelid}")
        #print(featurecol)
        #if testcolhas_target == 'yes':
        #    cols = featurecol +  [targetcol, id_col]
        #else:
        #    cols = featurecol +  [ id_col]
        test_features = np.array(maindf[featurecol])
        test_ids = np.array(maindf[id_col])
        test_predictions = model.predict(test_features)
        print(f"INFO : {str(datetime.now())} : Prediction done for model {modelid}")
        predcol = 'y_pred_'+modelid
        submission = pd.DataFrame({id_col: test_ids, predcol: test_predictions})
        submission_list.append(submission)
    #appended_data = pd.concat(appended_data)
    return submission_list



def colate_and_predict(model_pred_list,maindf, reg_th=3,modelidentifier=['lgbm','catboost']):
    try:
        accuracy = {}
        maindf = maindf[['id', 'y']]
        combined_df = pd.merge(model_pred_list[0],model_pred_list[1],on='id',how='inner')
        sub = pd.merge(maindf, combined_df, on='id', how='inner')
        sub['y_pred_'+modelidentifier[0]+'-y'] = (sub['y_pred_'+modelidentifier[0]] - sub['y']).abs()
        sub['result_'+modelidentifier[0]] = sub['y_pred_'+modelidentifier[0]+'-y'].apply(lambda x: 0 if x > reg_th else 1)

        sub['y_pred_'+modelidentifier[1]+'-y'] = (sub['y_pred_'+modelidentifier[1]] - sub['y']).abs()
        sub['result_'+modelidentifier[1]] = sub['y_'+modelidentifier[1]+'-y'].apply(lambda x: 0 if x > reg_th else 1)

        if len(model_pred_list) > 1:
            sub['y_pred_comb_max'] = sub[['result_'+modelidentifier[0], 'result_'+modelidentifier[0]]].max(axis=1)
            sub['y_pred_comb_max-y'] = (sub['y_pred_comb_max'] - sub['y']).abs()
            sub['result_comb_max'] = sub['y_pred_comb_max-y'].apply(lambda x: 0 if x > reg_th else 1)

            sub['y_pred_comb_min'] = sub[['result_'+modelidentifier[0], 'result_'+modelidentifier[1]]].min(axis=1)
            sub['y_pred_comb_min-y'] = (sub['y_pred_comb_min'] - sub['y']).abs()
            sub['result_comb_min'] = sub['y_pred_comb_min-y'].apply(lambda x: 0 if x > reg_th else 1)

            sub['y_pred_comb_mean'] = (sub['result_'+modelidentifier[1]] + sub['result_'+modelidentifier[0]])/2
            sub['y_pred_comb_mean-y'] = (sub['y_pred_comb_mean'] - sub['y']).abs()
            sub['result_comb_mean'] = sub['y_pred_comb_mean-y'].apply(lambda x: 0 if x > reg_th else 1)

            med = sub.groupby(['id'])[['result_'+modelidentifier[0], 'result_'+modelidentifier[1]]].apply(np.median)
            med.name = 'y_pred_comb_median'
            sub = sub.join(med, on=['id'])
            sub['y_pred_comb_median-y'] = (sub['y_pred_comb_median'] - sub['y']).abs()
            sub['result_comb_median'] = sub['y_pred_comb_median-y'].apply(lambda x: 0 if x > reg_th else 1)


        if len(model_pred_list) > 1:
            result_lgbm_d = dict(sub['result_'+modelidentifier[0]].value_counts())
            result_catboost_d = dict(sub['result_'+modelidentifier[1]].value_counts())
            result_comb_max_d = dict(sub['result_comb_max'].value_counts())
            result_comb_min_d = dict(sub['result_comb_min'].value_counts())
            result_comb_mean_d = dict(sub['result_comb_mean'].value_counts())
            result_comb_median_d = dict(sub['result_comb_median'].value_counts())
            if len(result_lgbm_d) > 1:
                accuracy.update({'result_'+modelidentifier[0]:(result_lgbm_d[1] / (result_lgbm_d[0] + result_lgbm_d[1]))})
            else:
                if list(result_lgbm_d.items())[0][0] ==0:
                    accuracy.update({'result_lgbm':0.0})
                else:
                    accuracy.update({'result_lgbm': 100.0})
            if len(result_catboost_d) > 1:
                accuracy.update({'result_catboost':(result_catboost_d[1] / (result_catboost_d[0] + result_catboost_d[1]))})
            else:
                if list(result_lgbm_d.items())[0][0] ==0:
                    accuracy.update({'result_catboost':0.0})
                else:
                    accuracy.update({'result_catboost': 100.0})
            if len(result_comb_max_d) > 1:
                accuracy.update({'result_comb_max':(result_comb_max_d[1] / (result_comb_max_d[0] + result_comb_max_d[1]))})
            else:
                if list(result_comb_max_d.items())[0][0] ==0:
                    accuracy.update({'result_comb_max':0.0})
                else:
                    accuracy.update({'result_comb_max': 100.0})
            #accuracy.update({'result_comb_min':(result_comb_min_d[1] / (result_comb_min_d[0] + result_comb_min_d[1]))})
            if len(result_comb_min_d) > 1:
                accuracy.update({'result_comb_min':(result_comb_min_d[1] / (result_comb_min_d[0] + result_comb_min_d[1]))})
            else:
                if list(result_comb_min_d.items())[0][0] ==0:
                    accuracy.update({'result_comb_min':0.0})
                else:
                    accuracy.update({'result_comb_min': 100.0})
            #accuracy.update({'result_comb_mean':(result_comb_mean_d[1] / (result_comb_mean_d[0] + result_comb_mean_d[1]))})
            if len(result_comb_mean_d) > 1:
                accuracy.update({'result_comb_mean':(result_comb_mean_d[1] / (result_comb_mean_d[0] + result_comb_mean_d[1]))})
            else:
                if list(result_comb_mean_d.items())[0][0] ==0:
                    accuracy.update({'result_comb_mean':0.0})
                else:
                    accuracy.update({'result_comb_mean': 100.0})
            #accuracy.update({'result_comb_median':(result_comb_median_d[1] / (result_comb_median_d[0] + result_comb_median_d[1]))})
            if len(result_comb_median_d) > 1:
                accuracy.update({'result_comb_median':(result_comb_median_d[1] / (result_comb_median_d[0] + result_comb_median_d[1]))})
            else:
                if list(result_comb_median_d.items())[0][0] ==0:
                    accuracy.update({'result_comb_median':0.0})
                else:
                    accuracy.update({'result_comb_median': 100.0})
        sub.to_csv('submission.csv',index=False)
        return accuracy,sub
    except:
        sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

def colate_and_predict_pretrained(model_pred_list,maindf, modelidentifier=['lgbm','catboost'],id_col='id'):
    accuracy = {}
    maindf = maindf[[id_col]]
    dfs = [df.set_index(id_col) for df in model_pred_list]
    combined_df = pd.concat(dfs, axis=1)
    # print(combined_df.head())
    # print(combined_df.shape)
    sub = pd.merge(maindf, combined_df, on='id', how='inner')
    # print(sub.head())
    # print(sub.shape)
    resultcol = []
    predcol = []
    for mi in modelidentifier:
        #sub['y_pred_' + mi + '-y'] = (sub['y_pred_' + mi] - sub['y']).abs()
        #sub['y_pred_' + mi + '-y'] = (sub['y_pred_' + mi] - sub['y']).abs()
        predcol.append('y_pred_' + mi)
    sub['y_pred_comb_max'] = sub[predcol].max(axis=1)
    sub['y_pred_comb_min'] = sub[predcol].min(axis=1)
    sub['y_pred_comb_mean'] = sub[predcol].mean(axis=1)
    med = sub.groupby(['id'])[predcol].apply(np.median)
    med.name = 'y_pred_comb_median'
    sub = sub.join(med, on=[id_col])
    sub.to_csv('submission_pretrained.csv', index=False)
    print(f"INFO : {str(datetime.now())} : Prediction completed, Final submission file saved as submission_pretrained.csv")
    return sub


def colate_and_predict_imp(model_pred_list,maindf, reg_th=3,modelidentifier=['lgbm','catboost'],id_col='id',target_col='y'):
    try:
        accuracy = {}
        maindf = maindf[[id_col, target_col]]
        dfs = [df.set_index(id_col) for df in model_pred_list]
        combined_df = pd.concat(dfs,axis=1)
        #print(combined_df.head())
        #print(combined_df.shape)
        sub = pd.merge(maindf, combined_df, on=id_col, how='inner')
        #print(sub.head())
        #print(sub.shape)
        resultcol =[]
        predcol = []
        for mi in modelidentifier:
            sub['y_pred_'+mi+'-y'] = (sub['y_pred_'+mi] - sub[target_col]).abs()
            sub['result_'+mi] = sub['y_pred_'+mi+'-y'].apply(lambda x: 0 if x > reg_th else 1)
            sub['y_pred_'+mi+'-y'] = (sub['y_pred_'+mi] - sub[target_col]).abs()
            sub['result_'+mi] = sub['y_pred_'+mi+'-y'].apply(lambda x: 0 if x > reg_th else 1)
            resultcol.append('result_'+mi)
            predcol.append('y_pred_'+mi)
        if len(model_pred_list) > 1:
            sub['y_pred_comb_max'] = sub[predcol].max(axis=1)
            sub['y_pred_comb_max-y'] = (sub['y_pred_comb_max'] - sub[target_col]).abs()
            sub['result_comb_max'] = sub['y_pred_comb_max-y'].apply(lambda x: 0 if x > reg_th else 1)

            sub['y_pred_comb_min'] = sub[predcol].min(axis=1)
            sub['y_pred_comb_min-y'] = (sub['y_pred_comb_min'] - sub[target_col]).abs()
            sub['result_comb_min'] = sub['y_pred_comb_min-y'].apply(lambda x: 0 if x > reg_th else 1)

            #sub['y_pred_comb_mean'] = (sub[resultcol])/2
            sub['y_pred_comb_mean'] = sub[predcol].mean(axis=1)
            sub['y_pred_comb_mean-y'] = (sub['y_pred_comb_mean'] - sub[target_col]).abs()
            sub['result_comb_mean'] = sub['y_pred_comb_mean-y'].apply(lambda x: 0 if x > reg_th else 1)

            med = sub.groupby([id_col])[predcol].apply(np.median)
            med.name = 'y_pred_comb_median'
            sub = sub.join(med, on=[id_col])
            sub['y_pred_comb_median-y'] = (sub['y_pred_comb_median'] - sub[target_col]).abs()
            sub['result_comb_median'] = sub['y_pred_comb_median-y'].apply(lambda x: 0 if x > reg_th else 1)
            resultcol = resultcol + ['result_comb_max','result_comb_median','result_comb_mean','result_comb_min']
            #predcol = predcol + ['y_pred_comb_max', 'y_pred_comb_median', 'y_pred_comb_mean', 'y_pred_comb_min']

        for col in resultcol:
            val = dict(sub[col].value_counts())
            #print(val)
            #print(col)
            if len(val) > 1:
                accuracy.update({col: (val[1] / (val[0] + val[1]))})
            else:
                if list(val.keys())[0] == 0:
                    accuracy.update({col: 0.0})
                else:
                    accuracy.update({col: 100.0})
        sub.to_csv('submission.csv',index=False)
        print(f"INFO : {str(datetime.now())} : Prediction completed, Final submission file saved as submission.csv")
        #print(accuracy)
        return accuracy,sub
    except:
        sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])


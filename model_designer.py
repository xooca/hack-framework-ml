from data_transformer import datacleaner
from trainer import catboost_model
from trainer import lightgbm_model
from trainer import colate_and_predict
import gc
import configparser
config = configparser.ConfigParser()
config.read('config.ini')



filename = r'C:\Users\8prab\Downloads\Data Scientist Test\Data Scientist Test\dataset_00_with_header.csv'
dc = datacleaner(filename,targetcol='y',cat_threshold=20)

@dc.model_null_impute_cat_rf()
@dc.model_null_impute_notcat_rf()
@dc.retail_reject_cols(threshold=0.4)
def impute_nulls(tmpdf):
  return tmpdf

cleandf = impute_nulls(dc.df_train)
f_imp_list,f_impdf = dc.importantfeatures(cleandf,tobepredicted='y',modelname='lgbmregressor',
                                          skipcols=[],featurelimit=0)

important_columns = f_impdf[:200]['feature'].tolist() + ['y']
cleandf = cleandf[important_columns]

@dc.add_id_column()
#@dc.refresh_cat_noncat_cols(threshold=100)
@dc.standardize_simple_auto(range_tuple = (0,10))
#@dc.refresh_cat_noncat_cols(threshold=200)
#@dc.ohe_on_column(drop_converted_col=True)
@dc.refresh_cat_noncat_cols(threshold=20)
#@dc.featurization(cat_coltype=False)
#@dc.two_column_featurization(cat_coltype=False)
#@dc.refresh_cat_noncat_cols(threshold=100)
#@dc.convertdatatypes(cat_threshold=150)
#@dc.applypca(cat_coltype=False ,number_of_components = 10)
#@dc.refresh_cat_noncat_cols(threshold=100)
#@dc.high_coor_target_column(targetcol = 'y',th=0.05)
@dc.remove_collinear(th=0.95)
#@dc.model_null_impute_cat_rf()
#@dc.model_null_impute_notcat_rf()
#@dc.refresh_cat_noncat_cols(threshold=100)
@dc.convertdatatypes(cat_threshold=20)
#@dc.retail_reject_cols(threshold=0.4)
def cleandata_lgbm(tmpdf):
  return tmpdf

cleandf = cleandata_lgbm(cleandf)
train,test = dc.split_test_train(cleandf,test_size=0.2)

lgbmparams1 ={'num_leaves': 31,
    'learning_rate': 0.02,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_estimators': 20000,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'use_best_model':True}

submission_lgbm, fi_lgbm ,metrics_lgb= lightgbm_model(train, test,lgbmparams1,n_folds = 2)

lgbmparamscatboost ={
    #'task_type':'GPU',
    'iterations':10000,
    'learning_rate': 0.05,
    'verbose': -1,
    #'eval_metric':'RMSE',
    #'n_estimators': 20000,
    #'boosting_type': 'gbdt',
    #'objective': 'regression',
    'loss_function':'RMSE',
    'use_best_model':True}

submission_cat, fi_cat ,metrics_cat= catboost_model(train, test,lgbmparamscatboost,catthreshold=20,n_folds = 2)

accuracy = colate_and_predict([submission_lgbm,submission_cat],cleandf, reg_th=3)
cleandf.to_csv('cleandf.csv',index=False)
print(accuracy)
gc.enable()
del dc,cleandf
gc.collect()



from data_transformer import datacleaner
from trainer import catboost_model
from trainer import lightgbm_model
from trainer import colate_and_predict_pretrained,load_predict,colate_and_predict_imp
import gc
import configparser
import sys
from datetime import datetime

class model_designer:
    def __init__(self,configfile):
        config = configparser.ConfigParser()
        config.read(configfile)
        #print(config['DATACLEAN']['filename'])
        self.dc_param = config['DATACLEAN']
        #print(self.dc_param)
        self.ds_param = config['DATASPLIT']
        self.run_param = config['RUN']
        self.fi_param = config['FEATUREIMP']
        self.lgbm_param = config['LIGHTGBM']
        self.catboost_param = config['CATBOOST']
        self.predict_param = config['PREDICT']
        self.filesave_param = config['FILESAVE']
        #print(self.filesave_param['catboost_identifier'])
        self.model_identifier = [self.filesave_param['lightgbm_identifier'],self.filesave_param['catboost_identifier']]

    def designed_pipeline_train_test(self):
        dc = datacleaner(str(self.dc_param['filename']), targetcol=self.dc_param['targetcol'],
                         cat_threshold=int(self.dc_param['cat_threshold']))

        # Cleaning of data - remove nulls
        #@dc.add_id_column(self.dc_param['id_col'])
        @dc.model_null_impute_cat_rf()
        @dc.model_null_impute_notcat_rf()
        @dc.retail_reject_cols(threshold=float(self.dc_param['null_column_reject_threshold']))
        def impute_nulls(tmpdf):
          return tmpdf

        # Transformation of data
        @dc.add_id_column(self.dc_param['id_col'])
        @dc.standardize_simple_auto(range_tuple=(0, 10))
        @dc.refresh_cat_noncat_cols(threshold=int(self.dc_param['cat_threshold']))
        @dc.remove_collinear(th=float(self.dc_param['remove_collinear_col_threshold']))
        @dc.convertdatatypes(cat_threshold=int(self.dc_param['cat_threshold']))
        def transform_data(tmpdf):
            return tmpdf

        cleandf = impute_nulls(dc.df_train)
        cleandf = transform_data(cleandf)
        #print(str(self.fi_param['modelname']))
        f_imp_list,f_impdf = dc.importantfeatures(cleandf,tobepredicted=str(self.dc_param['targetcol']),
                                                  modelname=str(self.fi_param['modelname']),
                                                    skipcols=self.fi_param['skipcols'],
                                                  featurelimit=int(self.fi_param['featurelimit']))
        #dfforimp, tobepredicted, modelname = 'regressor', skipcols = [], featurelimit = 0
        #important_columns = f_impdf[:int(self.fi_param['totalfeature_limit'])]['feature'].tolist() + [self.dc_param['targetcol']]
        #cleandf = cleandf[important_columns]
        train, test = dc.split_test_train(cleandf, test_size=float(self.ds_param['testsize']))
        lgbm_param_dict = {}
        for key in self.lgbm_param:
            val_dtype = self.lgbm_param[key].split(',')[1]
            val = self.lgbm_param[key].split(',')[0]
            if val_dtype == 'str':
                newval = eval(val_dtype + '(' + "'"+val+"'" + ')')
            else:
                newval = eval(val_dtype+'('+val+')')
            lgbm_param_dict.update({key:newval})
        submission_lgbm, fi_lgbm, metrics_lgb = lightgbm_model(train, test, lgbm_param_dict,
                                                               idcol=self.dc_param['id_col'],
                                                               targetcol=self.dc_param['targetcol'],
                                                               testcolhas_target=self.dc_param['testcolhas_target'],
                                                               n_folds=int(self.run_param['lgbm_n_folds']),
                                                               filename=self.filesave_param['lightgbm_pickle'],
                                                               identifier = self.filesave_param['lightgbm_identifier'])
        catboost_param_dict = {}
        for key in self.catboost_param:
            val_dtype = self.catboost_param[key].split(',')[1]
            val = self.catboost_param[key].split(',')[0]
            if val_dtype == 'str':
                newval = eval(val_dtype + '(' + "'"+val+"'" + ')')
            else:
                newval = eval(val_dtype+'('+val+')')
            catboost_param_dict.update({key:newval})
        print(catboost_param_dict)
        submission_cat, fi_cat, metrics_cat = catboost_model(train, test, catboost_param_dict,
                                                             idcol=self.dc_param['id_col'],
                                                             targetcol=self.dc_param['targetcol'],
                                                             testcolhas_target=self.dc_param['testcolhas_target'],
                                                             catthreshold=int(self.dc_param['cat_threshold']),
                                                             n_folds=int(self.run_param['catboost_n_folds']),
                                                             filename=self.filesave_param['catboost_pickle'],
                                                             identifier = self.filesave_param['catboost_identifier'])
        accuracy,submission = colate_and_predict_imp([submission_lgbm,submission_cat],
                                                     cleandf,
                                                    reg_th=int(self.predict_param['abosulte_error_threshold']),
                                                    modelidentifier = self.model_identifier,
                                                    id_col=self.dc_param['id_col'],
                                                    target_col=self.dc_param['targetcol']
                                                     )
        cleandf.to_csv(self.filesave_param['save_clean_df_file'],index=False)
        print(f"Accuracy of the model with absolute error {self.predict_param['abosulte_error_threshold']} is {accuracy}")
        gc.enable()
        del dc,cleandf
        gc.collect()
        return accuracy,submission

    def designed_pipeline_only_train(self):
        dc = datacleaner(str(self.dc_param['filename']), targetcol=self.dc_param['targetcol'],
                         cat_threshold=int(self.dc_param['cat_threshold']))

        @dc.model_null_impute_cat_rf()
        @dc.model_null_impute_notcat_rf()
        @dc.retail_reject_cols(threshold=float(self.dc_param['null_column_reject_threshold']))
        def impute_nulls(tmpdf):
          return tmpdf

        # Transformation of data
        @dc.add_id_column(self.dc_param['id_col'])
        @dc.standardize_simple_auto(range_tuple=(0, 10))
        @dc.refresh_cat_noncat_cols(threshold=int(self.dc_param['cat_threshold']))
        @dc.remove_collinear(th=float(self.dc_param['remove_collinear_col_threshold']))
        @dc.convertdatatypes(cat_threshold=int(self.dc_param['cat_threshold']))
        def transform_data(tmpdf):
            return tmpdf

        cleandf = impute_nulls(dc.df_train)
        cleandf = transform_data(cleandf)
        #cleandf = transform_data(cleandf)
        #print(str(self.fi_param['modelname']))
        #f_imp_list,f_impdf = dc.importantfeatures(cleandf,tobepredicted=str(self.dc_param['targetcol']),modelname=str(self.fi_param['modelname']),
        #                                  skipcols=self.fi_param['skipcols'],featurelimit=int(self.fi_param['featurelimit']))
        #dfforimp, tobepredicted, modelname = 'regressor', skipcols = [], featurelimit = 0
        #important_columns = f_impdf[:int(self.fi_param['totalfeature_limit'])]['feature'].tolist() + [self.dc_param['targetcol']]
        #cleandf = cleandf[important_columns]
        train, test = dc.split_test_train(cleandf, test_size=float(self.ds_param['testsize']))
        lgbm_param_dict = {}
        for key in self.lgbm_param:
            val_dtype = self.lgbm_param[key].split(',')[1]
            val = self.lgbm_param[key].split(',')[0]
            if val_dtype == 'str':
                newval = eval(val_dtype + '(' + "'"+val+"'" + ')')
            else:
                newval = eval(val_dtype+'('+val+')')
            lgbm_param_dict.update({key:newval})
        submission_lgbm, fi_lgbm, metrics_lgb = lightgbm_model(train, test, lgbm_param_dict,
                                                               idcol=self.dc_param['id_col'],
                                                               targetcol=self.dc_param['targetcol'],
                                                               testcolhas_target=self.dc_param['testcolhas_target'],
                                                               n_folds=int(self.run_param['lgbm_n_folds']),
                                                               filename=self.filesave_param['lightgbm_pickle'],
                                                               identifier = self.filesave_param['lightgbm_identifier'])
        catboost_param_dict = {}
        for key in self.catboost_param:
            val_dtype = self.catboost_param[key].split(',')[1]
            val = self.catboost_param[key].split(',')[0]
            if val_dtype == 'str':
                newval = eval(val_dtype + '(' + "'"+val+"'" + ')')
            else:
                newval = eval(val_dtype+'('+val+')')
            catboost_param_dict.update({key:newval})
        print(catboost_param_dict)
        submission_cat, fi_cat, metrics_cat = catboost_model(train, test, catboost_param_dict,
                                                             idcol=self.dc_param['id_col'],
                                                             targetcol=self.dc_param['targetcol'],
                                                             testcolhas_target=self.dc_param['testcolhas_target'],
                                                             catthreshold=int(self.dc_param['cat_threshold']),
                                                             n_folds=int(self.run_param['catboost_n_folds']),
                                                             filename=self.filesave_param['catboost_pickle'],
                                                               identifier = self.filesave_param['catboost_identifier'])
        cleandf.to_csv(self.filesave_param['save_clean_df_file'], index=False)
        return None, None

    def designed_pipeline_only_predict(self):
        dc = datacleaner(str(self.dc_param['filename']), targetcol=self.dc_param['targetcol'],
                         cat_threshold=int(self.dc_param['cat_threshold']))

        # Cleaning of data - remove nulls
        #@dc.add_id_column(self.dc_param['id_col'])
        @dc.model_null_impute_cat_rf()
        @dc.model_null_impute_notcat_rf()
        @dc.retail_reject_cols(threshold=float(self.dc_param['null_column_reject_threshold']))
        def impute_nulls(tmpdf):
          return tmpdf

        # Transformation of data
        @dc.add_id_column(self.dc_param['id_col'])
        @dc.standardize_simple_auto(range_tuple=(0, 10))
        @dc.refresh_cat_noncat_cols(threshold=int(self.dc_param['cat_threshold']))
        @dc.remove_collinear(th=float(self.dc_param['remove_collinear_col_threshold']))
        @dc.convertdatatypes(cat_threshold=int(self.dc_param['cat_threshold']))
        def transform_data(tmpdf):
            return tmpdf

        cleandf = impute_nulls(dc.df_train)
        cleandf = transform_data(cleandf)
        modelfilename_list = [self.filesave_param['lightgbm_pickle'],self.filesave_param['catboost_pickle']]

        submissionlist = load_predict(modelfilename_list, cleandf,
                                      targetcol=self.dc_param['targetcol'],
                                      id_col=self.dc_param['id_col'],
                                      testcolhas_target=self.dc_param['testcolhas_target'],
                                      modelidentifier=self.model_identifier
                                      )
        #accuracy = colate_and_predict(submissionlist, cleandf,
        #                              reg_th=self.predict_param.abosulte_error_threshold)
        if self.dc_param['testcolhas_target'] == 'yes':
            accuracy,submission = colate_and_predict_imp(submissionlist,cleandf,
                                                reg_th=int(self.predict_param['abosulte_error_threshold']),
                                                modelidentifier=self.model_identifier,
                                                id_col=self.dc_param['id_col'],
                                                target_col=self.dc_param['targetcol'])
            print(f"Accuracy of the model with absolute error {self.predict_param['abosulte_error_threshold']} is {accuracy}")
        else:
            submission = colate_and_predict_pretrained(submissionlist,cleandf,
                                                modelidentifier=self.model_identifier,
                                                id_col=self.dc_param['id_col']
                                                       )
            accuracy = None
        return accuracy,submission

def main(configfile,option=1):
    md = model_designer(configfile)
    if option==1:
        accuracy,submission = md.designed_pipeline_train_test()
        return accuracy,submission
    elif option ==2:
        md.designed_pipeline_train_test()
    elif option ==3:
        accuracy,submission = md.designed_pipeline_only_predict()
        return accuracy, submission


if __name__ == '__main__':
    print(f"INFO : {str(datetime.now())} : option 1 is {sys.argv[1]}")
    print(f"INFO : {str(datetime.now())} : option [l;;;;
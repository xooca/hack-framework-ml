import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
from collections import OrderedDict
from datetime import datetime
import glob
import numpy as np
import pandas as pd
import os
import itertools
from sklearn.metrics import r2_score
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from datetime import datetime
import glob
import numpy as np
import pandas as pd
import os
import itertools
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import lightgbm as lgb
from sklearn.preprocessing import Imputer, PolynomialFeatures, StandardScaler, OneHotEncoder, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')
import gc

os.getcwd()
pd.options.display.max_columns = None


class datacleaner:
    def __init__(self, trainfile, targetcol, cat_threshold=100):
        # self.df_test = pd.read_csv(testfile,index_col=0)
        # print(self.df_test.columns)
        self.df_train = pd.read_csv(trainfile)
        self.target = targetcol
        self.dfcolumns = self.df_train.columns.tolist()
        self.dfcolumns_nottarget = [col for col in self.dfcolumns if col != self.target]
        self.rejectcols = []
        self.retainedcols = []
        self.catcols = {}
        self.noncatcols = {}
        self.catcols_list = []
        self.noncatcols_list = []
        self.hightarge_corr_col = []
        self.threshold = cat_threshold

    def retail_reject_cols(self, threshold):
        def retail_reject_cols_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    # print('start')
                    for col in tmpdf:
                        if sum(tmpdf[col].isnull()) > 0:
                            percent_val = sum(tmpdf[col].isnull()) / tmpdf[col].shape[0]
                            # print(f"For {col} number of nulls are {sum(df[col].isnull())} : {sum(df[col].isnull())/df[col].shape[0]}")
                            if percent_val > threshold:
                                self.rejectcols.append(col)
                    self.retainedcols = [col for col in tmpdf.columns.tolist() if col not in self.rejectcols]
                    print(f"INFO : {str(datetime.now())} : Number of rejected columns {len(self.rejectcols)}")
                    print(f"INFO : {str(datetime.now())} : Number of retained columns {len(self.retainedcols)}")
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return retail_reject_cols_lvl

    # def init_values(self,)
    def refresh_cat_noncat_cols_fn(self, tmpdf, threshold=100):
        try:
            self.catcols = {}
            self.catcols_list = []
            self.noncatcols = {}
            self.noncatcols_list = []
            self.dfcolumns = tmpdf.columns.tolist()
            self.dfcolumns_nottarget = [col for col in self.dfcolumns if col != self.target]
            for col in self.dfcolumns_nottarget:
                col_unique_cnt = tmpdf[col].nunique()
                if (col_unique_cnt < threshold) and (
                        (tmpdf[col].dtype != 'float32') and (tmpdf[col].dtype != 'float64')):
                    self.catcols.update({col: col_unique_cnt})
                    self.catcols_list.append(col)
                else:
                    self.noncatcols.update({col: col_unique_cnt})
                    self.noncatcols_list.append(col)
        except:
            sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

    def refresh_cat_noncat_cols(self, threshold):
        def refresh_cat_noncat_cols_lvl1(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    self.catcols = {}
                    self.catcols_list = []
                    self.noncatcols = {}
                    self.noncatcols_list = []
                    self.dfcolumns = tmpdf.columns.tolist()
                    self.dfcolumns_nottarget = [col for col in self.dfcolumns if col != self.target]
                    for col in self.dfcolumns_nottarget:
                        col_unique_cnt = tmpdf[col].nunique()
                        if (col_unique_cnt < threshold) and (
                                (tmpdf[col].dtype != 'float32') and (tmpdf[col].dtype != 'float64')):
                            self.catcols.update({col: col_unique_cnt})
                            self.catcols_list.append(col)
                        else:
                            self.noncatcols.update({col: col_unique_cnt})
                            self.noncatcols_list.append(col)
                    #print('ref' + str(len(self.noncatcols_list)))
                    #print('ref' + str(self.noncatcols_list))
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return refresh_cat_noncat_cols_lvl1

    def findcatcols(self, th):
        for col in self.dfcolumns_nottarget:
            col_unique_cnt = self.df_train[col].nunique()
            if col_unique_cnt < th:
                # print(f"{col} is category with unique values {df[col].nunique()}")
                self.catcols.update({col: col_unique_cnt})
            else:
                self.noncatcols.update({col: col_unique_cnt})
        print(f"INFO : {str(datetime.now())} : Number of categorical column is {len(self.catcols.keys())}")
        print(f"INFO : {str(datetime.now())} : Number of Non categorical column is {len(self.noncatcols.keys())}")
        return self.catcols, self.noncatcols

    def standardize_stratified(self, includestandcols=[]):
        def standardize_stratified_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    for col in tmpdf:
                        if tmpdf[col].dtype == 'float32' or tmpdf[col].dtype == 'float64' or (col in includestandcols):
                            tmpdf[col] = tmpdf[col].replace(np.inf, 0.0)
                            if tmpdf[col].mean() > 1000:
                                scaler = MinMaxScaler(feature_range=(0, 10))
                                tmpdf[col] = scaler.fit_transform(np.asarray(tmpdf[col]).reshape(-1, 1))
                            elif tmpdf[col].mean() > 100:
                                scaler = MinMaxScaler(feature_range=(0, 5))
                                # print(col)
                                tmpdf[col] = scaler.fit_transform(np.asarray(tmpdf[col]).reshape(-1, 1))
                            elif tmpdf[col].mean() > 10:
                                scaler = MinMaxScaler(feature_range=(0, 2))
                                # print(col)
                                tmpdf[col] = scaler.fit_transform(np.asarray(tmpdf[col]).reshape(-1, 1))
                            else:
                                scaler = MinMaxScaler(feature_range=(0, 1))
                                # print(col)
                                tmpdf[col] = scaler.fit_transform(np.asarray(tmpdf[col]).reshape(-1, 1))
                            print("INFO : " + str(datetime.now()) + ' : ' + 'Column ' + col + 'is standardized')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return standardize_stratified_lvl

    def featurization(self, cat_coltype=False):
        def featurization_lvl1(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    self.refresh_cat_noncat_cols_fn(tmpdf, self.threshold)
                    if cat_coltype:
                        column_list = self.catcols_list
                    else:
                        column_list = self.noncatcols_list
                    print("INFO : " + str(datetime.now()) + ' : ' + 'Shape of dataframe before featurization ' + str(
                        tmpdf.shape))
                    for col in column_list:
                        tmpdf[col + '_minus_mean'] = tmpdf[col] - np.mean(tmpdf[col])
                        tmpdf[col + '_minus_mean'] = tmpdf[col + '_minus_mean'].astype(np.float32)
                        tmpdf[col + '_minus_max'] = tmpdf[col] - np.max(tmpdf[col])
                        tmpdf[col + '_minus_max'] = tmpdf[col + '_minus_max'].astype(np.float32)
                        tmpdf[col + '_minus_min'] = tmpdf[col] - np.min(tmpdf[col])
                        tmpdf[col + '_minus_min'] = tmpdf[col + '_minus_min'].astype(np.float32)
                    print("INFO : " + str(datetime.now()) + ' : ' + 'Shape of dataframe after featurization ' + str(
                        tmpdf.shape))
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return featurization_lvl1

    def two_column_featurization(self, cat_coltype=False):
        def featurization_lvl1(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    self.refresh_cat_noncat_cols_fn(tmpdf, self.threshold)
                    if cat_coltype:
                        column_list = self.catcols_list
                    else:
                        column_list = self.noncatcols_list
                    # print('2col'+str(tmpdf.shape))
                    # print('2col'+str(tmpdf.columns))
                    # print('2col'+str(self.dfcolumns))
                    # print('2col'+str(column_list))
                    print("INFO : " + str(
                        datetime.now()) + ' : ' + 'Shape of dataframe before two column featurization ' + str(
                        tmpdf.shape))
                    for col in column_list:
                        for col1 in column_list:
                            if col != col1:
                                tmpdf[col + '_diff_' + col1] = tmpdf[col] - tmpdf[col1]
                                tmpdf[col + '_diff_' + col1] = tmpdf[col + '_diff_' + col1].astype(np.float32)
                                tmpdf[col + '_sum_' + col1] = tmpdf[col] + tmpdf[col1]
                                tmpdf[col + '_sum_' + col1] = tmpdf[col + '_sum_' + col1].astype(np.float32)
                                tmpdf[col + '_ratio_' + col1] = tmpdf[col] / tmpdf[col1]
                                tmpdf[col + '_ratio_' + col1] = tmpdf[col + '_ratio_' + col1].astype(np.float32)
                    print("INFO : " + str(
                        datetime.now()) + ' : ' + 'Shape of dataframe after two column featurization ' + str(
                        tmpdf.shape))
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return featurization_lvl1

    def standardize_simple(self, cat_coltype=False, range_tuple=(0, 1)):
        def standardize_simple_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    self.refresh_cat_noncat_cols_fn(tmpdf, self.threshold)
                    if cat_coltype:
                        column_list = self.catcols_list
                    else:
                        column_list = self.noncatcols_list
                    scaler = MinMaxScaler(feature_range=range_tuple)
                    # tmpdf_col = [col for col in tmpdf.columns.tolist(),]
                    for col in column_list:
                        tmpdf[col] = tmpdf[col].replace(np.inf, 0.0)
                        tmpdf[col] = scaler.fit_transform(np.asarray(tmpdf[col]).reshape(-1, 1))
                        print("INFO : " + str(datetime.now()) + ' : ' + 'Column ' + col + 'is standardized')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return standardize_simple_lvl

    def standardize_simple_auto(self, range_tuple=(0, 1)):
        def standardize_simple_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    scaler = MinMaxScaler(feature_range=range_tuple)
                    # tmpdf_col = [col for col in tmpdf.columns.tolist(),]
                    for col in tmpdf:
                        if (tmpdf[col].dtype == 'float32') or (tmpdf[col].dtype == 'float64'):
                            tmpdf[col] = tmpdf[col].replace(np.inf, 0.0)
                            tmpdf[col] = tmpdf[col].replace(np.nan, 0.0)
                            tmpdf[col] = tmpdf[col].replace(-np.inf, 0.0)
                            tmpdf[col] = scaler.fit_transform(np.asarray(tmpdf[col]).reshape(-1, 1))
                            print("INFO : " + str(datetime.now()) + ' : ' + 'Column ' + col + 'is standardized')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return standardize_simple_lvl

    def columnmapondf(self, col_list_of_dict=[['colname', {}], ['colname', {}]]):
        def columnmapondf_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    for col in col_list_of_dict:
                        # print(mapdict)
                        # print(col)
                        convertedcolumns = tmpdf[col[0]].map(col[1])
                        tmpdf[col] = convertedcolumns
                        print("INFO : " + str(
                            datetime.now()) + ' : ' + 'Column ' + col[0] + ',mapped using dictionary : ' + str(col[1]))
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return columnmapondf_lvl

    def dropcolumnfromdf(self, column_list=[]):
        def dropcolumnfromdf_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    tmpdf = tmpdf.drop(column_list, axis=1)
                    print("INFO : " + str(datetime.now()) + ' : ' + 'Column ' + str(
                        column_list) + ',dropped from base dataframe')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return dropcolumnfromdf_lvl

    def feature_importance(self, dfforimp, tobepredicted, modelname, featurelimit=0):
        colname = [col for col in dfforimp.columns.tolist() if col != tobepredicted]
        X = dfforimp[colname]
        y = dfforimp[tobepredicted]
        #print(modelname)
        #t =''
        if modelname == 'rfclassifier':
            model = RandomForestClassifier(n_estimators=100, random_state=10)
        elif modelname == 'rfregressor':
            model = RandomForestRegressor(n_estimators=100, random_state=10)
        elif modelname == 'lgbmclassifier':
            model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, verbose=-1)
        elif modelname == 'lgbmregressor':
            #print('yes')
            model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1)
        model.fit(X, y)
        feature_names = X.columns
        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
        feature_importances = feature_importances.sort_values(by=['importance'], ascending=False).reset_index()
        feature_importances = feature_importances[['feature', 'importance']]
        if featurelimit == 0:
            return feature_importances
        else:
            return feature_importances[:featurelimit]

    def importantfeatures(self, dfforimp, tobepredicted, modelname, skipcols=[], featurelimit=0):
        #print(modelname)
        f_imp = self.feature_importance(dfforimp, tobepredicted, modelname, featurelimit)
        allimpcols = list(f_imp['feature'])
        stuff = []
        for col in allimpcols:
            for skipcol in skipcols:
                if col != skipcol:
                    stuff.append(col)
                else:
                    pass
        return stuff, f_imp

    def convertdatatypes(self, cat_threshold=100):
        def convertdatatypes_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    original_memory = tmpdf.memory_usage().sum()
                    print("INFO : " + str(
                        datetime.now()) + ' : ' + 'Size of dataset before applying converdatatypes ' + str(
                        original_memory))
                    for c in tmpdf:
                        if ('y' in c):
                            tmpdf[c] = tmpdf[c].fillna(0).astype(np.int32)
                        elif (tmpdf[c].dtype == 'object') and (tmpdf[c].nunique() < cat_threshold):
                            tmpdf[c] = tmpdf[c].astype('category')
                            # elif (tmpdf[c].nunique() < cat_threshold):
                        #    tmpdf[c] = tmpdf[c].astype('category')
                        elif tmpdf[c].dtype == float:
                            tmpdf[c] = tmpdf[c].astype(np.float32)
                        elif tmpdf[c].dtype == int:
                            tmpdf[c] = tmpdf[c].astype(np.int32)
                    new_memory = tmpdf.memory_usage().sum()
                    print("INFO : " + str(
                        datetime.now()) + ' : ' + 'Size of dataset after applying converdatatypes ' + str(new_memory))
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return convertdatatypes_lvl

    def sumtwocolumns(self, combination_list=[['col1', 'col2', 'col1+col2']]):
        def sumtwocolumns_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    for col in combination_list:
                        tmpdf[col[2]] = tmpdf[col[0]] + tmpdf[col[1]]
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return sumtwocolumns_lvl

    def difftwocolumns(self, combination_list=[['col1', 'col2', 'col1+col2']]):

        def difftwocolumns_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    for col in combination_list:
                        tmpdf[col[2]] = tmpdf[col[0]] + tmpdf[col[1]]
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return difftwocolumns_lvl

    def ohe_on_threshold(self, threshold, drop_converted_col=True):
        def ohe_on_threshold_lvl1(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    for col in tmpdf.columns.tolist():
                        if tmpdf[col].nunique() < threshold:
                            dummy = pd.get_dummies(tmpdf[col])
                            tmpdf = pd.concat([tmpdf, dummy], axis=1)
                            if drop_converted_col:
                                tmpdf = tmpdf.drop(col, axis=1)
                            print("INFO : " + str(datetime.now()) + ' : ' + 'Column ' + col + ' converted to dummies')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return ohe_on_threshold_lvl1

    def remove_collinear(self, th=0.95):
        def converttodummies_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    print("INFO : " + str(datetime.now()) + ' : ' + 'Shape of dataframe before collinear drop ' + str(
                        tmpdf.shape))
                    corr_matrix = tmpdf.corr().abs()
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                    to_drop = [column for column in upper.columns if any(upper[column] > th)]
                    tmpdf = tmpdf.drop(to_drop, axis=1)
                    print("INFO : " + str(datetime.now()) + ' : ' + 'Shape of dataframe after collinear drop ' + str(
                        tmpdf.shape))
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return converttodummies_lvl

    def high_coor_target_column(self, targetcol='y', th=0.5):
        def high_coor_target_column_lvl1(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    print("INFO : " + str(
                        datetime.now()) + ' : ' + 'Shape of dataframe before retaining only highly corelated col with target ' + str(
                        tmpdf.shape))
                    cols = [col for col in tmpdf.columns.tolist() if col != targetcol]
                    for col in cols:
                        tmpdcorr = tmpdf[col].corr(tmpdf[targetcol])
                        if tmpdcorr > th:
                            self.hightarge_corr_col.append(col)
                    cols = self.hightarge_corr_col + [targetcol]
                    tmpdf = tmpdf[cols]
                    print("INFO : " + str(
                        datetime.now()) + ' : ' + 'Shape of dataframe after retaining only highly corelated col with target ' + str(
                        tmpdf.shape))
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return high_coor_target_column_lvl1

    def ohe_on_column(self, drop_converted_col=True):
        def converttodummies_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    # self.refresh_cat_noncat_cols_fn(tmpdf,self.threshold)
                    column_list = self.catcols_list
                    for col in column_list:
                        dummy = pd.get_dummies(tmpdf[col])
                        dummy.columns = [col.lower() + '_' + str(x).lower().strip() for x in dummy.columns]
                        tmpdf = pd.concat([tmpdf, dummy], axis=1)
                        # tmpdf.columns = [col.lower()+'_'+str(x).lower().strip() for x in tmpdf.columns]
                        if drop_converted_col:
                            tmpdf = tmpdf.drop(col, axis=1)
                        print("INFO : " + str(datetime.now()) + ' : ' + 'Column ' + col + ' converted to dummies')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return converttodummies_lvl

    def logtransform(self, logtransform_col=[]):
        def logtransform_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    for col in logtransform_col:
                        tmpdf[col] = tmpdf[col].apply(lambda x: np.log(x) if x != 0 else 0)
                        print("INFO : " + str(
                            datetime.now()) + ' : ' + 'Column ' + col + ' converted to corresponding log using formula: log(x)')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return logtransform_lvl

    def applyformulaoncolumns(self, column_formula=[]):
        def applyformulaoncolumns_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    for col, formula in column_formula.items():
                        formulabuilder = 'tmpdf[' + col + '].apply(lambda x: ' + formula + ' if x!=0 else 0)'
                        tmpdf[col] = eval(formulabuilder)
                        print("INFO : " + str(
                            datetime.now()) + ' : ' + 'Column ' + col + ' converted to corresponding log using formula:' + formula)
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return applyformulaoncolumns_lvl

    def applypca(self, cat_coltype=False, number_of_components=50):
        def applypca_lvl1(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    self.refresh_cat_noncat_cols_fn(tmpdf, self.threshold)
                    if cat_coltype:
                        column_list = self.catcols_list
                    else:
                        column_list = self.noncatcols_list
                    print("INFO : " + str(datetime.now()) + ' : ' + 'Shape of dataframe before pca ' + str(tmpdf.shape))
                    non_included_cols = [col for col in self.dfcolumns_nottarget if col not in column_list]
                    non_included_cols = non_included_cols + [self.target]
                    x = tmpdf.loc[:, column_list].values
                    # y = tmpdf.loc[:,[self.target]].values
                    # x = StandardScaler().fit_transform(x)
                    pca = PCA(n_components=number_of_components)
                    # pca = PCA()
                    pca_comp = pca.fit_transform(x)
                    pca_df = pd.DataFrame(data=pca_comp, columns=['pca_' + str(i) for i in range(number_of_components)])
                    # pca_df = pd.DataFrame(data = pca_comp)
                    # print(non_included_cols)
                    tmpdf = pd.concat([pca_df, tmpdf[non_included_cols]], axis=1)
                    # print(tmpdf1.shape)
                    # print(non_included_cols)
                    # print(tmpdf1[non_included_cols].shape)
                    # tmpdf = pd.concat([tmpdf1,tmpdf[[self.target]]], axis = 1)
                    self.explained_variance = pca.explained_variance_ratio_
                    print("INFO : " + str(datetime.now()) + ' : ' + 'Shape of dataframe after pca ' + str(tmpdf.shape))
                    gc.enable()
                    del x, pca_comp, pca_df
                    gc.collect()
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return applypca_lvl1

    def applytsne(self, cat_coltype=False, number_of_components=50):
        def applytsne_lvl1(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    self.refresh_cat_noncat_cols_fn(tmpdf, self.threshold)
                    if cat_coltype:
                        column_list = self.catcols_list
                    else:
                        column_list = self.noncatcols_list
                    print(
                        "INFO : " + str(datetime.now()) + ' : ' + 'Shape of dataframe before tsne ' + str(tmpdf.shape))
                    tmpdf_col = tmpdf.columns.tolist()
                    non_included_cols = [col for col in self.dfcolumns if col not in column_list]
                    x = tmpdf.loc[:, column_list].values
                    # y = tmpdf.loc[:,[self.target]].values
                    # x = StandardScaler().fit_transform(x)
                    tsne = TSNE(n_components=number_of_components)
                    tsne_comp = tsne.fit_transform(x)
                    tsne_df = pd.DataFrame(data=tsne_comp,
                                           columns=['tsne_' + str(i) for i in range(number_of_components)])
                    tmpdf = pd.concat([tsne_df, tmpdf[non_included_cols]], axis=1)
                    # print(tmpdf1.shape)
                    # tmpdf = pd.concat([tmpdf1,tmpdf[[self.target]]], axis = 1)
                    # print("INFO : " + str(datetime.now()) + ' : ' + 'Shape of dataframe after tsne ' + str(tmpdf.shape))
                    gc.enable()
                    del x, tsne_comp, tsne_df
                    gc.collect()
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return applytsne_lvl1

    def convertnulls_to(self, columns_list=[], to_val=''):
        def convertnullsto_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    tmpdf.update(tmpdf[columns_list].fillna(to_val))
                    print("INFO : " + str(
                        datetime.now()) + ' : ' + 'Columns : ' + columns_list + ' converted to ' + str(to_val))
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return convertnullsto_lvl

    def addbackcolumn(self, stage):
        dictargname_col = 'addbackcolumn_col_' + str(stage)
        dictargname_map = 'addbackcolumn_map_' + str(stage)
        addbackcolumn_col = self.colnamelistdict[dictargname_col]
        addbackcolumn_map = self.maplistdict[dictargname_map]

        def addbackcolumn_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    for col in addbackcolumn_col:
                        if addbackcolumn_map == 'train':
                            tmpdf = pd.concat([tmpdf, self.df_train[col]], axis=1)
                        elif addbackcolumn_map == 'test':
                            tmpdf = pd.concat([tmpdf, self.df_test[col]], axis=1)
                        else:
                            print("INFO : " + str(
                                datetime.now()) + ' : ' + 'Unable to add column.Please enter test or train in input sheet')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return addbackcolumn_lvl

    def testtrainna(self, tmpdf, tobepred_col):
        key = tmpdf[tobepred_col].isnull()
        df_na = tmpdf.loc[key]
        df_notNA_train_x = pd.DataFrame()
        df_notNA_train_y = pd.DataFrame()
        df_NA_test_x = pd.DataFrame()
        df_NA_test_y = pd.DataFrame()
        try:
            # print(df_na.shape)
            if df_na.empty:
                print(
                    "INFO : " + str(
                        datetime.now()) + ' : ' + 'Column ' + tobepred_col + 'has no null values so imputing cancelled')
                return df_notNA_train_x, df_notNA_train_y, df_NA_test_x, df_NA_test_y
            else:
                df_notNA = tmpdf.loc[~key]
                df_notNA_train_x = df_notNA.dropna(how='any')
                df_notNA_train_y = df_notNA_train_x[tobepred_col]
                df_notNA_train_x = df_notNA_train_x.drop(tobepred_col, axis=1)
                df_NA_test_y = df_na[tobepred_col]
                df_NA_test_x = df_na.drop(tobepred_col, axis=1)
                df_NA_test_x = df_NA_test_x.ffill().bfill()
                df_NA_test_x = df_NA_test_x.dropna(axis=1, how='all')
                df_notNA_train_x = df_notNA_train_x[df_NA_test_x.columns]
            return df_notNA_train_x, df_notNA_train_y, df_NA_test_x, df_NA_test_y
        except:
            sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

    def model_null_impute_notcat_dt(self):
        def model_null_impute_notcat_dt_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    self.refresh_cat_noncat_cols_fn(tmpdf, self.threshold)
                    column_list = self.catcols_list
                    for col in column_list:
                        # print(col)
                        df_notNA_train_x, df_notNA_train_y, df_NA_test_x, df_NA_test_y = self.testtrainna(tmpdf, col)
                        if df_notNA_train_x.empty:
                            pass
                            # print("INFO : " + str(datetime.now()) + ' : ' + 'imputing cancelled')
                        else:
                            dtree = DecisionTreeRegressor()
                            # print(df_notNA_train_x.columns[df_notNA_train_x.isna().any()].tolist())
                            # print(df_notNA_train_y.columns[df_notNA_train_y.isna().any()].tolist())
                            # print(df_notNA_train_x.shape, df_notNA_train_y.shape)
                            dtree.fit(df_notNA_train_x, df_notNA_train_y)
                            predictions = dtree.predict(df_NA_test_x)
                            df_NA_test_x[col] = predictions
                            tmpdf[col].update(df_NA_test_x[col])
                            print("INFO : " + str(
                                datetime.now()) + ' : ' + 'Column ' + col + ',data imputing completed : No. of data imputed = ' + str(
                                df_NA_test_x.shape[0]))
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return model_null_impute_notcat_dt_lvl

    def model_null_impute_cat_dt(self):
        def model_null_impute_notcat_dt_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    self.refresh_cat_noncat_cols_fn(tmpdf, self.threshold)
                    column_list = self.noncatcols_list
                    for col in column_list:
                        # print(col)
                        df_notNA_train_x, df_notNA_train_y, df_NA_test_x, df_NA_test_y = self.testtrainna(tmpdf, col)
                        if df_notNA_train_x.empty:
                            pass
                            # print("INFO : " + str(datetime.now()) + ' : ' + 'imputing cancelled')
                        else:
                            dtree = DecisionTreeClassifier()
                            # print(df_notNA_train_x.columns[df_notNA_train_x.isna().any()].tolist())
                            # print(df_notNA_train_y.columns[df_notNA_train_y.isna().any()].tolist())
                            # print(df_notNA_train_x.shape, df_notNA_train_y.shape)
                            dtree.fit(df_notNA_train_x, df_notNA_train_y)
                            predictions = dtree.predict(df_NA_test_x)
                            # print(predictions)
                            df_NA_test_x[col] = predictions
                            tmpdf[col].update(df_NA_test_x[col])
                            print("INFO : " + str(
                                datetime.now()) + ' : ' + 'Column ' + col + ',data imputing completed : No. of data imputed = ' + str(
                                df_NA_test_x.shape[0]))
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return model_null_impute_notcat_dt_lvl

    def model_null_impute_notcat_rf(self):
        def model_null_impute_notcat_rf_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    self.refresh_cat_noncat_cols_fn(tmpdf, self.threshold)
                    column_list = self.noncatcols_list
                    for col in column_list:
                        # df_notNA_train_x, df_notNA_train_y, df_NA_test_x, df_NA_test_y = self.testtrainna(tmpdf, col)
                        df_notNA_train_x, df_notNA_train_y, df_NA_test_x, df_NA_test_y = self.testtrainna(tmpdf, col)
                        if df_notNA_train_x.empty:
                            pass
                            # print("INFO : " + str(datetime.now()) + ' : ' + 'imputing cancelled')
                        else:
                            rfc = RandomForestRegressor()
                            rfc.fit(df_notNA_train_x, df_notNA_train_y)
                            predictions = rfc.predict(df_NA_test_x)
                            df_NA_test_x[col] = predictions
                            tmpdf[col].update(df_NA_test_x[col])
                            print("INFO : " + str(
                                datetime.now()) + ' : ' + 'Column ' + col + ',data imputing completed : No. of data imputed = ' + str(
                                df_NA_test_x.shape[0]))
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return model_null_impute_notcat_rf_lvl

    def model_null_impute_cat_rf(self):

        def model_null_impute_cat_rf_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    self.refresh_cat_noncat_cols_fn(tmpdf, self.threshold)
                    column_list = self.catcols_list
                    for col in column_list:
                        df_notNA_train_x, df_notNA_train_y, df_NA_test_x, df_NA_test_y = self.testtrainna(tmpdf, col)
                        if df_notNA_train_x.empty:
                            pass
                            # print("INFO : " + str(datetime.now()) + ' : ' + 'imputing cancelled')
                        else:
                            rfc = RandomForestClassifier()
                            rfc.fit(df_notNA_train_x, df_notNA_train_y)
                            predictions = rfc.predict(df_NA_test_x)
                            df_NA_test_x[col] = predictions
                            tmpdf[col].update(df_NA_test_x[col])
                            print("INFO : " + str(
                                datetime.now()) + ' : ' + 'Column ' + col + ',data imputing completed : No. of data imputed = ' + str(
                                df_NA_test_x.shape[0]))
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return model_null_impute_cat_rf_lvl

    def add_id_column(self,id_col):
        def add_id_column_lvl1(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    tmpdf[id_col] = tmpdf.index +1
                    print("INFO : " + str(datetime.now()) + ' : ' + 'Added id column')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])
            return wrapper
        return add_id_column_lvl1

    def split_test_train(self,df,test_size=0.2):
        try:
            train,test = train_test_split(df,test_size=test_size,random_state=120)
            train = train.reset_index(drop=True)
            test = test.reset_index(drop=True)
            print("INFO : " + str(datetime.now()) + ' : ' + 'Train dataframe shape '+str(train.shape))
            print("INFO : " + str(datetime.now()) + ' : ' + 'Test dataframe shape ' + str(test.shape))
            return train,test
        except:
            sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

    def join_test_train(self,train,test):
        try:
            train['type'] = 'train'
            test['type'] = 'test'
            maindf = pd.concat([train,test])
            maindf = maindf.reset_index(drop=True)
            print("INFO : " + str(datetime.now()) + ' : ' + 'Combined dataframe shape '+str(maindf.shape))
            return maindf
        except:
            sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

    def df_onlyselected(self, maindf,selected_columns):
        try:
            print("INFO : " + str(datetime.now()) + ' : ' + 'Dataframe size before selecting needed columns' +
                  str(maindf.shape))
            maindf = maindf[selected_columns]
            print("INFO : " + str(datetime.now()) + ' : ' + 'Dataframe size after selecting needed columns' +
                  str(maindf.shape))
            return maindf
        except:
            sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])


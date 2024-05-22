import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import plotly.graph_objs as go
import xgboost as xgb
import pickle
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from anndata import read_h5ad, concat, AnnData
# import plotly.graph_objects as go
# from itertools import cycle, islice
import pyreadr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 
from tqdm import tqdm
from copy import deepcopy
import scanpy as sc
import os
from openpyxl import load_workbook

def read_xlsx(file, sheet_name=None, header=None):
    excel = pd.ExcelFile(load_workbook(file), engine="openpyxl")
    sheet_name = sheet_name or excel.sheet_names[0]
    sheet = excel.book[sheet_name]
    df = excel.parse(sheet_name, header=header)

    for item in sheet.merged_cells:
        top_col, top_row, bottom_col, bottom_row = item.bounds
        base_value = item.start_cell.value
        top_row -= 1
        top_col -= 1
        if header is not None:
            top_row -= header + 1
            bottom_row -= header + 1
        df.iloc[top_row:bottom_row, top_col:bottom_col] = base_value
    return df


def read_data(data_name,ind_annotation, j):
    adata = read_h5ad(f"fulldata/{data_name[j]}.h5ad")
    df = read_xlsx(ind_annotation, sheet_name=ind_annotation.sheet_names[j], header=0)
    # adata_gw = read_h5ad(ind_annotation.sheet_names[j].lower() + ".h5ad")
    df = df[["H1", "H1 annotation", "H2", "H2 annotation"]]
    df = df[(df["H1 annotation"] != "Artifact") & (df["H2 annotation"] != "artifact, remove") & (df["H2 annotation"] != "artifact remove")]
    h1_annotation = {int(i): df["H1 annotation"][df["H1"] == i].unique()[0] for i in np.unique(df["H1"])}
    h2_annotation = {int(i): df["H2 annotation"][df["H2"] == i].unique()[0] for i in np.unique(df["H2"])}
    adata.obs["H1_annotation"] = adata.obs["H1_cluster"].map(h1_annotation)
    adata.obs["H2_annotation"] = adata.obs["H2_cluster"].map(h2_annotation)
    adata = adata[~pd.isna(adata.obs["H1_annotation"])]
    adata = adata[~pd.isna(adata.obs["H2_annotation"])]
    adata = adata[adata.obs["H1_annotation"].isin(["EN-ET", "EN-IT", "EN-IT-1", "EN-IT-2"])]
    adata.obs = adata.obs.fillna(value={"H3_cluster": 0})
    adata.obs["H3_cluster"] = adata.obs["H3_cluster"].astype(int)
    adata.obs["H3_cluster"] = adata.obs["H3_cluster"] + 1
    return adata


def resample(data, cluster="EN-ET", threshold=7000):
    cluster_num = pd.Series(data[data.obs.H1_annotation == cluster].obs["H3_annotation"]).value_counts()
    resample_cluster = list(cluster_num[cluster_num <= threshold].index)
    cluster_ind = list(cluster_num.index)
    data_ind = []
    for i in range(len(cluster_ind)):
        data_cluster = data[data.obs.H3_annotation == cluster_ind[i]]
        if cluster_ind[i] in resample_cluster:
            resample_ind = np.random.choice(range(data_cluster.shape[0]), threshold - data_cluster.shape[0], replace=True)
            data_cluster = concat([data_cluster, data_cluster[resample_ind, :]])
            data_cluster.obs_names_make_unique()
        if i == 0:
            data_tot = data_cluster.copy()
        else:
            data_tot = concat([data_tot, data_cluster])
    data_tot.obs.H3_annotation = data_tot.obs.H3_annotation.astype("category")
    print(pd.Series(data_tot[data_tot.obs.H1_annotation == cluster].obs["H3_annotation"]).value_counts())
    return data_tot


def trainmodel(gw_train_tot, gw_test_tot, cluster="EN-ET", nround=200, thre=1500):
    gw_train_tot = gw_train_tot[gw_train_tot.obs.H1_annotation == cluster]
    gw_test_tot = gw_test_tot[gw_test_tot.obs.H1_annotation == cluster]
    
    Rdata_ind, Qdata_ind, _, _ =train_test_split(range(gw_train_tot.shape[0]), gw_train_tot.obs['H3_annotation'], test_size=0.2,random_state=1,stratify=gw_train_tot.obs['H3_annotation'])
    gw_train = gw_train_tot[np.sort(Rdata_ind), :]
    gw_test = gw_train_tot[np.sort(Qdata_ind), :]
    
    gw_train = resample(gw_train, cluster=cluster, threshold= int(thre*0.8))
    train_xgb = xgb.DMatrix(gw_train.X, label=gw_train.obs.H3_annotation.cat.codes.values)
    xgb_params_train = {
                'objective':'multi:softmax',
                'eval_metric':'mlogloss',
                'num_class': len(np.unique(gw_train.obs.H3_annotation)),
                'eta':0.2,
                'max_depth':20,
                'subsample': 0.6}

    bst_model_train = xgb.train(
                params = xgb_params_train,
                dtrain = train_xgb,
                num_boost_round = nround)

    label = gw_train.obs.H3_annotation.cat.categories
    test_xgb = xgb.DMatrix(gw_test.X)
    test_pred = bst_model_train.predict(test_xgb)
    gw_test.obs['pred'] = label[test_pred.astype(int)]
    score = metrics.accuracy_score(gw_test.obs.H3_annotation, gw_test.obs.pred)
    crosstab = pd.crosstab(gw_test.obs.H3_annotation, gw_test.obs.pred)

    ## retrain using all data
    gw_train_tot = resample(gw_train_tot, cluster=cluster, threshold=thre)
    train_xgb = xgb.DMatrix(gw_train_tot.X, label=gw_train_tot.obs.H3_annotation.cat.codes.values)
    bst_model_train = xgb.train(
            params = xgb_params_train,
            dtrain = train_xgb,
            num_boost_round = nround)
    test_xgb = xgb.DMatrix(gw_test_tot.X)
    test_pred = bst_model_train.predict(test_xgb)
    gw_test_tot.obs['pred'] = label[test_pred.astype(int)]
    crosstab_test = pd.crosstab(gw_test.obs.H3_annotation, gw_test.obs.pred)
    return gw_test_tot, crosstab_test, score, crosstab


def get_result(gw_tot, cluster):
    gw_result_tot = []
    crosstab_test_tot = []
    score_tot = []
    crosstab_tot = []
    for i in tqdm(range(len(gw_tot) - 1)):
        # gw_test_tot, crosstab_test, score, crosstab = trainmodel(gw_tot[i+1], gw_tot[i], cluster=cluster, nround=100, thre=1500)
        gw_test_tot, crosstab_test, score, crosstab = trainmodel(gw_tot[i], gw_tot[i+1], cluster=cluster, nround=1000, thre=15000)
        gw_result_tot.append(gw_test_tot)
        crosstab_test_tot.append(crosstab_test)
        score_tot.append(score)
        crosstab_tot.append(crosstab)
    return gw_result_tot, crosstab_test_tot, score_tot

def flatten(ll):
    if isinstance(ll, list):
        for i in ll:
            for element in flatten(i):
                yield element
    else:
        yield ll

def getData(gw_test_tot, sourcekey, targetkey, filter_prop):
    source = []
    target = []
    source_gw = []
    target_gw = []
    value = []  
    # color_link = []
    # color_dict = {}
    gw_lst = ["gw15", "gw20", "gw22", "gw34"]
    # gw_lst = ["gw34", "gw22", "gw20", "gw15"]    
    source_dict = {}
    node_label = []
    
    for k in range(len(gw_test_tot)):
        data = gw_test_tot[k].obs
        s = np.sort(np.unique(data[sourcekey]))
        t = np.sort(np.unique(data[targetkey]))
        if k == 0:
            source_dict[gw_lst[k]] = {s[m]: m for m in range(len(s))}
            # color_dict[gw_lst[k]] = {s[m]: color_lst[m] for m in range(len(s))}
            node_label.append(list(s))

        begin = np.max(list(list(source_dict.values())[len(source_dict)-1].values()))+1
        source_dict[gw_lst[k+1]] = {t[m]: m+begin for m in range(len(t))}
        # color_dict[gw_lst[k+1]] = {t[m]: color_lst[m+begin] for m in range(len(t))}
        node_label.append(list(t))

        for i in range(len(s)):
            source_filter = filter_prop * (data[ data[sourcekey] == s[i] ].shape[0])
            for j in range(len(t)):
                target_filter = filter_prop * (data[ data[targetkey] == t[j] ].shape[0])
                df_detail = data[(data[sourcekey] == s[i]) & (data[targetkey] == t[j])]
                if df_detail.shape[0] >= source_filter or df_detail.shape[0] >= target_filter:
                    source.append( source_dict[gw_lst[k]][s[i]] )
                    target.append( source_dict[gw_lst[k+1]][t[j]] )
                    source_gw.append( gw_lst[k] )
                    target_gw.append( gw_lst[k+1] )
                    value.append(df_detail.shape[0])
                    # color_link.append( color_dict[gw_lst[k]][s[i]] )
        
    table = pd.DataFrame({'source':source, 'target': target, 'value': value, 'source_gw': source_gw, 'target_gw': target_gw})
    table2 = table.groupby(['source','target']).sum()
    table2 = table2.reset_index() #.sort_values('value',ascending = False)
    node_label = list(flatten(node_label))
    # color = list(flatten([list(m.values()) for m in list(color_dict.values())]))
    return node_label, table2, table, source_dict

def savedata(data, h1name, filename):
    src_name = np.array(data[0])[data[3]['source'].values]
    trg_name = np.array(data[0])[data[3]['target'].values]
    df = data[4].copy()
    df['source'] = src_name
    df['target'] = trg_name
    df = df.rename(columns={"value": "Number of cells"})
    df = df[['source', 'source_gw', 'target', 'target_gw', 'Number of cells']]
    df.to_csv(f"/Users/shunzhou/Desktop/RA/Xuyu_Project/sankey/full/{h1name}/{filename}.csv")

def main():
    ### EN-ET
    ind_annotation = pd.ExcelFile("/Users/shunzhou/Desktop/RA/Xuyu_Project/segment_data/ind_clustering annotation.xlsx", engine="openpyxl")
    data_name = ["gw15", "gw20", "gw22", "gw34"]
    gw15 = read_data(data_name,ind_annotation, 0)
    gw20 = read_data(data_name,ind_annotation, 1)
    gw22 = read_data(data_name,ind_annotation, 2)
    gw34 = read_data(data_name,ind_annotation, 3)

    gw_lst = ["gw15", "gw20", "gw22", "gw34"]
    for gw in gw_lst:
        if "H3_annotation" in locals()[gw].obs.columns:
            locals()[gw].obs = locals()[gw].obs.drop(columns="H3_annotation")
        # locals()[gw].obs = locals()[gw].obs.join(shc_result_tot, how="left")
        locals()[gw].obs["H3_annotation"] = locals()[gw].obs["H2_annotation"].astype(str) + " Cluster " + locals()[gw].obs["H3_cluster"].astype(str)
        locals()[gw].obs["H3_annotation"] = "'" + locals()[gw].obs["H3_annotation"]
        
    gw_tot = [gw15, gw20, gw22, gw34]
    gw_result_tot_et, crosstab_test_tot_et, score_tot_et = get_result(gw_tot, cluster = "EN-ET")
    print(score_tot_et)
    data = getData(gw_result_tot_et, sourcekey="pred", targetkey="H3_annotation", filter_prop=0.2)
    savedata(data, h1name="EN-ET", filename=f"EN-ET")
    
    
    ### EN-IT
    gw22_copy = gw22.copy()
    gw22_copy.obs['H1_annotation'] = gw22_copy.obs['H1_annotation'].replace({'EN-IT-1': 'EN-IT', 'EN-IT-2': 'EN-IT'})
    gw34_copy = gw34.copy()
    gw34_copy.obs['H1_annotation'] = gw34_copy.obs['H1_annotation'].replace({'EN-IT-1': 'EN-IT', 'EN-IT-2': 'EN-IT'})

    gw_tot = [gw15, gw20, gw22_copy, gw34_copy]
    gw_result_tot_it, crosstab_test_tot_it, score_tot_it = get_result(gw_tot, cluster = "EN-IT")
    print(score_tot_it)
    data = getData(gw_result_tot_it, sourcekey="pred", targetkey="H3_annotation", filter_prop=0.2)
    savedata(data, h1name="EN-IT", filename = f"EN-IT")
    
if __name__ == "__main__":
    main()
import os, sys
import numpy as np
import pandas as pd

from IPython.display import display, HTML

def beautify_df_mc(mtx_mean, mtx_min, mtx_max, mtx_std):
    # Cria Matriz com as Informações
    row = list()
    for j in range(7):
        cell = f'{mtx_mean[j]} ± {mtx_std[j]} ({mtx_min[j]},{mtx_max[j]})'
        row.append(cell)
    df = pd.DataFrame([row])
    df.columns = ['AUC','FPR','ACC','BACC','PREC','REC','F1']
    return df

def gen_results_mc(mc_title, mc_views, sc_gen_basepath, vfetype, exec_sufix):
    print(mc_title)
    dfs = list()
    
    mtx_means, mtx_mins, mtx_maxs, mtx_stds = list(),list(),list(),list()
    for view in mc_views:
        base_path = sc_gen_basepath(view)
        
        
        alphas = os.listdir(base_path )
        
        rows_result = list()
        for _alpha in alphas:
            
            path_alpha = os.path.join(base_path, _alpha)
            lambdas = os.listdir(path_alpha)
            for lamb in lambdas:
        
                path_lambda = os.path.join(path_alpha, lamb, vfetype)
                execs = os.listdir(path_lambda)
    
                execs = [e for e in execs if exec_sufix in e]
                
                
                sample_size = [4,4,4,4,4]
                for index_exec, execution in enumerate(execs): 
                    path_execution = os.path.join(path_lambda, execution, 'result_metrics.txt')
        
                    with open(path_execution) as f:
                        lines = f.readlines()
                    
                    row_result = f"{_alpha}\t{lamb}\t{sample_size[index_exec]}\t{lines[-1]}"
                    #print(row_result, end="")
                    rows_result.append([float(s) for s in row_result.replace("\n","").split("\t")[5:] ])
        
        # Gera Matriz para Impressão
        mtx_mean = np.round(np.array(rows_result).mean(axis=0), decimals=2).ravel()
        mtx_std = np.round(np.array(rows_result).std(axis=0), decimals=2).ravel()
        mtx_min = np.round(np.array(rows_result).min(axis=0), decimals=2).ravel()
        mtx_max = np.round(np.array(rows_result).max(axis=0), decimals=2).ravel()
    
        mtx_means.append(mtx_mean)
        mtx_stds.append(mtx_std)
        mtx_mins.append(mtx_min)
        mtx_maxs.append(mtx_max)
        
        df = beautify_df_mc(mtx_mean, mtx_min, mtx_max, mtx_std)
        df.index = [view]
        dfs.append(df)
    
    #display(pd.concat(dfs))
    
    #### np.array(mtx_means).mean(axis=0)
    df_mean = beautify_df_mc(
        np.round(np.array(mtx_means).mean(axis=0),2), 
        np.round(np.array(mtx_mins).mean(axis=0),2), 
        np.round(np.array(mtx_maxs).mean(axis=0),2), 
        np.round(np.array(mtx_stds).mean(axis=0),2)
    )
    df_mean.index = ["mean"]
    #display(df_mean)
    
    display(pd.concat([
        pd.concat(dfs),
        df_mean
    ]))


gen_results_mc(
    mc_title = ". . .", 
    mc_views = ["2-4",], 
    sc_gen_basepath  = lambda view: os.path.join(
    f"/media/dev/LaCie/MC-VAD-MIL-OUT/hqfs-arnet-format-i3d",
    f"outputs-{view}/result/model_single/i3d/{view}/"), 
    vfetype = "rgb", 
    exec_sufix = "-LCMax-LFMax",
)
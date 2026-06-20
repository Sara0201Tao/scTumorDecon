# -*- coding:utf-8 -*-


import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from scipy.stats import pearsonr
from scipy.optimize import nnls
import argparse
import json

from .utils import *
from .model import *


def rename_from_filename(file):
    name = os.path.splitext(os.path.basename(file))[0]
    parts = name.split("_")
    parts = [p for p in parts if p not in ("train", "test")]
    new_name = "".join(p.capitalize() for p in parts)

    return new_name

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="MPL model train and predict for bulkRNAseq", add_help=add_help)
    
    # general 
    parser.add_argument("--device", default=r"cpu", type=str, help="computation device")
    parser.add_argument("--randomSeed", default= 42, type=int, help="random seed setting for reproducibility")
    # file info
    parser.add_argument("--trainFile-path", type=str, help="h5ad train data file path")
    parser.add_argument("--testFile-path", type=str, help="test data (h5ad or txt) file path")
    parser.add_argument("--resultFile-path", type=str, help="result file path")
    # model related
    parser.add_argument("--batchSize", default= 128, type=int, help="batch size in training process")
    
    parser.add_argument("--epochsMajor", default= 128, type=int, help="epochs in training major model process")
    parser.add_argument("--layerNumMajor", default= 4, type=int, help="layer numbers in major model")
    parser.add_argument("--hiddenDimMajor", default= "512, 256, 128, 64", type=str, help="hidden dims in major model")
    parser.add_argument("--dropoutMajor", default= "0, 0.3, 0.2, 0.1", type=str, help="dropouts in major model")
    parser.add_argument("--learningRateMajor", default= 1e-4, type=float, help="learning rate")
    parser.add_argument("--epsMajor", default= 1e-7, type=float, help="eps")
    
    parser.add_argument("--epochsMinor", default= 256, type=int, help="epochs in training Minor model process")
    parser.add_argument("--layerNumMinor", default= 4, type=int, help="layer numbers in Minor model")
    parser.add_argument("--hiddenDimMinor", default= "1024, 512, 256, 128", type=str, help="hidden dims in Minor model")
    parser.add_argument("--dropoutMinor", default= "0, 0.3, 0.2, 0.1", type=str, help="dropouts in Minor model")
    parser.add_argument("--learningRateMinor", default= 1e-4, type=float, help="learning rate")
    parser.add_argument("--epsMinor", default= 1e-7, type=float, help="eps")    
    
    parser.add_argument("--trainMode", default="gene_selection", type=str, help="trainMode") # gene_selection, variance_selection
    parser.add_argument("--majorGene-path", type=str, help="file path of differential expressed genes between major types") 
    parser.add_argument("--minorGene-path", type=str, help="file path of differential expressed genes between minor types")
    parser.add_argument("--majorNum", default= 6, type=int, help="number of major types")
    parser.add_argument("--majorVarianceThs", default = 0.6, type=float, help="threshold of getting genes for major types") 
    parser.add_argument("--minorVarianceThs", default = 0.8, type=float, help="threshold of getting genes for minor types") 
    parser.add_argument("--scalingMode", default="ss", type=str, help="scalingMode") # mms, ss
    parser.add_argument('--trueLabelUse', action='store_true', default=False, help='whether to use true label proportions in model training')
    parser.add_argument("--majorWeights", type=str, help="mae loss function weights in major predictions")
    parser.add_argument("--minorWeights", type=str, help="mae loss function weights in minor predictions")
    parser.add_argument("--lambdaP", default = 0.2, type=float, help="penalty coefficient in minor predictions with major info") 
    # info and file output
    parser.add_argument('--preprocessOut', action='store_true', default=False, help='whether to show boxplot and save as output')
    parser.add_argument('--realBulkData', action='store_true', default=False, help='whether used real bulk data for predict(without scores evaluation)')
    parser.add_argument('--boxplotOut', action='store_true', default=False, help='whether to show boxplot and save as output')
    parser.add_argument('--scatterplotOut', action='store_true', default=False, help='whether to show scatter plot and save as output')
    parser.add_argument("--scatterplotOutMajorHang", default= 2, type=int, help="num of rows in the scatter plot of major types")
    parser.add_argument("--scatterplotOutMajorLie", default= 3, type=int, help="num of columns in the scatter plot of major types") 
    parser.add_argument("--scatterplotOutMinorHang", default= 4, type=int, help="num of rows in the scatter plot of minor types")
    parser.add_argument("--scatterplotOutMinorLie", default= 4, type=int, help="num of columns in the scatter plot of minor types") 

    return parser

def main():
    #################################
    ## 参数获取
    #################################
    args = get_args_parser().parse_args()
    time_info = datetime.now().strftime("deconM1_%Y%m%d-%H-%M_")
    
    device = args.device
    reproducibility(seed=args.randomSeed)
    
    traindata = args.trainFile_path
    testdata = args.testFile_path
    result_dir = args.resultFile_path

    batch_size = args.batchSize
    
    epochs_major = args.epochsMajor
    num_layers_major = args.layerNumMajor
    hidden_dims_major = [int(item.strip()) for item in args.hiddenDimMajor.split(',')]
    dropout_rates_major = [float(item.strip()) for item in args.dropoutMajor.split(',')]
    lr_major = args.learningRateMajor
    eps_major = args.epsMajor
    
    epochs_minor = args.epochsMinor
    num_layers_minor = args.layerNumMinor
    hidden_dims_minor = [int(item.strip()) for item in args.hiddenDimMinor.split(',')]
    dropout_rates_minor = [float(item.strip()) for item in args.dropoutMinor.split(',')]
    lr_minor = args.learningRateMinor
    eps_minor = args.epsMinor
    
    mode = args.trainMode
    gene_file_major = args.majorGene_path
    gene_file_minor = args.minorGene_path
    majorN = args.majorNum
    variance_t1 = args.majorVarianceThs
    variance_t2 = args.minorVarianceThs
    scaler = args.scalingMode
    use_true_labels = True if args.trueLabelUse else False
    loss_weights1 = [float(item.strip()) for item in args.majorWeights.split(',')]
    loss_weights2 = [float(item.strip()) for item in args.minorWeights.split(',')]
    lambdaP = args.lambdaP
    
    preprocess_plot = True if args.preprocessOut else False
    
    if args.realBulkData:
        test_infos = testdata.split("/")[-1].split(".")[0].split("_")
        test_info = "".join(p.capitalize() for p in test_infos[1:-1])
    else:
        # test_info = testdata.split("/")[-1].split(".")[0].split("_")[-1] 
        test_info =  rename_from_filename(testdata)

    if mode == "gene_selection":
        mode_info = gene_file_minor.split("/")[-1].split(".")[0].split("_")[-1]
    else:
        mode_info = str(variance_t1) + "-" + str(variance_t2)

    if all(x == 1.0 for x in loss_weights2):
        weight_info = "NoWeight"
    else:
        weight_info = "WithWeight"

    para_summary = rename_from_filename(traindata) + "_" + test_info + "_" + "GeneMode_" + \
        ("".join(p.capitalize() for p in mode.split("_"))) + "_" + mode_info + "_Epoch_"+ str(epochs_major) + "-" + str(epochs_minor) + "_Scaling_" + scaler + "_lambdaP_" + str(lambdaP) + \
            "_" + weight_info + "_Dropout_" + ("+".join(str(x) for x in dropout_rates_major)) + "_" + ("+".join(str(x) for x in dropout_rates_minor))
            
    #################################
    ### 数据读取
    #################################
    train_x_major, train_y_major, train_x_minor, train_y_minor,\
    test_x_major, test_y_major, test_x_minor, test_y_minor, \
    major_types, minor_types, major_genes, minor_genes, types_info, samplename \
    = ProcessInputData(train_data=traindata, test_data=testdata, majorN = majorN, 
            mode = mode, variance_t1 = variance_t1, variance_t2 = variance_t2, 
            gene_file_major = gene_file_major, gene_file_minor = gene_file_minor, 
            scaler=scaler, scaler_plot = preprocess_plot, width=6, height=6, save_file = (result_dir + time_info + para_summary + "_Preprocess.jpg"))
    major_types[4] = "Bcell"
    types_info_temp = {key: list(types_info[key]) for key in major_types}
    types_info = types_info_temp

    train_loader = DataLoader(CustomDataset(train_x_major, train_x_minor, train_y_major, train_y_minor), batch_size=batch_size, shuffle=True)
    print("*"*100)
    
    
    #################################
    ### log file 创建和信息记录
    #################################
    log_file = open((result_dir + time_info + ".log"), 'a', encoding='utf-8')

    for key, value in vars(args).items():
        log_file.write(f"{key}: {value}\n")
    log_file.write("---"*30 + "\n")
    log_file.write(str(major_types) + "\n")
    log_file.write(str(minor_types) + "\n")
    log_file.write(str(types_info) + "\n")
    log_file.write(str(len(major_genes)) + "\n")
    log_file.write(str(len(minor_genes)) + "\n")
    log_file.write("---"*30 + "\n")
    log_file.write('training major data shape is ' + str(train_x_major.shape) + "\n")
    log_file.write('test major data shape is ' +  str(test_x_major.shape) + "\n") 
    log_file.write('training minor data shape is ' +  str(train_x_minor.shape) + "\n")
    log_file.write('test minor data shape is ' +  str(test_x_minor.shape) + "\n")
    log_file.write("---"*30 + "\n")
    log_file.write('training major proportions shape is ' +  str(train_y_major.shape) + "\n")
    log_file.write('test major proportions shape is ' +  str(test_y_major.shape) + "\n")
    log_file.write('training minor proportions shape is ' +  str(train_y_minor.shape) + "\n")
    log_file.write('test minor proportions shape is ' +  str(test_y_minor.shape) + "\n" )
    log_file.close()
    
    ################################################
    ### train
    ################################################
    loss_weights1 = torch.tensor(loss_weights1).to(device)
    loss_weights2 = torch.tensor(loss_weights2).to(device)

    #### stage1
    print('Stepping into the stage1 of the process')

    # 模型构建和训练
    model_major = MLPmodel(input_dim=train_x_major.shape[1], output_dim=train_y_major.shape[1], 
            layer_num=num_layers_major, hidden_dims=hidden_dims_major, dropout_rates=dropout_rates_major)
    optimizer_major = torch.optim.Adam(model_major.parameters(), lr=lr_major, eps=eps_major)

    model_major, loss1 = model_train(model_major, train_loader, loss_major, loss_weights1, optimizer_major, device, epochs_major)


    #### stage2 
    print('Stepping into the stage2 of the process')
    # 冻结model_major的参数
    model_major.eval()
    for p in model_major.parameters():
        p.requires_grad_(False)

     # 模型构建和训练
    model_minor = MLPmodel(input_dim=train_x_minor.shape[1], output_dim=train_y_minor.shape[1], 
            layer_num=num_layers_minor, hidden_dims=hidden_dims_minor, dropout_rates=dropout_rates_minor)
    optimizer_minor = torch.optim.Adam(model_minor.parameters(), lr=lr_minor, eps=eps_minor)

    model_minor, loss2= model_train(model_minor, train_loader, loss_minor_mae, loss_weights2, optimizer_minor, device, epochs_minor,
                        model_major=model_major, use_true_labels=use_true_labels, types_info=types_info, lambdaP=lambdaP)

    ################################################
    #### predict and save
    ################################################
    print('Stepping into the prediction')

    # 1) predict minor first
    pred_minor = model_predict(model_minor, test_x_minor, device)
    pred_minor_df = pd.DataFrame(pred_minor, columns=minor_types, index=samplename)
    pred_minor_df.to_csv((result_dir + time_info + para_summary + "_minor_PredProps.csv"), index=True)
    
    # 2) aggregate minor -> major (guarantee consistency)
    pred_major_from_minor_df = pd.DataFrame(index=samplename, columns=major_types, dtype=float)
    for major, minors in types_info.items():
        # 对于每个大类，累加其下所有小类的占比
        if len(minors) > 0:
            pred_major_from_minor_df[major] = pred_minor_df[minors].sum(axis=1)
        else:
            # 如果大类没有对应的小类，则默认比例为0
            pred_major_from_minor_df[major] = 0
    pred_major_from_minor_df.to_csv(result_dir + time_info + para_summary + "_major_fromMinor_PredProps.csv", index=True)

    # 3) also compute major directly for debugging
    pred_major_direct = model_predict(model_major, test_x_major, device)
    pred_major_direct_df = pd.DataFrame(pred_major_direct, columns=major_types, index=samplename)
    pred_major_direct_df.to_csv((result_dir + time_info + para_summary + "_major_direct_PredProps.csv"), index=True)
    

    ################################################
    #### evaluating results for simualtion test data
    ################################################
    if not args.realBulkData:
        print('evaluating for simulation test')
        
        #### major types， using aggregated one
        pred_major = pred_major_from_minor_df.values

        true_major_df = pd.DataFrame(test_y_major, columns=major_types)
        MAEscores = MAEscore(pred_major, test_y_major, mode="sep")
        CCCscores = CCCscore(pred_major, test_y_major, "sep")   
        pearsons = []
        for col in pred_major_from_minor_df.columns:
            pearsons.append(pearsonr(pred_major_from_minor_df[col], true_major_df[col])[0])
        # final evaluating scores save
        scores = {'Labels': major_types,'MAEs':MAEscores, 'CCCs': CCCscores,'Ps': pearsons}
        scores_df = pd.DataFrame(scores)
        scores_df.to_csv((result_dir + time_info + para_summary + "_FinalScores_majorFromMinor.csv"), index=False)
        # plotting
        if args.boxplotOut:
            boxplot_h(CCCscores, major_types, "", para_summary, 15, 2, (result_dir+time_info+para_summary+"_boxplot_majorFromMinor.jpg"))
        if args.scatterplotOut:
            scatter_pred_true(pred_major_from_minor_df, true_major_df, para_summary, args.scatterplotOutMajorHang, args.scatterplotOutMajorLie, 6, 5, (result_dir+time_info+para_summary+"_scatters_majorFromMinor.jpg"))

        #### minor types
        true_minor_df = pd.DataFrame(test_y_minor, columns=minor_types)
        MAEscores = MAEscore(pred_minor, test_y_minor, mode="sep")
        CCCscores = CCCscore(pred_minor, test_y_minor, "sep")
        pearsons = []
        for col in pred_minor_df.columns:
            pearsons.append(pearsonr(pred_minor_df[col], true_minor_df[col])[0])
        # final evaluating scores save
        scores = {'Labels': minor_types,'MAEs':MAEscores, 'CCCs': CCCscores,'Ps': pearsons}
        scores_df = pd.DataFrame(scores)
        scores_df.to_csv((result_dir + time_info + para_summary + "_FinalScores_minor.csv"), index=False)
        # plotting
        if args.boxplotOut:
            boxplot_h(CCCscores, minor_types, "", para_summary, 15, 2, (result_dir+time_info+para_summary+"_boxplot_minor.jpg"))
        if args.scatterplotOut:
            scatter_pred_true(pred_minor_df, true_minor_df, para_summary, args.scatterplotOutMinorHang, args.scatterplotOutMinorLie, 12, 12, (result_dir+time_info+para_summary+"_scatters_minor.jpg"))

if __name__ == "__main__":
    main()

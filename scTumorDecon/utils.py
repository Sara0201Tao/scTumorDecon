# -*- coding:utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
import torch
import anndata
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from glob import glob
from adjustText import adjust_text
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings

def ProcessInputData(train_data, test_data, sep="\t", majorN = 6, 
                     mode = "variance_selection", # variance_selection or gene_selection
                     variance_t1=0.60, variance_t2=0.90, 
                     gene_file_major = None, gene_file_minor = None, 
                     scaler="mms", scaler_plot = True, width=6, height=6, save_file=None):
    #################################
    ### read train data
    #################################
    print('Reading training data')
    if type(train_data) is str:
        if ".h5ad" in train_data:
            train_data = anndata.read_h5ad(train_data)
            train_x = pd.DataFrame(train_data.X, columns=train_data.var.index)
            train_y_major = train_data.obs.iloc[:,:majorN]
            train_y_minor = train_data.obs.iloc[:,majorN:]
            types_info = train_data.uns
    print('Reading is done')

    #################################
    ### read test data
    #################################
    print('Reading test data')
    if type(test_data) is str:
        if ".txt" in test_data:
            test_x = pd.read_csv(test_data, index_col=0, header = 0, sep=sep).T
            test_y_major = pd.DataFrame(columns=train_y_major.columns)
            test_y_minor = pd.DataFrame(columns=train_y_minor.columns)
        elif ".h5ad" in test_data:
            test_data = anndata.read_h5ad(test_data)
            test_x = pd.DataFrame(test_data.X, columns=test_data.var.index)
            test_y_major = test_data.obs.iloc[:,:majorN]
            test_y_minor = test_data.obs.iloc[:,majorN:]
    print('Reading test data is done')
    print('*'*50)

    #################################
    ### choose final featured genes
    #################################
    if mode == "variance_selection": # same as orginal paper
        print('Cutting variance...')
        var_cutoff_train = train_x.var(axis=0).sort_values(ascending=False)[int(train_x.shape[1] * variance_t1)]
        train_x_major = train_x.loc[:, train_x.var(axis=0) > var_cutoff_train]
        var_cutoff_train = train_x.var(axis=0).sort_values(ascending=False)[int(train_x.shape[1] * variance_t2)]
        train_x_minor = train_x.loc[:, train_x.var(axis=0) > var_cutoff_train]
        
        var_cutoff_test = test_x.var(axis=0).sort_values(ascending=False)[int(test_x.shape[1] * variance_t1)]
        test_x_major = test_x.loc[:, test_x.var(axis=0) > var_cutoff_test]
        var_cutoff_test = test_x.var(axis=0).sort_values(ascending=False)[int(test_x.shape[1] * variance_t2)]
        test_x_minor = test_x.loc[:, test_x.var(axis=0) > var_cutoff_test]
        
        print('Finding intersected genes...')
        inter = train_x_major.columns.intersection(test_x_major.columns)
        train_x_major = train_x_major[inter]
        test_x_major = test_x_major[inter]
        print('Intersected gene number in majors is ', len(inter))
        inter = train_x_minor.columns.intersection(test_x_minor.columns)
        train_x_minor = train_x_minor[inter]
        test_x_minor = test_x_minor[inter]
        print('Intersected gene number in minors is ', len(inter))
        print('*' * 50)
        
    elif mode == "gene_selection" and gene_file_major is not None and gene_file_minor is not None: # using differential genes calculated before
        print("using differential genes calculated before")
        genes_major = load_strings_from_file(gene_file_major)
        genes_minor = load_strings_from_file(gene_file_minor)
        genes_minor = list(set(genes_major).union(set(genes_minor)))
        print("Original genes_major number is ", len(genes_major))
        print("Original genes_minor number is ", len(genes_minor))


        inter_train = train_x.columns.intersection(genes_major)
        inter_test = test_x.columns.intersection(genes_major)
        inter = inter_train.intersection(inter_test)
        train_x_major = train_x[inter]
        test_x_major = test_x[inter]
        print('Intersected gene number in majors is ', len(inter))
        inter_train = train_x.columns.intersection(genes_minor)
        inter_test = test_x.columns.intersection(genes_minor)
        inter = inter_train.intersection(inter_test)
        train_x_minor = train_x[inter]
        test_x_minor = test_x[inter]
        print('Intersected gene number in minors is ', len(inter))
        print('*' * 50)
                
    ### getting info for return
    majors = list(train_y_major.columns)
    minors = list(train_y_minor.columns)
    major_genes = list(train_x_major.columns)
    minor_genes = list(train_x_minor.columns)
    samplename = test_x.index

    #################################
    ### data scaling process
    #################################
    print('Scaling...')
    train_x_major = np.log(train_x_major + 1)
    test_x_major = np.log(test_x_major + 1)
    train_x_minor = np.log(train_x_minor + 1)
    test_x_minor = np.log(test_x_minor + 1)
    
    if scaler=='ss':
        print("Using standard scaler...")
        ss = StandardScaler()
        trans_train_x_major = ss.fit_transform(train_x_major.T).T
        trans_test_x_major = ss.fit_transform(test_x_major.T).T
        trans_train_x_minor = ss.fit_transform(train_x_minor.T).T
        trans_test_x_minor = ss.fit_transform(test_x_minor.T).T
    elif scaler == 'mms':
        print("Using minmax scaler...")
        mms = MinMaxScaler()
        trans_train_x_major = mms.fit_transform(train_x_major.T).T
        trans_test_x_major = mms.fit_transform(test_x_major.T).T
        trans_train_x_minor = mms.fit_transform(train_x_minor.T).T
        trans_test_x_minor = mms.fit_transform(test_x_minor.T).T
    
    if scaler_plot:
        colors = sns.color_palette('RdYlBu', 10)
        fig, axes = plt.subplots(2, 2, figsize=(width, height))
        # 第一个子图
        sns.histplot(data=np.mean(train_x_major, axis=0), kde=True, color=colors[3],edgecolor=None, ax=axes[0,0])
        sns.histplot(data=np.mean(test_x_major, axis=0), kde=True, color=colors[7],edgecolor=None, ax=axes[0,0])
        axes[0,0].legend(title='BeforeScaling of majors', labels=['Training Data', 'Test Data'])
        # 第二个子图
        sns.histplot(data=np.mean(trans_train_x_major, axis=0), kde=True, color=colors[3],edgecolor=None, ax=axes[0,1])
        sns.histplot(data=np.mean(trans_test_x_major, axis=0), kde=True, color=colors[7],edgecolor=None, ax=axes[0,1])
        axes[0,1].legend(title='AfterScaling of majors', labels=['Training Data', 'Test Data'])
        # 第三个子图
        sns.histplot(data=np.mean(train_x_minor, axis=0), kde=True, color=colors[3],edgecolor=None, ax=axes[1,0])
        sns.histplot(data=np.mean(test_x_minor, axis=0), kde=True, color=colors[7],edgecolor=None, ax=axes[1,0])
        axes[1,0].legend(title='BeforeScaling of minors', labels=['Training Data', 'Test Data'])
        # 第四个子图
        sns.histplot(data=np.mean(trans_train_x_minor, axis=0), kde=True, color=colors[3],edgecolor=None, ax=axes[1,1])
        sns.histplot(data=np.mean(trans_test_x_minor, axis=0), kde=True, color=colors[7],edgecolor=None, ax=axes[1,1])
        axes[1,1].legend(title='AfterScaling of minors', labels=['Training Data', 'Test Data'])

        plt.tight_layout()
        
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        
        plt.show()

    return trans_train_x_major, train_y_major.values,trans_train_x_minor, train_y_minor.values,\
        trans_test_x_major, test_y_major.values,trans_test_x_minor, test_y_minor.values,\
            majors, minors, major_genes, minor_genes, types_info, samplename

def reproducibility(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
def load_strings_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return [line.rstrip('\n') for line in file]
           
def around_sum1(data, round):
    data_rounded = np.round(data, round)
    # 检查总和是否为1，并进行归一化处理
    sum_data_rounded = np.sum(data_rounded)
    diff = 1.0 - sum_data_rounded
    # 确保不会因为调整而导致负数
    if not np.isclose(sum_data_rounded, 1.0):
        index_to_adjust = np.random.choice(np.arange(len(data_rounded)))
        # 调整值时确保不会变为负数
        if diff < 0 and data_rounded[index_to_adjust] + diff < 0:
            # 如果调整会导致负数，则选择其他元素
            valid_indices = [i for i in range(len(data_rounded)) if data_rounded[i] + diff >= 0]
            if valid_indices:
                index_to_adjust = np.random.choice(valid_indices)
            else:
                raise ValueError("Cannot adjust any element without making it negative.")
        data_rounded[index_to_adjust] += diff

    return data_rounded

def MAEscore(y_pred, y_true, mode="all"):
    # mode有三种，分别是all(avg和其一致), sep
    # pred and label: shape{n sample, m cell}
    
    # Check if the shapes of pred and true are the same
    if y_pred.shape != y_true.shape:
        raise ValueError("pred and true must have the same shape.")
    
    if mode == 'all':
        return np.mean(np.abs(y_pred - y_true))
    
    elif mode == 'sep':
        mae_values = []
        for i in range(y_pred.shape[1]):
            mae = np.mean(np.abs(y_pred[:, i] - y_true[:, i]))
            mae_values.append(mae)
        return mae_values
    
def CCCscore(y_pred, y_true, mode='all'):
    # mode有三种，分别是all, avg, sep
    # pred and true: shape{n sample, m cell}
    
    # Check if the shapes of pred and true are the same
    if y_pred.shape != y_true.shape:
        raise ValueError("pred and true must have the same shape.")
    
    if mode == 'all':
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
    
    ccc_value = []
    for i in range(y_pred.shape[1]):
        r = np.corrcoef(y_pred[:, i], y_true[:, i])[0, 1]
        # Mean
        mean_true = np.mean(y_true[:, i])
        mean_pred = np.mean(y_pred[:, i])
        # Variance
        var_true = np.var(y_true[:, i])
        var_pred = np.var(y_pred[:, i])
        # Standard deviation
        sd_true = np.std(y_true[:, i])
        sd_pred = np.std(y_pred[:, i])
        # Calculate CCC
        numerator = 2 * r * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / denominator
        ccc_value.append(ccc)
    
    if mode == "all":
        return ccc_value[0]
    elif mode == "avg":
        return sum(ccc_value) / len(ccc_value)
    elif mode =="sep":
        return ccc_value
    
### 箱线图
def boxplot_h(data, labels, info, title, width, height, save_file):

    data_to_plot = [data]
    fig, ax = plt.subplots(figsize=(width, height)) # 设置图表大小
    bp = plt.boxplot(data_to_plot, vert=False) # vert=False 表示水平显示
    
    # 添加散点图来表示每个数据点
    scatter = plt.scatter(data, [1]*len(data), color='red', zorder=5)
    # 散点标注，自动调整文本位置以避免重叠
    texts = []
    for i, label in enumerate(labels):
        texts.append(plt.text(data[i], 1, label, ha='center', va='bottom'))
    adjust_text(texts, data, [1]*len(data), arrowprops=dict(arrowstyle="->", color='r', lw=1.5))
    # 添加自定义的多行文本信息，使用fig.transFigure变换
    plt.text(0.15, 0.60, info, fontsize=12, color='blue',
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(facecolor='gray', alpha=0.5),
             transform=fig.transFigure)

    plt.title(title)
    plt.xlabel('CCCscore values') # X轴标签
    plt.yticks([]) # 隐藏y轴刻度，因为我们只有一个箱子
    
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    
    plt.show() # 显示图表  
    
### 结果散点图
def scatter_pred_true(pred, true, whole_title, ax_w, ax_y, width, height, save_file):
    
    # 设置图形风格
    sns.set(style="whitegrid")
    # 创建一个3x3的子图网格
    fig, axes = plt.subplots(ax_w, ax_y, figsize=(width, height))
    # 展平axes数组以便于迭代
    axes = axes.flatten()  
    
    # 迭代每个列名和对应的子图
    for ax, column in zip(axes, pred.columns):
        x = pred[column]
        y = true[column]
        # 绘制散点图和拟合线
        sns.regplot(x=x, y=y, ax=ax, scatter_kws={"s": 5}, line_kws={"color": "black"}, ci=None)
        # 计算皮尔森相关系数
        corr, _ = pearsonr(x, y)
        ax.set_title(f'{column}\nCorrelation: {corr:.3f}', fontweight='bold')
        # 移除每个子图的坐标轴名称
        ax.set_xlabel('')
        ax.set_ylabel('')
        
    # 设置整张大图的横纵坐标名和名字
    fig.text(0.5, 0.04, 'proportions from prediction', ha='center', fontsize=8)  # X轴标签
    fig.text(0.04, 0.5, 'proportions from truth', va='center', rotation='vertical', fontsize=8)  # Y轴标签
    plt.suptitle(whole_title, fontsize=12, fontweight='bold')

    # 调整子图之间的距离避免标签重叠
    plt.tight_layout(rect=(0.05, 0.05, 1.0, 0.98))

    plt.savefig(save_file, dpi=300, bbox_inches='tight')

    # 显示图像
    plt.show()
    
### 单个数据集，不同参数的结果对比箱线图   
def boxplot_comparison(file_paths, column_to_plot, fill_color, dot_color, X_infos, title = "", xlabel="", ylabel="", width=10, height=6, save_file=None):
    Num = len(file_paths)
    if (len(column_to_plot) == 1) and (len(fill_color) == 1) and (len(dot_color) == 1):
        plt.figure(figsize=(width, height))
        column_to_plot = column_to_plot[0]
        # 读取所有文件并提取指定列的数据
        dt = []
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            dt.append(df[column_to_plot])  
        # 绘制箱线图和原始数据点
        box_positions = range(1, Num + 1)  # 箱线图的位置
        bp = plt.boxplot(dt, positions=box_positions, patch_artist=True)
        # 设置箱体颜色
        for box in bp['boxes']:
            box.set_facecolor(fill_color[0])
        # 添加原始数据点
        for i, d in enumerate(dt):
            x = np.full(len(d), i + 1)  # 创建一个数组，其值等于当前箱线图的位置
            plt.scatter(x, d, alpha=0.8, color=dot_color[0], s=12)  # s控制点的大小
            
        plt.ylabel(ylabel, fontsize=16, fontweight='bold')
        plt.xlabel(xlabel, fontsize=16, fontweight='bold')
        plt.title(title, fontsize=20,fontweight='bold')
        plt.xticks(box_positions, X_infos, fontsize=12)  # 设置x轴标签
        plt.grid(True, linestyle='--', alpha=0.5)
        if save_file != None:
            plt.savefig(save_file) 
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(width, height))
        # 读取所有文件并提取指定列的数据
        data = {col: [] for col in column_to_plot}
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            for column in column_to_plot:
                data[column].append(df[column])
        # 对于每个需要绘制的列，添加一个boxplot
        positions = range(1, Num * (len(column_to_plot) + 1), len(column_to_plot) + 1)
        for idx, (column, dt) in enumerate(data.items()):
            bp = ax.boxplot(dt, positions=[p + idx for p in positions], patch_artist=True)
            for box in bp['boxes']:
                box.set_facecolor(fill_color[idx]) 
            # 直接添加原始数据点
            for i, d in enumerate(dt):
                x = np.full(len(d), positions[i] + idx)  # 创建与数据长度相同的x坐标数组，值为当前位置
                ax.scatter(x, d, alpha=0.8, color=dot_color[idx], s=12)  # s控制点的大小  
        
        # 添加轴标签
        plt.ylabel(ylabel, fontsize=16, fontweight='bold')
        plt.xlabel(xlabel, fontsize=16, fontweight='bold')
        plt.title(title, fontsize=20,fontweight='bold')
        plt.xticks([p + 0.5 for p in positions], X_infos, fontsize=12)  # 设置x轴标签
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=fill_color[i], label=column_to_plot[i]) for i in range(len(column_to_plot))]
        ax.legend(handles=legend_elements, prop={'size': 15})
        ax.grid(True, linestyle='--', alpha=0.5)
        if save_file != None:
            plt.savefig(save_file, dpi=600, bbox_inches="tight") 
        plt.show()  

### 多组数据不同参数设置下，FinalScor结果箱线图
def plot_grouped_boxplot( df: pd.DataFrame, metric: str, dataset_type: str = "dataset", param_col: str = "param",
    order_datasets=None, order_params=None, box_width: float = 0.12, group_gap: float = 0.6,
    title: str = "", xlabel: str = "", ylabel: str = "",
    figsize=(10, 4), dpi=600, save_path = None):

    # 决定排序
    datasets = list(order_datasets) if order_datasets is not None else list(pd.unique(df[dataset_type]))
    params = list(order_params) if order_params is not None else list(pd.unique(df[param_col]))

    # 随机生成颜色（按param数量）
    cmap = plt.cm.hsv
    base_colors = [cmap(i / max(1, len(params))) for i in range(len(params))]
    rng = np.random.default_rng(42)
    rng.shuffle(base_colors)  # 保留“随机”，但差异明显
    color_map = {p: base_colors[i] for i, p in enumerate(params)}

    # 计算每个箱子的位置. 每个dataset是一个“组”，组内若干param箱子
    n_p = len(params)
    group_width = n_p * box_width + (n_p - 1) * 0.02
    centers = np.arange(len(datasets)) * (group_width + group_gap)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 逐个 param 画箱子
    for j, p in enumerate(params):
        positions = centers + (j - (n_p - 1) / 2) * (box_width + 0.02)

        # 组装每个dataset对应的一组数据（箱线图需要 list-of-arrays）
        data_list = []
        for d in datasets:
            vals = df.loc[(df[dataset_type] == d) & (df[param_col] == p), metric].dropna().values
            data_list.append(vals)
        # print(data_list)

        bp = ax.boxplot(data_list, positions=positions, widths=box_width, patch_artist=True, showfliers=False, manage_ticks=False)

        # 上色
        for patch in bp["boxes"]:
            patch.set_facecolor(color_map[p])
            patch.set_alpha(0.8)
        for elem in ["whiskers", "caps", "medians"]:
            for item in bp[elem]:
                item.set_color("black")
                item.set_linewidth(1.0)
        
        for i, vals in enumerate(data_list):
                x0 = positions[i]
                jitter = (np.random.rand(len(vals)) - 0.5) * (box_width * 0.25)
                # ax.scatter(x0 + jitter, vals, s=13, c=[color_map[p]], alpha=0.8, edgecolors="none",zorder=3)
                ax.scatter(x0 + jitter, vals, s=8, c="black", alpha=0.8, edgecolors="none",zorder=3)

    n_datasets = len(datasets)
    if n_datasets > 6:  # 如果数据集数量多，自动减小字体
        font_size = max(6, 8 - (n_datasets - 6) * 0.3)  # 动态计算字体大小
    else:
        font_size = 8

    ax.set_xticks(centers)
    ax.set_xticklabels(datasets, rotation=0, fontweight="bold", fontsize=font_size)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=8)    

    # 图例
    handles = [mpatches.Patch(color=color_map[p], label=p) for p in params]
    ax.legend(handles=handles, title=param_col, loc="center left", bbox_to_anchor=(1.02, 0.5),ncol=1, frameon=True,
              handlelength=0.8, handleheight=0.8,  # 控制图例句柄大小
              borderpad=0.4, labelspacing=0.4)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    # 给右侧图例留空间（小修但很关键）
    fig.tight_layout(rect=[0, 0, 0.75, 1])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    
    plt.show()

### 多组数据不同参数设置下，FinalScor结果的条形图
def plot_grouped_bar_with_error(df: pd.DataFrame, metric: str, dataset_type: str = "dataset", param_col: str = "param",
    order_datasets=None, order_params=None, bar_width: float = 0.10, group_gap: float = 0.5,
    title: str = "", xlabel: str = "", ylabel: str = "",
    figsize=(10, 4), dpi=150, save_path = None,
    agg: str = "median",   # "mean" or "median"
    err: str = "std",    # "std" or "sem"
    capsize: float = 3.0):

    # 决定排序
    datasets = list(order_datasets) if order_datasets is not None else list(pd.unique(df[dataset_type]))
    params = list(order_params) if order_params is not None else list(pd.unique(df[param_col]))

    # 随机生成差异大的颜色（按param数量）
    cmap = plt.cm.hsv
    base_colors = [cmap(i / max(1, len(params))) for i in range(len(params))]
    rng = np.random.default_rng(42)
    rng.shuffle(base_colors)  # 保留“随机”，但差异明显
    color_map = {p: base_colors[i] for i, p in enumerate(params)}

    # 计算每个箱子的位置. 每个dataset是一个“组”，组内若干param箱子
    n_p = len(params)
    group_width = n_p * bar_width + (n_p - 1) * 0.02
    centers = np.arange(len(datasets)) * (group_width + group_gap)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

   # 预聚合（每个 dataset × param -> center & error）
    g = df.groupby([dataset_type, param_col])[metric]

    if agg == "mean":
        center = g.mean()
    elif agg == "median":
        center = g.median()
    else:
        raise ValueError("agg must be 'mean' or 'median'")

    if err == "std":
        spread = g.std()
    elif err == "sem":
        spread = g.sem()
    else:
        raise ValueError("err must be 'std' or 'sem'")

    # 逐个 param 画 bar + errorbar
    for j, p in enumerate(params):
        x = centers + (j - (n_p - 1) / 2) * (bar_width + 0.02)

        y = []
        e = []
        for d in datasets:
            y.append(center.get((d, p), np.nan))
            e.append(spread.get((d, p), np.nan))

        ax.bar(x, y, width=bar_width, color=color_map[p], alpha=0.8, edgecolor="black", linewidth=0.8)
        ax.errorbar(x, y, yerr=e, fmt="none", ecolor="black", elinewidth=1.0, capsize=capsize)

    # 坐标轴/标题
    ax.set_xticks(centers)
    ax.set_xticklabels(datasets, rotation=0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=12)    

    # 图例
    handles = [mpatches.Patch(color=color_map[p], label=p) for p in params]
    ax.legend(handles=handles, title=param_col, loc="center left", bbox_to_anchor=(1.02, 0.5),ncol=1, frameon=True)

    # 给右侧图例留空间（小修但很关键）
    fig.tight_layout(rect=[0, 0, 0.82, 1])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    
    plt.show()

#### 为支持上述两种绘图模式的文件读取，和df构建
def build_df_from_dir(dir_path, pattern="*.csv", sep="_", filestrs=(3, 9), col_names=("dataset", "param"), save_path=None):
    file_list = sorted(glob(os.path.join(dir_path, pattern)))
    
    all_df = []
    for file in file_list:
        print(file)
        df = pd.read_csv(file)

        filename = os.path.splitext(os.path.basename(file))[0]
        parts = filename.split(sep)

        # 根据 index 列表提取字段
        for idx, col in zip(filestrs, col_names):
            df[col] = parts[idx]

        all_df.append(df)

    final_df = pd.concat(all_df, ignore_index=True)

    # ===== 是否保存 =====
    if save_path is not None:
        final_df.to_csv(save_path, index=False)

    return final_df

### 不同样本分布的占比bar图 
def prop_barplot_h(props, labels, id_flag = True, title = "test"): # props输入即为原始的simulation生成的list[array]
    import matplotlib.pyplot as plt
    import numpy as np
    SampleN = len(props)

    fig, ax = plt.subplots(figsize=(8, 3))
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    
    # 绘制每个数据集的水平条形图
    props_data = [prop.tolist() for prop in props]
    for idx, props in enumerate(props_data):
        y = SampleN-1-idx
        cumulative_prop = [sum(props[:i]) for i in range(len(props))]
        for t_idx, p in enumerate(props):
            ax.barh(y, left=cumulative_prop[t_idx], width=p, height=0.4, color=colors[t_idx])

    ax.set_xlim(0, 1)
    if id_flag:
        ax.set_yticks(list(range(SampleN)))
        ax.set_yticklabels([str(i) for i in range(SampleN)][::-1])  # 因为 y 被翻转了    
    else:   
        ax.yaxis.set_visible(False)
        
    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(labels))]
    ax.legend(handles, labels, title="Types", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()



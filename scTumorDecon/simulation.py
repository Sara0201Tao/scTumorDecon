# -*- coding:utf-8 -*-

import random
import numpy as np
import pandas as pd
import anndata
from tqdm import tqdm
import warnings
import argparse

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

def generate_pure_props(samplesN, types, pure_type):
    # 创建一个全零的数据框
    simulation_df = pd.DataFrame(0, index=range(samplesN), columns=types)
    # 将指定类型的值设置为1.0（即100%）
    simulation_df[pure_type] = 1.0
    # 将DataFrame转换为list，每个元素为一个numpy数组
    simulation_vector_list = [row.to_numpy() for _, row in simulation_df.iterrows()]

    return simulation_vector_list

def generate_even_props(samplesN, typesN, variance_pert, round):
    simulation_vector_list = []
    for _ in range(samplesN):
        # 生成正态分布随机数，取绝对值并四舍五入到3位小数
        m = np.abs(np.random.normal(loc=1/typesN, scale=variance_pert, size=typesN))
        m_normalized = m / m.sum()
        m_normalized = around_sum1(m_normalized, round)
        simulation_vector_list.append(m_normalized)

    return simulation_vector_list

def generate_random_props(samplesN, typesN, round):
    simulation_vector_list = []
    for _ in range(samplesN):
        # 生成均匀分布随机数，取绝对值并四舍五入到3位小数
        m = np.random.uniform(low=0.0, high=1.0, size=typesN)
        m_normalized = m / m.sum()
        m_normalized = around_sum1(m_normalized, round)
        simulation_vector_list.append(m_normalized)

    return simulation_vector_list

def generate_mirror_N_props(samplesN, bacis_props, variance_pert, round, permute_label=False):
    simulation_vector_list = []
    for _ in range(samplesN):
        # 生成先验类似+Norm扰动的数据，取绝对值并四舍五入到3位小数
        m = np.array([abs(np.random.normal(loc=y, scale=variance_pert)) for y in bacis_props])
        m_normalized = m / m.sum()
        m_normalized = around_sum1(m_normalized, round)

        # 可选：随机置换标签（打乱顺序）
        if permute_label:
            perm = np.random.permutation(len(bacis_props))
            m_normalized = m_normalized[perm]

        simulation_vector_list.append(m_normalized)
        
    return simulation_vector_list

def generate_mirror_D_props(samplesN, bacis_props, variance_factor, round, permute_label=False):
        props = [x * variance_factor for x in bacis_props]
        simulation_vector_list = []
        for _ in range(samplesN):
            # 生成先验类似的D分布数据，取绝对值并四舍五入到3位小数
            m = np.array(np.random.dirichlet(props, size=1)[0])
            m_normalized = around_sum1(m, round)

            # 可选：随机置换标签（打乱顺序）
            if permute_label:
                perm = np.random.permutation(len(bacis_props))
                m_normalized = m_normalized[perm]
            
            simulation_vector_list.append(m_normalized)
            
        return simulation_vector_list

        
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Data simualtion from scRNA-seq countTable", add_help=add_help)

    parser.add_argument("--scRNA-path", type=str, help="scRNA file path")
    parser.add_argument("--cellType-path", type=str, help="cell type info path")
    parser.add_argument("--outfile-path", type=str, help="output file path")
    parser.add_argument("--majorName", type=str, help="column name of major types in cell type info path")
    parser.add_argument("--minorName", type=str, help="column name of minor types in cell type info path")
    parser.add_argument("--desiredMajors", type=str, help="desired order of major types (comma-separated)")
    parser.add_argument("--digitalRound", default= 3, type=int, help="data precision")
    parser.add_argument("--randomSeed", default= 42, type=int, help="random seed setting for reproducibility")
    parser.add_argument("--cellNum", default= 2000, type=int, help="cell nums in each sample simulation")
    parser.add_argument("--sampleNum", default= 5000, type=int, help="sample nums in simulation")
    parser.add_argument('--priorSet', action='store_true', default=False, help='whether to use prior proportions in simulation')
    parser.add_argument("--ratioSplit", type=str, help="比例分割 (comma-separated values), total 7 ratios")
    parser.add_argument("--pureType", default=r"MaligantEpi", type=str, help="possible pure type in simulated sample")
    parser.add_argument("--vanEven", default= 0.02, type=float, help="variance in even proportion generation")
    parser.add_argument("--vanNorm", default= 0.01, type=float, help="variance in normal proportion generation")
    parser.add_argument("--vanFactor", default= 50, type=int, help="variance in dirichlet proportion generation")
    parser.add_argument('--spareSet', action='store_true', default=False, help='whether to generate spare samples in simulation')
    parser.add_argument("--spareRatio", default= 0.1, type=float, help="sample ratio in geneartion of spare samples")
    parser.add_argument("--spareTypeNum", default= 1, type=int, help="num of types in spare samples")
    parser.add_argument("--add_noise", action="store_true", default=False, help="Whether to add Gaussian noise to simulated bulk expression (DeepDecon Eq.6).")
    parser.add_argument("--gene_noise_alpha",type=float,default=0.02,help="Noise level alpha. Variance is alpha * X_ij,like 0.01, 0.05, 0.1, must be >= 0")
    parser.add_argument("--mask_genes",action="store_true",default=False,help="Whether to randomly mask gene expression values to 0 in each bulk sample.")
    parser.add_argument("--gene_mask_ratio",type=float,default=0.05,help="Ratio of genes to mask to 0 for each bulk sample (e.g., 0.10 for 10%),must be between 0 and 1")
    
    return parser

def main():
    #################################
    ## 参数获取
    #################################
    args = get_args_parser().parse_args()

    sc_data_file = args.scRNA_path
    celltype_file = args.cellType_path
    outfile = args.outfile_path
    selected_types_major = args.majorName
    selected_types_minor = args.minorName
    DigitalRound = args.digitalRound
    random_seed = args.randomSeed
    cellnum = args.cellNum
    samplenum = args.sampleNum
    method_split = [float(x.strip()) for x in args.ratioSplit.split(',')]
    pure_type = args.pureType
    variance_pert_even = args.vanEven
    variance_pert_normal = args.vanNorm
    variance_factor = args.vanFactor
    sparse_sample = args.spareRatio
    sparse_typeN = args.spareTypeNum
    do_add_noise = args.add_noise
    do_mask = args.mask_genes
    gene_noise_alpha = args.gene_noise_alpha
    gene_mask_ratio = args.gene_mask_ratio
    
    #################################
    ## set seed for reproducibility
    #################################
    print('You specified a random state, which will improve the reproducibility.')
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    #################################
    ### read input scRNA matrix data
    #################################
    # sc_data should be a cell*gene matrix, no null value, txt file, sep='\t'
    # index should be cell names and columns should be gene labels
    print('Reading single-cell dataset, this may take several mins')
    if '.txt' in sc_data_file:  # whole6w, ~5-6min
        total_lines = sum(1 for line in open(sc_data_file)) - 1  # 减去header行
        dropped_sc_list = []; total_dropped_sc = 0   # 记录每个 chunk 的丢弃情况

        # 使用分块读取，并在每一块读取后更新进度条
        chunk_size = 1000; chunks = []
        with tqdm(total=total_lines, desc="Processing", unit="lines") as pbar:
            for chunk in pd.read_csv(sc_data_file, index_col=0, header=0, sep="\t", chunksize=chunk_size):
                # 统计并丢弃 NA
                na_rows = chunk.isna().any(axis=1).sum()
                if na_rows > 0: # 记录：每个 chunk 丢弃行数
                    dropped_sc_list.append(int(na_rows)); total_dropped_sc += int(na_rows)

                chunk.dropna(inplace=True)
                chunk.index = range(len(chunk))
                chunks.append(chunk)
                pbar.update(len(chunk))
        sc_data = pd.concat(chunks, axis=0)
        sc_data.index = range(len(sc_data))
        
    if '.txt' in celltype_file:        
        celltype_info = pd.read_csv(celltype_file, index_col=0, header=0, sep="\t")
        total_dropped_ct = int(celltype_info.isna().any(axis=1).sum())

        celltype_info.dropna(inplace=True)
        celltype_info.index = range(len(celltype_info))
    
    # ------- warning / error check -------
    if total_dropped_sc > 0:
        warnings.warn(
            f"[scRNA] dropna removed {total_dropped_sc} rows in total; per-chunk drops (first 10): {dropped_sc_list[:10]}",
            RuntimeWarning
        )

    if total_dropped_ct > 0:
        warnings.warn(
            f"[celltype] dropna removed {total_dropped_ct} rows in total.",
            RuntimeWarning
        )

    # 严格错误检查：需要明确 sc_data 和 celltype_info 否则对齐会出错
    if (len(sc_data) != len(celltype_info)) or (total_dropped_sc != total_dropped_ct):
        msg = (
            f"Mismatch after dropna:\n"
            f"  sc_data rows      = {len(sc_data)}\n"
            f"  celltype rows     = {len(celltype_info)}\n"
            f"  scRNA dropped     = {total_dropped_sc}\n"
            f"  celltype dropped  = {total_dropped_ct}\n"
            f"Possible reason: NA rows are not aligned between scRNA and celltype files."
        )
        raise ValueError(msg)
   
    print('Input files reading done')    
        
    #################################
    ### info getting
    #################################       
    # 获取所有的大类，以及占比
    major_types = [item.strip() for item in args.desiredMajors.split(',')] if args.desiredMajors else sorted(celltype_info[selected_types_major].unique())
    print(major_types)

    major_counts = celltype_info[selected_types_major].value_counts()
    major_counts = major_counts.loc[major_types]
    total_count = major_counts.sum()
    major_proportions = around_sum1([count / total_count for count in major_counts], DigitalRound)
    print(major_proportions)

    # 大类index获取
    cellindex_major = celltype_info.groupby(selected_types_major).groups
    cellindex_major_temp = {key: cellindex_major[key] for key in major_types}
    for key, value in cellindex_major_temp.items():
        cellindex_major_temp[key] = np.array(value)
    cellindex_major = cellindex_major_temp

    # 按照顺序统计minor类型
    minor_types = []
    for major in major_types:
        temp_df = celltype_info[celltype_info[selected_types_major] == major]
        minor_counts_temp = temp_df[selected_types_minor].value_counts() # 从高到低排序了
        minor_types_temp = minor_counts_temp.index.tolist() # 读取类型
        minor_types += minor_types_temp
    print(minor_types)
    
    minor_counts = celltype_info[selected_types_minor].value_counts()
    minor_counts = minor_counts.loc[minor_types]
    total_count = minor_counts.sum()
    minor_proportions = around_sum1([count / total_count for count in minor_counts], DigitalRound)
    print(minor_proportions)

    # 小类index获取
    cellindex_minor = celltype_info.groupby(selected_types_minor).groups
    cellindex_minor_temp = {key: cellindex_minor[key] for key in minor_types}
    for key, value in cellindex_minor_temp.items():
        cellindex_minor_temp[key] = np.array(value)
    cellindex_minor = cellindex_minor_temp
    
    # 存储大类和小类间的关系
    types_info = {}
    for major in major_types:
        temp_df = celltype_info[celltype_info[selected_types_major] == major]
        minor_counts_temp = temp_df[selected_types_minor].value_counts() # 从高到低排序了
        minor_types_temp = minor_counts_temp.index.tolist() # 读取类型
        types_info[major] = minor_types_temp
    print(types_info)
        
    # 基因名读取
    genename = sc_data.columns

    # 将数据转换为连续内存的numpy数组以加速计算
    sc_data = np.ascontiguousarray(sc_data.values, dtype=np.float32)
    
    
    #################################
    ### cell proportions generation
    ################################# 
    print('Cell proportions generation starting')
    method_samplesN = [int(item * samplenum) for item in method_split]
    method_samplesN[-1] += samplenum - sum(method_samplesN)
    print(method_samplesN)

    if args.priorSet: 
        pure_props = generate_pure_props(method_samplesN[0], minor_types, pure_type)
        even_props = generate_even_props(method_samplesN[1], len(minor_types), variance_pert_even, DigitalRound)
        random_props = generate_random_props(method_samplesN[2], len(minor_types), DigitalRound)
        mirrorN_props = generate_mirror_N_props(method_samplesN[3], minor_proportions, variance_pert_normal, DigitalRound, permute_label= False)
        mirrorN_props_s = generate_mirror_N_props(method_samplesN[4], minor_proportions, variance_pert_normal, DigitalRound, permute_label= True)
        mirrorD_props = generate_mirror_D_props(method_samplesN[5], minor_proportions, variance_factor, DigitalRound, permute_label= False)
        mirrorD_props_s = generate_mirror_D_props(method_samplesN[6], minor_proportions, variance_factor, DigitalRound, permute_label= True)

        # 合并所有列表，并做数据转换
        all_props = pure_props + even_props + random_props + mirrorN_props + mirrorN_props_s + mirrorD_props + mirrorD_props_s
        all_props = np.array(all_props)

    else: # 没有先验分布时, [0.05, 0.15, 0.8] pure, even, random
        pure_props = generate_pure_props(method_samplesN[0], minor_types, pure_type)
        even_props = generate_even_props(method_samplesN[1], len(minor_types), variance_pert_even, DigitalRound)
        random_props = generate_random_props(method_samplesN[2], len(minor_types), DigitalRound)
        # mirrorD_props = generate_mirror_D_props(method_samplesN[2], [1]*len(minor_types), variance_factor=1, round=DigitalRound) # 后面发现不应该是这个mirror_D分布
        
        # 合并所有列表，并做数据转换
        all_props = pure_props + even_props + random_props
        all_props = np.array(all_props)    
    
    ###### adding sparse samples
    if args.spareSet:
        print("You set sparse as True, some cell's fraction will be zero, the samples ratio is " + str(sparse_sample) + ", the sparse_celltypeN is" + str(sparse_typeN))
        selecting_samples_sparse = list(range(method_samplesN[0], samplenum))
        samples_temp_sparse = random.sample(selecting_samples_sparse, int(all_props.shape[0] * sparse_sample))
        for i in samples_temp_sparse:
            indices = np.random.choice(np.arange(all_props.shape[1]), replace=False, size=sparse_typeN)
            all_props[i, indices] = 0
            if all_props[i].sum() > 0:
                all_props[i] = around_sum1(all_props[i] / all_props[i].sum(), DigitalRound)


    #################################
    ### cell sampling and save
    #################################
    # precise number for each celltype
    cell_num_minors = np.floor(cellnum * all_props + 0.1).astype(int)
    celltype_prop_minors = cell_num_minors / np.sum(cell_num_minors, axis=1).reshape(-1, 1)

    # ===== 1) precise number for each celltype =====
    print('Sampling cells to compose pseudo-bulk data')
    simulated_bulk = np.zeros((cell_num_minors.shape[0], sc_data.shape[1]))
    for i, cell_row_counts in tqdm(enumerate(cell_num_minors)):
        for j, cellname in enumerate(cellindex_minor.keys()):
            n_needed = cell_row_counts[j]
            if n_needed <= 0: continue

            available_total_cells = cellindex_minor[cellname]
            replace_flag = True if n_needed > len(available_total_cells) else False
            select_index = np.random.choice(available_total_cells, size=n_needed, replace=replace_flag)
            simulated_bulk[i] += sc_data[select_index].sum(axis=0)

    print('Sampling is done')
    
    # ===== 2) add Gaussian noise (adapted from DeepDecon method Eq.6) =====
    if do_add_noise:
        print('noise id adding to bulkRNAseq')
        print("gene_noise_alpha is "+ str(gene_noise_alpha))
        # std = sqrt(alpha * X); X >=0 assumed; clip to avoid tiny negatives due to numeric issues
        X = simulated_bulk
        std = np.sqrt(np.maximum(gene_noise_alpha * X, 0.0))
        noise = np.random.normal(loc=0.0, scale=std, size=X.shape)
        simulated_bulk = np.maximum(0.0, X + noise)

    # ===== 3) random mask genes to 0 per sample =====
    if do_mask:
        print('masking id performanced to bulkRNAseq')
        print("gene_mask_ratio in each sample is "+ str(gene_mask_ratio))
        n_genes = simulated_bulk.shape[1]
        mask_genes_n = int(round(gene_mask_ratio * n_genes))
        mask_genes_n = max(0, min(mask_genes_n, n_genes))

        if mask_genes_n > 0:
            for i in range(simulated_bulk.shape[0]):
                mask_idx = np.random.choice(n_genes, size=mask_genes_n, replace=False)
                simulated_bulk[i, mask_idx] = 0.0


    # final major and minor cell type proportions 
    celltype_prop_minors_df = pd.DataFrame(celltype_prop_minors, columns=minor_types)
    celltype_prop_majors_df = pd.DataFrame(index=celltype_prop_minors_df.index)
    for major, minors in types_info.items():
        # 对于每个大类，累加其下所有小类的占比
        if len(minors) > 0:
            celltype_prop_majors_df[major] = celltype_prop_minors_df[minors].sum(axis=1)
        else:
            # 如果大类没有对应的小类，则默认比例为0
            celltype_prop_majors_df[major] = 0
    celltype_prop_majors_df.rename(columns={'Bcell': 'BcellMain'}, inplace=True)
    final_props = celltype_prop_majors_df.join(celltype_prop_minors_df)
    
    final_props.index = final_props.index.astype(str)
    simudata = anndata.AnnData(X=simulated_bulk, obs=final_props, var=pd.DataFrame(index=genename), uns = types_info)
    simudata.write_h5ad(outfile)
    print(f"Success! Result saved to: {args.outfile_path}")

if __name__ == "__main__":
    main()

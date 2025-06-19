import os
import argparse
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    # Initialize the ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add the required arguments to the ArgumentParser object
    parser.add_argument('--path2', '-p2', type=str, required=True, help='The second omics file name.')
    parser.add_argument('--path3', '-p3', type=str, required=True, help='The third omics file name.')
    parser.add_argument('--path4', '-p4', type=str, required=True, help='The forth omics file name.')
    parser.add_argument('--path5', '-p5', type=str, required=True, help='The survival file name.')

    # Parse the arguments passed to the script
    args = parser.parse_args()


# Read in the data
miRNA = pd.read_table(args.path2, sep=',', index_col=0)
RNA = pd.read_table(args.path3, sep=',', index_col=0)
Meth = pd.read_table(args.path4, sep=',', index_col=0)
survive = pd.read_table(args.path5, sep=',', index_col=0)

# Import the necessary modules
from CACAE import Model2
from CACAE import Process
from CACAE.utils1 import ClusterProcessor, do_km_plot
from CACAE.Survive_select1 import survive_select

# Process the data
RNA_processor = Process.DataProcessor(RNA)
RNA = RNA_processor.sort_corr(5000)
RNA.to_csv('ACC_sorted_RNA.csv', index=False)

miRNA_processor = Process.DataProcessor(miRNA)
miRNA = miRNA_processor.sort_corr(100)
miRNA.to_csv('ACC_sorted_miRNA.csv', index=False)

Meth_processor = Process.DataProcessor(Meth)
Meth = Meth_processor.sort_corr(2000)
Meth.to_csv('ACC_sorted_Meth.csv', index=False)

# Build the models
RNA_model = Model2.ProgCAE(RNA.shape[1])
miRNA_model = Model2.ProgCAE(miRNA.shape[1])
Meth_model = Model2.ProgCAE(Meth.shape[1])

# Train the models
RNA_model.fit(RNA)
miRNA_model.fit(miRNA)
Meth_model.fit(Meth)

# Extract features from the models
RNA_feature = pd.DataFrame(RNA_model.extract_feature(RNA), columns=[f"ACC_RNA_feat_{i}" for i in range(RNA_model.extract_feature(RNA).shape[1])])
miRNA_feature = pd.DataFrame(miRNA_model.extract_feature(miRNA), columns=[f"ACC_miRNA_feat_{i}" for i in range(miRNA_model.extract_feature(miRNA).shape[1])])
Meth_feature = pd.DataFrame(Meth_model.extract_feature(Meth), columns=[f"ACC_Meth_feat_{i}" for i in range(Meth_model.extract_feature(Meth).shape[1])])

# 将特征组合成一个数据框
flatten = pd.concat([RNA_feature, miRNA_feature, Meth_feature], axis=1)
SURVIVE_SELECT = survive_select(survive, flatten, 0.05)

# Save the dataframe to a filez
SURVIVE_SELECT.to_csv('ACC_features.csv')

# Perform clustering and survival analysis
# Initialize the cluster processor
cp = ClusterProcessor(SURVIVE_SELECT, survive)
# Compute indexes for clustering using KMeans
cp.compute_indexes(3)
# Compute p-value and clusters for Log-Rank test using KMeans
p_value, clusters = cp.LogRankp(3)

# Plot survival curves using Kaplan-Meier method
do_km_plot(survive, pvalue=p_value.p_value, cindex = None, cancer_type='ACC', model_name='CAE')

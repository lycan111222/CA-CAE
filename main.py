import os

cancer_name = "LAML"


os.makedirs(f'sorted_data', exist_ok=True)
os.makedirs(f'features', exist_ok=True)


import pandas as pd
import warnings
warnings.filterwarnings("ignore")

path2 = f"example/{cancer_name}_miRNA.csv"
path3 = f"example/{cancer_name}_mRNA.csv"
path4 = f"example/{cancer_name}_METH.csv"
path5 = f"example/{cancer_name}_sur.csv"


miRNA = pd.read_table(path2, sep=',', index_col=0)
RNA = pd.read_table(path3, sep=',', index_col=0)
Meth = pd.read_table(path4, sep=',', index_col=0)
survive = pd.read_table(path5, sep=',', index_col=0)


from CACAE import Model, Process
from CACAE.utils import ClusterProcessor, do_km_plot
from CACAE.Survive_select import survive_select


RNA_processor = Process.DataProcessor(RNA)
RNA = RNA_processor.sort_corr(5000)
RNA.to_csv(os.path.join(f'sorted_data', f'{cancer_name}_sorted_mRNA.csv'), index=False)

miRNA_processor = Process.DataProcessor(miRNA)
miRNA = miRNA_processor.sort_corr(100)
miRNA.to_csv(os.path.join(f'sorted_data', f'{cancer_name}_sorted_miRNA.csv'), index=False)

Meth_processor = Process.DataProcessor(Meth)
Meth = Meth_processor.sort_corr(2000)
Meth.to_csv(os.path.join(f'sorted_data', f'{cancer_name}_sorted_Meth.csv'), index=False)



RNA_model = Model.CACAE(RNA.shape[1])
miRNA_model = Model.CACAE(miRNA.shape[1])
Meth_model = Model.CACAE(Meth.shape[1])
RNA_model.fit(RNA)
miRNA_model.fit(miRNA)
Meth_model.fit(Meth)



RNA_feature = pd.DataFrame(RNA_model.extract_feature(RNA), columns=[f"{cancer_name}_RNA_feat_{i}" for i in range(RNA_model.extract_feature(RNA).shape[1])])
miRNA_feature = pd.DataFrame(miRNA_model.extract_feature(miRNA), columns=[f"{cancer_name}_miRNA_feat_{i}" for i in range(miRNA_model.extract_feature(miRNA).shape[1])])
Meth_feature = pd.DataFrame(Meth_model.extract_feature(Meth), columns=[f"{cancer_name}_Meth_feat_{i}" for i in range(Meth_model.extract_feature(Meth).shape[1])])

flatten = pd.concat([RNA_feature, miRNA_feature, Meth_feature], axis=1)
SURVIVE_SELECT = survive_select(survive, flatten, 0.05)
SURVIVE_SELECT.to_csv(os.path.join(f'features', f'{cancer_name}_features.csv'))



cp = ClusterProcessor(SURVIVE_SELECT, survive)
cp.compute_indexes(2)
p_value, clusters = cp.LogRankp(2)

do_km_plot(survive, pvalue=p_value.p_value, cindex=None, cancer_type=cancer_name, model_name='CA-CAE')

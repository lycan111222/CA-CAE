def survive_select(survive_data, data, p_thresh, lasso_alpha=0.01):
    """
    使用Cox比例风险回归和Lasso回归筛选与生存数据相关的特征。

    参数：
    - survive_data (pandas DataFrame): 包含生存数据的DataFrame (包括OS和OS.time)。
    - data (pandas DataFrame): 需要评估的特征数据。
    - p_thresh (float): 用于筛选特征的p值阈值。
    - lasso_alpha (float): Lasso回归的正则化参数。

    返回：
    - pandas DataFrame: 包含与生存数据相关的特征。
    """
    from lifelines import CoxPHFitter
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler

    # 获取生存时间和事件状态
    time_col = survive_data['OS.time'].values
    event_col = survive_data['OS'].values

    # 标准化特征数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # 使用Lasso回归进行特征选择
    lasso = Lasso(alpha=lasso_alpha)
    lasso.fit(scaled_data, time_col)

    # 仅选择Lasso系数不为0的特征
    selected_features_indices = [i for i, coef in enumerate(lasso.coef_) if coef != 0]

    # 初始化存储筛选出来的特征列表
    selected_features = []

    # 使用Cox比例风险回归进行进一步筛选
    for i in selected_features_indices:
        # 创建Cox模型
        cpf = CoxPHFitter()
        # 将特征加入生存数据
        temp_df = survive_data.copy()
        temp_df['feature'] = data.iloc[:, i].values
        # 拟合Cox模型
        cpf.fit(temp_df, 'OS.time', 'OS')

        # 检查p值是否小于阈值
        if cpf.summary['p'].values[0] <= p_thresh:
            selected_features.append(i)

        # 返回经过筛选的特征，并保留样本名（索引）
    selected_data = data.iloc[:, selected_features]

    # 保留原始数据的索引（样本名）
    selected_data.index = data.index

    return selected_data
def param_types():
    """
    定义模型参数对应的数据类型（完整版本，去重处理）
    """
    return {
        # 通用参数
        'random_state': 'int',
        'n_jobs': 'int',

        # 随机森林 RandomForestRegressor / RandomForestClassifier
        'n_estimators': 'int',
        'criterion': 'select',
        'max_depth': 'int',
        'min_samples_split': 'int',
        'min_samples_leaf': 'int',
        'min_weight_fraction_leaf': 'float',
        'max_features': 'select',
        'max_leaf_nodes': 'int',
        'min_impurity_decrease': 'float',
        'bootstrap': 'bool',
        'oob_score': 'bool',
        'ccp_alpha': 'float',
        'max_samples': 'int',

        # 梯度提升树 GradientBoostingRegressor / GradientBoostingClassifier
        'learning_rate': 'float',
        'subsample': 'float',
        'validation_fraction': 'float',
        'n_iter_no_change': 'int',
        'tol': 'float',
        'init': 'select',  # estimator 或 None
        'warm_start': 'bool',

        # 线性回归 LinearRegression
        'fit_intercept': 'bool',
        'normalize': 'bool',  # 已弃用
        'copy_X': 'bool',
        'positive': 'bool',

        # Logistic回归 LogisticRegression
        'penalty': 'select',
        'dual': 'bool',
        'C': 'float',
        'intercept_scaling': 'float',
        'class_weight': 'select',
        'solver': 'select',
        'multi_class': 'select',
        'l1_ratio': 'float',  # elasticnet

        # 支持向量机 SVR / SVC
        'kernel': 'select',
        'degree': 'int',
        'gamma': 'select',
        'coef0': 'float',
        'shrinking': 'bool',
        'cache_size': 'int',
        'verbose': 'bool',
        'max_iter': 'int',
        'epsilon': 'float',  # SVR
        'probability': 'bool',  # SVC

        # 决策树 DecisionTreeRegressor / DecisionTreeClassifier
        'splitter': 'select',


        # KNN KNeighborsRegressor / KNeighborsClassifier
        'n_neighbors': 'int',
        'weights': 'select',
        'algorithm': 'select',
        'leaf_size': 'int',
        'p': 'int',
        'metric': 'select',
        'metric_params': 'dict',
    }


def param_options_map():
    """
    定义枚举类型参数的可选值（完整版本，去重处理）
    """
    return {
        # 随机森林 & 决策树
        'criterion': ['mse', 'friedman_mse', 'mae', 'poisson', 'gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2', None],

        # 决策树
        'splitter': ['best', 'random'],

        # Logistic回归
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
        'multi_class': ['auto', 'ovr', 'multinomial'],
        'class_weight': ['balanced', None],

        # 支持向量机
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'gamma': ['scale', 'auto'],

        # KNN
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev', 'precomputed'],

        # GradientBoosting init
        'init': [None],  # 可以是 estimator, 这里简单处理
    }

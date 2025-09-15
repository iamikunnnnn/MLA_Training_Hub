import pandas as pd
import numpy as np
from typing import List, Union
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    """
    数据预处理
    """

    def __init__(self, data: pd.DataFrame = None):
        """
        初始化数据预处理器
        :param: data
        """
        # 创建时直接初始胡data
        self.data = data.copy() if data is not None else None

    def get_data_null(self):
        return self.data.isna().sum()

    # 空值处理 - 3个常用方法

    def drop_null_columns(self,columns):
        self.data = self.data.drop(columns=columns)
        return self.data


    def drop_null_rows(self, columns: Union[str, List[str], None] = None):
        """
        删除包含空值的行
        :param:columns: 要检查的列，None表示检查所有列
        :return:self: 返回自身，支持链式调用
        """
        if self.data is None:
            raise ValueError("没有加载数据")

        if columns:
            # 如果columns的类型确实是str
            if isinstance(columns, str):
                columns = [columns]

            # 检查列是否存在
            missing_cols = [col for col in columns if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"列不存在: {missing_cols}")

            self.data = self.data.dropna(subset=columns)
        else:
            self.data = self.data.dropna()

        return self

    def fill_null_with_mean(self, columns: Union[str, List[str], None] = None) -> 'DataPreprocessor':
        """
        用均值填充空值（仅适用于数值列）
        :param:columns: 要处理的列，None表示处理所有数值列
        :return:self: 返回自身，支持链式调用
        """
        if self.data is None:
            raise ValueError("没有加载数据")

        if columns is None:
            # 只处理数值列
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                self.data[col].fillna(self.data[col].mean(), inplace=True)
        else:
            if isinstance(columns, str):
                columns = [columns]

            for col in columns:
                if col not in self.data.columns:
                    raise ValueError(f"列 '{col}' 不存在")
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    self.data[col].fillna(self.data[col].mean(), inplace=True)
                else:
                    print(f"警告: 列 '{col}' 不是数值类型，跳过处理")

        return self

    # 重复值处理 - 2个常用方法
    def remove_duplicates(self, columns: Union[str, List[str], None] = None) -> 'DataPreprocessor':
        """
        删除重复行
        Args:
            columns: 检查重复的列，None表示检查所有列
        Returns:
            self: 返回自身，支持链式调用
        """
        if self.data is None:
            raise ValueError("没有加载数据")
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            self.data = self.data.drop_duplicates(subset=columns)
        else:
            self.data = self.data.drop_duplicates()
        return self

    # 数据信息查看 - 2个常用方法
    def get_basic_info(self):
        """
        获取数据基本信息
        :return:dict: 包含数据基本信息
        """
        if self.data is None:
            return {"error": "没有加载数据"}

        return {
            "形状": self.data.shape,
            "列名": list(self.data.columns),
            "空值数量": dict(self.data.isna().sum()),
            "重复行数量": self.data.duplicated().sum()
        }

    def get_null_info(self, columns: Union[str, List[str], None] = None) -> dict:
        """
        获取空值详细信息
        :param:columns: 指定列，None表示所有列
        :return:dict: 空值信息
        """
        if self.data is None:
            return {"error": "没有加载数据"}

        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]

            # 检查列是否存在
            missing_cols = [col for col in columns if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"列不存在: {missing_cols}")

            data_subset = self.data[columns]
        else:
            data_subset = self.data

        null_counts = data_subset.isnull().sum()
        null_percentage = (null_counts / len(data_subset) * 100).round(2)

        return {
            "空值数量": dict(null_counts),
            "空值百分比": dict(null_percentage),
            "总空值数": null_counts.sum()
        }

    def get_data(self) -> pd.DataFrame:
        """
        获取当前数据
        :return:pd.DataFrame: 当前处理后的数据
        """
        if self.data is None:
            raise ValueError("没有加载数据")
        return self.data.copy()

    def save_data(self, filepath: str) -> None:
        """
        保存数据到CSV文件
        :param:filepath: 文件路径
        """
        if self.data is None:
            raise ValueError("没有数据可保存")

        self.data.to_csv(filepath, index=False)

    def get_dummy_data(self, columns) -> pd.DataFrame:
        """
        :param columns: 需要独热编码的列
        :return: 去除分类变量的列
        """
        if isinstance(self.data[columns], pd.Series):
            dummies = pd.get_dummies(self.data[columns])
            self.data = pd.concat([self.data.drop(columns=columns), dummies], axis=1)
        else:
            categorical_columns = self.data[columns].select_dtypes(include=['object']).columns
            columns_to_encode = [col for col in categorical_columns if len(self.data[col].unique()) <= 300]


            if not columns_to_encode:
                return self.data

            transfer = DictVectorizer(sparse=True)
            dummies = transfer.fit_transform(self.data[columns_to_encode].to_dict(orient='records'))

            # 转成稀疏 DataFrame
            dummies_df = pd.DataFrame.sparse.from_spmatrix(
                dummies,
                columns=transfer.get_feature_names_out(),
                index=self.data.index
            )

            # 转成 int
            dummies_df = dummies_df.astype(int)

            self.data = pd.concat([self.data.drop(columns=categorical_columns), dummies_df], axis=1)
            return self.data

    def Label_Encoding(self, columns):
        """
        将指定列进行标签编码（Label Encoding），将类别转换为整数
        :param columns: 需要编码的列
        :return: 编码后的 DataFrame
        """
        # 如果是series不需要那么操作
        if isinstance(self.data[columns], pd.Series):
            le = LabelEncoder()
            self.data[columns] = le.fit_transform(self.data[columns].astype(str))  # 转成 str 以防有 NaN
        else:

            # 先筛选出 object 类型的列
            categorical_columns = self.data[columns].select_dtypes(include=['object']).columns

            # 可选：限制类别数量，避免高基数编码
            columns_to_encode = [col for col in categorical_columns if len(self.data[col].unique()) <= 10]

            if not columns_to_encode:
                return self.data

            # 对每列进行 Label Encoding
            for col in columns_to_encode:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))  # 转成 str 以防有 NaN

        return self.data

    def Standard_Scaling(self, columns):
        """
        对指定列进行标准化（StandardScaler），将数据缩放到均值为0、方差为1的分布
        :param columns: 需要标准化的列，可以是单列名或列名列表
        :return: 标准化后的 DataFrame
        """
        scaler = StandardScaler()

        # 如果是单列（Series）
        if isinstance(self.data[columns], pd.Series):
            self.data[columns] = scaler.fit_transform(self.data[columns].values.reshape(-1, 1))
        else:
            # 先筛选出数值型的列
            numeric_columns = self.data[columns].select_dtypes(include=['int64', 'float64']).columns

            if not len(numeric_columns):
                return self.data

            # 批量标准化
            self.data[numeric_columns] = scaler.fit_transform(self.data[numeric_columns])

        return self.data

    def __repr__(self) -> str:
        """字符串表示"""
        if self.data is None:
            return "DataPreprocessor(无数据)"
        else:
            return f"DataPreprocessor(数据形状: {self.data.shape})"


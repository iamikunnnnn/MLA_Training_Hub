import time
import traceback

import streamlit as st
import numpy as np
from matplotlib import pyplot as plt

from Deep_Learning import reg_forward, markdown, cls_forward, visualization_nn

st.markdown("""
    <style>
    .reportview-container .main .block-container{
        padding-left: 2rem;  /* 左边距，默认更大 */
        padding-right: 2rem; /* 右边距，默认更大 */
    }
    </style>
""", unsafe_allow_html=True)


def show():
    page = st.sidebar.selectbox(
        "选择页面",
        ['理论学习', '回归拟合效果展示', '分类效果展示', '神经网络演示'],
    )

    if page == "理论学习":
        st.session_state["study_choose"] = st.selectbox("请选择你想要了解的知识", ["基本知识", "基本流程", "激活函数"])

        if st.session_state["study_choose"] == "基本知识":
            study_basic = markdown.basic()
            st.markdown(study_basic)
        elif st.session_state["study_choose"] == "基本流程":
            study_progress = markdown.progress()
            st.markdown(study_progress)
        elif st.session_state["study_choose"] == "激活函数":
            study_activation = markdown.activation()
            st.markdown(study_activation)

    elif page == "回归拟合效果展示":

        st.markdown("# 拟合效果展示")
        a_fn = st.selectbox("请选择激活函数", ["无", "sigmoid"])

        if st.button("开始拟合"):

            if a_fn == "sigmoid":
                data = np.array([
                    [0.8, 0],
                    [1.1, 0],
                    [1.7, 0],
                    [3.2, 1],
                    [3.7, 1],
                    [4.0, 1],
                    [4.2, 1]
                ])
                x_data = data[:, 0]
                y_data = data[:, 1]

                # 获取生成器（包含每一步的fig）
                fig_generator = reg_forward.forward(x_data, y_data, "sigmoid")
            elif a_fn == "无":
                data = np.array([
                    [-0.5, 7.7],
                    [1.8, 93.5],
                    [0.9, 57.8],
                    [0.4, 39.2],
                    [-1.4, -15.7],
                    [-1.4, -37.3],
                    [-1.8, -49.1],
                    [1.5, 75.6],
                    [0.4, 34],
                    [0.8, 62.3]
                ])
                x_data = data[:, 0]  # 特征值
                y_data = data[:, 1]  # 目标值
                # 获取生成器（包含每一步的fig）
                fig_generator = reg_forward.forward(x_data, y_data, "无")
            else:
                return
            # 创建占位符用于实时更新
            plot_placeholder = st.empty()
            # 迭代生成器，逐个显示每一步的图表
            for fig in fig_generator:
                plot_placeholder.pyplot(fig)  # 用当前fig更新占位符
                plt.close(fig)  # 释放资源
                time.sleep(0.01)  # 控制更新速度
    elif page == "分类效果展示":
        if st.button("开始分类"):
            # 定义第一类样本点
            class1_points = np.array([[1.9, 1.2],
                                      [1.5, 2.1],
                                      [1.9, 0.5],
                                      [1.5, 0.9],
                                      [0.9, 1.2],
                                      [1.1, 1.7],
                                      [1.4, 1.1]])

            # 定义第二类样本点
            class2_points = np.array([[3.2, 3.2],
                                      [3.7, 2.9],
                                      [3.2, 2.6],
                                      [1.7, 3.3],
                                      [3.4, 2.6],
                                      [4.1, 2.3],
                                      [3.0, 2.9]])

            fig_generator = cls_forward.forward(class1_points, class2_points)
            # 创建占位符用于实时更新
            plot_placeholder = st.empty()
            # 迭代生成器，逐个显示每一步的图表
            gen = fig_generator
            try:
                while True:
                    fig = next(gen)
                    plot_placeholder.pyplot(fig)
                    plt.close(fig)
                    time.sleep(0.01)
            except StopIteration as e:
                fig, acc, f1 = e.value
                print(fig, acc, f1)

            st.pyplot(fig, use_container_width=True)
            st.write("acc:", acc)
            st.write("f1:", f1)

    elif page == "神经网络演示":
        try:
            st.sidebar.header("神经网络参数")

            # 数据集选择
            dataset_type = st.sidebar.selectbox(
                "选择数据集",
                ["分类", "圆形", "月形", "自定义", "回归","回归2.0"]
            )

            # 选择是否加入衍生特征
            selected_derived_features = st.sidebar.multiselect("是否加入衍生特征",["平方项","交叉项","正弦项","余弦项"])

            n_features = 2  # 先初始化为2，因为分类任务的特征都是2，只有回归任务的特征数量可以调整，这里
            if dataset_type == "回归" or dataset_type == "回归2.0":
                n_features = st.sidebar.slider("特征数量", min_value=1, max_value=20, value=5)

            # base_n_features用于保存原始的特征数，用来下面计算加入衍生特征后的维度
            "这个原始特征数等会还要传入visualization的，很重要，因为生成数据要根据原始的n_features来"
            base_n_features = n_features
            if "平方项" in selected_derived_features:
                n_features+=base_n_features
            if "交叉项"  in selected_derived_features:
                n_features+= int(base_n_features*(base_n_features-1)/2)
            if "正弦项" in selected_derived_features:
                n_features+= base_n_features
            if "余弦项" in selected_derived_features:
                n_features+= base_n_features

            # 数据量
            n_samples = st.sidebar.slider(
                "数据量",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100
            )

            # 网络架构
            hidden_layers = st.sidebar.slider("隐藏层数", 1, 4, 2)
            layer_sizes = []

            if dataset_type == "回归" or dataset_type == "回归2.0":
                layer_sizes.append(n_features)  # 先加一个输入层
            else:
                layer_sizes.append(n_features)  # 先加一个输入层

            for i in range(hidden_layers):
                n = st.sidebar.slider(f"第{i + 1}层神经元数", 2, 8, 4, key=f"layer_{i}")
                layer_sizes.append(n)
            if dataset_type == "回归" or dataset_type == "回归2.0":
                layer_sizes.append(1)
            else:
                layer_sizes.append(2)  # 最后加上输出层

            # 训练参数
            learning_rate = st.sidebar.select_slider(
                "学习率",
                options=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                value=0.01
            )

            active_fn = st.sidebar.selectbox(
                "激活函数",
                ["relu", "tanh", "sigmoid"]
            )

            num_epochs = st.sidebar.selectbox("训练轮数", [100, 1000, 5000, 10000])
            batch_size = st.sidebar.selectbox("batch_size", [1, 20, 50])

            params = {
                "active_fn": active_fn,
                "n_samples": n_samples,
                "dataset_type": dataset_type,
                'layer_sizes': layer_sizes,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                "num_epochs": num_epochs,
                "n_features": n_features,
                "selected_derived_features":selected_derived_features,
                "base_n_features":base_n_features
            }

            if st.button("开始训练"):
                net = visualization_nn.TrainNet(**params)  # 用字典批量传入参数
                fig_generator = net.train()

                col1, col2 = st.columns([2, 2])
                # 第一个图表放入第一列
                with col1:
                    # 创建占位符用于实时更新
                    plot_placeholder_net = st.empty()  # 第一个图表的占位符
                # 第二个图表放入第二列
                with col2:
                    # 创建占位符用于实时更新
                    plot_placeholder_boundary = st.empty()  # 第二个图表的占位符


                write_epoch = st.empty()  # 创建占位符更新训练轮次
                write_loss = st.empty()  # 创建占位符更新损失
                write_weight = st.empty()  # 创建占位符更新权重

                # 迭代生成器，逐个显示每一步的图表
                for net_fig, boundary_fig,loss,weights,epoch in fig_generator:
                    plot_placeholder_net.pyplot(net_fig)  # 用当前fig更新占位符
                    plt.close(net_fig)  # 释放资源

                    plot_placeholder_boundary.pyplot(boundary_fig)  # 用当前fig更新占位符
                    plt.close(boundary_fig)  # 释放资源

                    time.sleep(0.01)  # 控制更新速度

                    write_epoch.write(epoch)
                    write_loss.write(loss)
                    write_weight.write(weights)


        except Exception as e:  # 返回详细错误
            st.error("运行失败，详细信息如下：")

            st.exception(e)

import io
import time
import tkinter as tk

import numpy as np
import streamlit as st
import torch
from matplotlib import pyplot as plt

from Deep_Learning import page2, page1_Knowledge, page3, page4_train_nn


def show():
    # 隐藏tkinter主窗口（仅用文件选择功能）
    root = tk.Tk()
    root.withdraw()  # 不显示主窗口

    st.markdown("""
        <style>
        .reportview-container .main .block-container{
            padding-left: 2rem;  /* 左边距，默认更大 */
            padding-right: 2rem; /* 右边距，默认更大 */
        }
        </style>
    """, unsafe_allow_html=True)

    page = st.sidebar.selectbox(
        "选择页面",
        ['理论学习', '回归拟合效果展示', '分类效果展示', '神经网络演示'],
    )

    if page == "理论学习":
        st.session_state["study_choose"] = st.selectbox("请选择你想要了解的知识", ["基本知识", "基本流程", "激活函数"])

        if st.session_state["study_choose"] == "基本知识":
            study_basic = page1_Knowledge.basic()
            st.markdown(study_basic)
        elif st.session_state["study_choose"] == "基本流程":
            study_progress = page1_Knowledge.progress()
            st.markdown(study_progress)
        elif st.session_state["study_choose"] == "激活函数":
            study_activation = page1_Knowledge.activation()
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
                fig_generator = page2.forward(x_data, y_data, "sigmoid")
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
                fig_generator = page2.forward(x_data, y_data, "无")
            else:
                return
            # 创建占位符用于实时更新
            plot_placeholder = st.empty()
            # 迭代生成器，逐个显示每一步的图表
            for fig in fig_generator:
                plot_placeholder.pyplot(fig)  # 用当前fig更新占位符
                plt.close(fig)  # 释放资源
                time.sleep(0.01)  # 控制更新速度

        if st.button("显示代码"):
            markdown = page1_Knowledge.reg_forward_code()
            st.markdown(markdown, unsafe_allow_html=True)
    elif page == "分类效果展示":
        if st.button("开始分类"):
            fig_generator = page3.forward()
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
        if st.button("显示代码"):
            markdown = page1_Knowledge.cls_forward_code()
            st.markdown(markdown, unsafe_allow_html=True)

    elif page == "神经网络演示":
        if "training_state" not in st.session_state:
            st.session_state["training_state"] = None
        if "net" not in st.session_state:
            st.session_state["net"] = None
        try:
            st.sidebar.header("神经网络参数")

            # 数据集选择
            dataset_type = st.sidebar.selectbox(
                "选择数据集",
                ["分类", "圆形", "月形", "自定义", "回归", "回归2.0"]
            )

            # 选择是否加入衍生特征
            selected_derived_features = st.sidebar.multiselect("是否加入衍生特征",
                                                               ["平方项", "交叉项", "正弦项", "余弦项"])

            n_features = 2  # 先初始化为2，因为分类任务的特征都是2，只有回归任务的特征数量可以调整，这里
            if dataset_type == "回归" or dataset_type == "回归2.0":
                n_features = st.sidebar.slider("特征数量", min_value=1, max_value=20, value=5)

            # base_n_features用于保存原始的特征数，用来下面计算加入衍生特征后的维度
            "这个原始特征数等会还要传入visualization的，很重要，因为生成数据要根据原始的n_features来"
            base_n_features = n_features
            if "平方项" in selected_derived_features:
                n_features += base_n_features
            if "交叉项" in selected_derived_features:
                n_features += int(base_n_features * (base_n_features - 1) / 2)
            if "正弦项" in selected_derived_features:
                n_features += base_n_features
            if "余弦项" in selected_derived_features:
                n_features += base_n_features

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

            # 选择轮数
            num_epochs = st.sidebar.slider("训练轮数", min_value=100, max_value=10000, value=500)
            # 选择批次
            batch_size = st.sidebar.selectbox("batch_size", [1, 8, 16, 32, 64])
            # 选择优化器
            optimizer = st.sidebar.selectbox("optimizer", ["SGD", "Adam", "RMSprop", "Nesterov", "SGD with Momentum"])
            # 选择学习率调度器
            # 初始化所有可能用到的参数（设置默认值）
            StepLR_step = 10
            StepLR_gamma = 0.1
            MultiStepLR_milestones = [10, 20, 30]
            MultiStepLR_gamma = 0.1
            ExponentialLR_gamma = 0.99
            CosineAnnealingLR_T_max = 50
            CosineAnnealingLR_eta_min = 0.0001
            ReduceLROnPlateau_factor = 0.2
            ReduceLROnPlateau_patience = 5
            ReduceLROnPlateau_min_lr = 0.0001
            ReduceLROnPlateau_mode = "min"
            scheduler = st.sidebar.selectbox("学习率调度器",
                                             ["StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR",
                                              "ReduceLROnPlateau","无"])

            # 选择学习率调度器及对应参数
            if scheduler == "StepLR":
                StepLR_step = st.sidebar.slider("Step size", min_value=1, max_value=50, value=10)
                StepLR_gamma = st.sidebar.slider("Gamma", min_value=0.01, max_value=0.8, value=0.1)

            elif scheduler == "MultiStepLR":
                # 允许用户输入多个调整节点，用逗号分隔
                step_list = st.sidebar.text_input("Milestones (comma separated)", "10,20,30")
                # 转换为整数列表（基础容错处理）
                MultiStepLR_milestones = [int(x.strip()) for x in step_list.split(",") if x.strip().isdigit()]
                MultiStepLR_gamma = st.sidebar.slider("Gamma", min_value=0.01, max_value=0.8, value=0.1)

            elif scheduler == "ExponentialLR":
                ExponentialLR_gamma = st.sidebar.slider("Gamma", min_value=0.9, max_value=0.999, value=0.99)
                # 指数衰减通常gamma接近1，这里调整范围更贴合实际使用场景

            elif scheduler == "CosineAnnealingLR":
                CosineAnnealingLR_T_max = st.sidebar.slider("T_max", min_value=5, max_value=100, value=50)
                # 学习率周期（通常设为总epoch的1/2或1/3）
                CosineAnnealingLR_eta_min = st.sidebar.slider("Eta min", min_value=0.00001, max_value=0.001,
                                                              value=0.0001)
                # 最小学习率

            elif scheduler == "ReduceLROnPlateau":
                ReduceLROnPlateau_factor = st.sidebar.slider("Factor", min_value=0.1, max_value=0.5, value=0.2)
                # 学习率衰减因子
                ReduceLROnPlateau_patience = st.sidebar.slider("Patience", min_value=1, max_value=20, value=5)
                # 多少个epoch指标无改善后调整
                ReduceLROnPlateau_min_lr = st.sidebar.slider("Min LR", min_value=0.00001, max_value=0.001, value=0.0001)
                # 最小学习率下限
                ReduceLROnPlateau_mode = st.sidebar.selectbox("Mode", ["min", "max"])
                # 监控指标的优化方向（min对应损失下降，max对应准确率上升）

            scheduler_dict = {
                "StepLR": {
                    "step_size": StepLR_step,
                    "gamma": StepLR_gamma
                },
                "MultiStepLR": {
                    "milestones": MultiStepLR_milestones,
                    "gamma": MultiStepLR_gamma
                },
                "ExponentialLR": {
                    "gamma": ExponentialLR_gamma
                },
                "CosineAnnealingLR": {
                    "T_max": CosineAnnealingLR_T_max,
                    "eta_min": CosineAnnealingLR_eta_min
                },
                "ReduceLROnPlateau": {
                    "factor": ReduceLROnPlateau_factor,
                    "patience": ReduceLROnPlateau_patience,
                    "min_lr": ReduceLROnPlateau_min_lr,
                    "mode": ReduceLROnPlateau_mode
                }
            }


            # 选择正则化力度
            # 初始化
            dropout_rate = 0.0
            _lamda= 0.0
            regularization_strength= st.sidebar.multiselect("正则化力度",
                                             ["dropout","L2"])
            if "dropout" in regularization_strength:
                dropout_rate = st.sidebar.slider("dropout概率",min_value=0.0,max_value=0.5,step=0.01,value=0.2)
            if "L2" in regularization_strength:
                _lamda = st.sidebar.slider("l2正则化力度",min_value=0.0001,max_value=0.01,step=0.00001,value=0.001,format="%.4f")

            # 所有参数列表
            params = {
                "active_fn": active_fn,
                "n_samples": n_samples,
                "dataset_type": dataset_type,
                'layer_sizes': layer_sizes,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                "num_epochs": num_epochs,
                "n_features": n_features,
                "selected_derived_features": selected_derived_features,
                "base_n_features": base_n_features,
                "optimizer": optimizer,
                "scheduler_dict":scheduler_dict, # 各个学习率调度器的字典
                "scheduler":scheduler,  # 学习率调度器的选择结果
                "dropout_rate":dropout_rate, # dropout_rate
                "_lamda":_lamda  #  _lamda
            }

            if st.button("释放上一个模型内存模型") and st.session_state["training_state"]==False:
                del st.session_state["net"]  # 直接删除深度学习的实例对象
                torch.cuda.empty_cache()
                st.success("模型已释放")

            if st.session_state["training_state"] == False:
                # 1. 选择文件保存路径（适用于保存模型）
                if st.button("保存模型") and st.session_state["net"] is not None :
                    buffer = io.BytesIO()
                    torch.save(st.session_state["net"].model, buffer)
                    buffer.seek(0)
                    st.download_button(
                        label="下载模型",
                        data=buffer,
                        file_name="model_final.pth",
                        mime="application/octet-stream"
                    )


            # 创建一个按钮用于停止训练,需要点击后并且此时正在训练
            if st.button("停止训练") and st.session_state["training_state"] == True:
                st.session_state["training_state"] = False
                st.rerun()

            if st.button("开始训练"):
                st.session_state["training_state"] = True
                st.write(st.session_state["training_state"])
                st.session_state["net"] = page4_train_nn.TrainNet(**params)  # 用字典批量传入参数
                fig_generator = st.session_state["net"].train()

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
                for net_fig, boundary_fig, loss, weights, epoch in fig_generator:
                    # 如果状态为True就训练
                    if st.session_state["training_state"]:
                        plot_placeholder_net.pyplot(net_fig)  # 用当前fig更新占位符
                        plt.close(net_fig)  # 释放资源

                        plot_placeholder_boundary.pyplot(boundary_fig)  # 用当前fig更新占位符
                        plt.close(boundary_fig)  # 释放资源

                        time.sleep(0.01)  # 控制更新速度

                        write_epoch.write(epoch)
                        write_loss.write(loss)
                        write_weight.write(weights)
                    # 如果状态未False则调用停止函数发送停止信号
                    elif not st.session_state["training_state"]:
                        st.session_state["net"].stop_train()  # 结束后台的计算
                        break
                # for-else特殊语法，只有当循环结束时才会触发，当训练完整结束时也将状态置为False'''
                else:
                    st.session_state["training_state"] = False


        except Exception as e:  # 返回详细错误
            st.error("运行失败，详细信息如下：")

            st.exception(e)

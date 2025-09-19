import io
import time
import tkinter as tk

import numpy as np
import streamlit as st
import torch
from matplotlib import pyplot as plt

from Deep_Learning import page2, Knowledge, page3, page4_train_nn


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
            study_basic = Knowledge.basic()
            st.markdown(study_basic)
        elif st.session_state["study_choose"] == "基本流程":
            study_progress = Knowledge.progress()
            st.markdown(study_progress)
        elif st.session_state["study_choose"] == "激活函数":
            study_activation = Knowledge.activation()
            st.markdown(study_activation)


    elif page == "回归拟合效果展示":
        st.markdown("# 拟合效果展示")
        # 选择激活函数
        a_fn = st.selectbox("请选择激活函数", ["无", "sigmoid"])
        # 开始拟合按钮
        if st.button("开始拟合"):
            # 根据选择的激活函数准备数据
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
        st.markdown("---")  # 分隔线

        # 显示代码按钮
        if st.button("显示代码"):
            markdown = Knowledge.reg_forward_code()
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
            markdown = Knowledge.cls_forward_code()
            st.markdown(markdown, unsafe_allow_html=True)

    elif page == "神经网络演示":

        # 初始化训练状态，判断是否正在训练
        if "training_state" not in st.session_state:
            st.session_state["training_state"] = None

        # 初始化net用于存储神经网络所有内容
        if "net" not in st.session_state:
            st.session_state["net"] = None
        try:
            st.sidebar.header("神经网络参数")

            # ================= 数据集与特征 =================
            with st.sidebar.expander("📊 数据设置", expanded=True):
                # 数据集选择
                dataset_type = st.selectbox(
                    "选择数据集",
                    ["分类", "圆形", "月形", "自定义", "回归", "回归2.0"]
                )

                # 选择是否加入衍生特征
                selected_derived_features = st.multiselect("是否加入衍生特征",
                                                           ["平方项", "交叉项", "正弦项", "余弦项"])
                n_features = 2  # 先初始化为2，因为分类任务的特征都是2，只有回归任务的特征数量可以调整，这里
                if dataset_type == "回归" or dataset_type == "回归2.0":
                    n_features = st.slider("特征数量", min_value=1, max_value=20, value=5)
                # base_n_features用于保存原始的特征数，用来下面计算加入衍生特征后的维度
                base_n_features = n_features
                # 衍生特征
                if "平方项" in selected_derived_features:
                    n_features += base_n_features
                if "交叉项" in selected_derived_features:
                    n_features += int(base_n_features * (base_n_features - 1) / 2)
                if "正弦项" in selected_derived_features:
                    n_features += base_n_features
                if "余弦项" in selected_derived_features:
                    n_features += base_n_features

                # 数据量
                n_samples = st.slider(
                    "数据量",
                    min_value=100,
                    max_value=5000,
                    value=1000,
                    step=100
                )

            # ================= 网络结构 =================

            with st.sidebar.expander("🧩 网络架构", expanded=True):
                hidden_layers = st.slider("隐藏层数", 1, 4, 2)
                layer_sizes = []
                layer_sizes.append(n_features)  # 输入层

                # 创建用于选择的神经网络层数
                for i in range(hidden_layers):
                    n = st.slider(f"第{i + 1}层神经元数", 2, 8, 4, key=f"layer_{i}")
                    layer_sizes.append(n)

                # 输出层
                if dataset_type == "回归" or dataset_type == "回归2.0":
                    layer_sizes.append(1)
                else:
                    layer_sizes.append(2)  # 最后加上输出层

            # ================= 训练参数 =================
            with st.sidebar.expander("⚙️ 训练参数", expanded=True):
                learning_rate = st.select_slider(
                    "学习率",
                    options=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                    value=0.01
                )

                active_fn = st.selectbox(
                    "激活函数",
                    ["relu", "tanh", "sigmoid"]
                )

                # 选择轮数
                num_epochs = st.slider("训练轮数", min_value=100, max_value=10000, value=500)
                # 选择批次
                batch_size = st.selectbox("batch_size", [1, 8, 16, 32, 64])
                # 选择优化器
                optimizer = st.selectbox("optimizer", ["SGD", "Adam", "RMSprop", "Nesterov", "SGD with Momentum"])

            # ================= 学习率调度器 =================
            with st.sidebar.expander("📉 学习率调度器", expanded=False):
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
                scheduler = st.selectbox("学习率调度器",
                                         ["StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR",
                                          "ReduceLROnPlateau", "无"])

                # 选择学习率调度器及对应参数
                if scheduler == "StepLR":
                    # 每多少个 epoch 衰减一次，一般 5~30
                    StepLR_step = st.slider("Step size", min_value=1, max_value=30, value=10)
                    # 衰减系数 gamma，常见范围 0.5~0.99
                    StepLR_gamma = st.slider("Gamma", min_value=0.5, max_value=0.99, value=0.9)

                elif scheduler == "MultiStepLR":
                    step_list = st.text_input("Milestones (comma separated)", "10,20,30")
                    MultiStepLR_milestones = [int(x.strip()) for x in step_list.split(",") if x.strip().isdigit()]
                    # gamma 建议 0.5~0.99
                    MultiStepLR_gamma = st.slider("Gamma", min_value=0.5, max_value=0.99, value=0.9)

                elif scheduler == "ExponentialLR":
                    # gamma 越接近 1，衰减越平缓
                    ExponentialLR_gamma = st.slider("Gamma", min_value=0.9, max_value=0.999, value=0.99)

                elif scheduler == "CosineAnnealingLR":
                    # T_max 一般设为总 epoch 的 1/2 或 1/3
                    CosineAnnealingLR_T_max = st.slider("T_max", min_value=5, max_value=100, value=50)
                    # eta_min 一般比初始学习率小 1~2 个数量级
                    CosineAnnealingLR_eta_min = st.slider("Eta min", min_value=1e-6, max_value=1e-3, value=1e-5)

                elif scheduler == "ReduceLROnPlateau":
                    # factor 一般 0.1~0.5
                    ReduceLROnPlateau_factor = st.slider("Factor", min_value=0.1, max_value=0.5, value=0.5)
                    # patience 一般 2~10
                    ReduceLROnPlateau_patience = st.slider("Patience", min_value=2, max_value=10, value=5)
                    # min_lr 建议设为 1e-6 ~ 1e-4
                    ReduceLROnPlateau_min_lr = st.slider("Min LR", min_value=1e-6, max_value=1e-4, value=1e-5,
                                                                 format="%.0e")
                    # mode 通常根据监控指标选择
                    ReduceLROnPlateau_mode = st.selectbox("Mode", ["min", "max"])

                # 学习率调度器参数字典
                scheduler_dict = {
                    "StepLR": {"step_size": StepLR_step, "gamma": StepLR_gamma},
                    "MultiStepLR": {"milestones": MultiStepLR_milestones, "gamma": MultiStepLR_gamma},
                    "ExponentialLR": {"gamma": ExponentialLR_gamma},
                    "CosineAnnealingLR": {"T_max": CosineAnnealingLR_T_max, "eta_min": CosineAnnealingLR_eta_min},
                    "ReduceLROnPlateau": {
                        "factor": ReduceLROnPlateau_factor,
                        "patience": ReduceLROnPlateau_patience,
                        "min_lr": ReduceLROnPlateau_min_lr,
                        "mode": ReduceLROnPlateau_mode
                    }
                }

            # ================= 正则化 =================

            with st.sidebar.expander("🛡️ 正则化", expanded=False):
                # 正则化参数初始化
                dropout_rate = 0.0
                _lamda = 0.0
                regularization_strength = st.multiselect("正则化力度", ["dropout", "L2"])
                if "dropout" in regularization_strength:
                    dropout_rate = st.slider("dropout概率", min_value=0.0, max_value=0.5, step=0.01, value=0.2)
                if "L2" in regularization_strength:
                    _lamda = st.slider("l2正则化力度", min_value=0.0001, max_value=0.01, step=0.00001,
                                       value=0.001, format="%.4f")

            # ================= 参数收集 =================

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
                "scheduler_dict": scheduler_dict,
                "scheduler": scheduler,
                "dropout_rate": dropout_rate,
                "_lamda": _lamda

            }

            # ================= 模型管理 =================

            with st.sidebar.expander("🗂️ 模型管理", expanded=True):
                if st.button("释放上一个模型内存模型") and st.session_state["training_state"] == False:
                    del st.session_state["net"]
                    torch.cuda.empty_cache()
                    st.success("模型已释放")

                if st.session_state["training_state"] == False:
                    if st.button("保存模型") and st.session_state["net"] is not None:
                        buffer = io.BytesIO()
                        torch.save(st.session_state["net"].model, buffer)
                        buffer.seek(0)
                        st.download_button(
                            label="下载模型",
                            data=buffer,
                            file_name="model_final.pth",
                            mime="application/octet-stream"
                        )

                if st.button("停止训练") and st.session_state["training_state"] == True:
                    st.session_state["training_state"] = False
                    st.rerun()

            if st.button("开始训练"):
                st.session_state["training_state"] = True
                st.write(st.session_state["training_state"])

                # 初始化模型训练类
                st.session_state["net"] = page4_train_nn.TrainNet(**params)

                # 训练模型并通过yield逐步接收参数等内容
                fig_generator = st.session_state["net"].train()

                # 占位，用于后面的绘图类绘图
                col1, col2 = st.columns([2, 2])
                with col1:
                    plot_placeholder_net = st.empty()
                with col2:
                    plot_placeholder_boundary = st.empty()
                write_epoch = st.empty()
                write_loss = st.empty()
                write_weight = st.empty()
                # 开始循环接收训练结果
                for net_fig, boundary_fig, loss, weights, epoch in fig_generator:
                    if st.session_state["training_state"]:

                        # 绘图
                        plot_placeholder_net.pyplot(net_fig)
                        plt.close(net_fig)  # 释放资源
                        plot_placeholder_boundary.pyplot(boundary_fig)
                        plt.close(boundary_fig)  # 释放资源
                        time.sleep(0.01)

                        # 打印轮数、损失和权重
                        write_epoch.write(epoch)
                        write_loss.write(loss)
                        write_weight.write(weights)

                    elif not st.session_state["training_state"]:
                        st.session_state["net"].stop_train()
                        break
                else:
                    st.session_state["training_state"] = False
        except Exception as e:  # 返回详细错误
            st.error("运行失败，详细信息如下：")

            st.exception(e)

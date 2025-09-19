import datetime
import pickle
import sqlite3
import uuid
from Traditional_ML import model, data_preprocessing, param
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# 设置显示中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
plt.rcParams["axes.unicode_minus"] = False
# st.title("机器学习训练平台")

# ========================================================登录页面=====================================================
#
# # SQLite 初始化
# conn = sqlite3.connect("../users.db")
# c = conn.cursor()
# c.execute('''
# CREATE TABLE IF NOT EXISTS users (
#     username TEXT PRIMARY KEY,
#     password TEXT
# )
# ''')
# conn.commit()
#
# # 初始化登录状态
# if 'logged_in' not in st.session_state:
#     st.session_state.logged_in = False
# if 'username' not in st.session_state:
#     st.session_state.username = ''
#
#
# # 登录功能
# def login_user(username, password):
#     c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
#     data = c.fetchone()
#     if data:
#         st.session_state.logged_in = True
#         st.session_state.username = username
#         st.success(f"登录成功，欢迎 {username}!")
#     else:
#         st.error("用户名或密码错误")
#
#
# # 注册功能
# def register_user(username, password):
#     try:
#         c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
#         conn.commit()
#         st.success("注册成功！请登录")
#     except sqlite3.IntegrityError:
#         st.error("用户名已存在，请换一个用户名")
# # ================== 页面控制 ==================
#
#
#
# # 登录/注册页面
# def login_page():
#     st.title("🔒 登录 / 注册")
#     tab = st.tabs(["登录", "注册"])
#
#     # 登录
#     with tab[0]:
#         username_login = st.text_input("用户名", key="login_user")
#         password_login = st.text_input("密码", type="password", key="login_pass")
#         if st.button("登录", key="login_btn"):
#             login_user(username_login, password_login)
#
#     # 注册
#     with tab[1]:
#         username_reg = st.text_input("用户名", key="reg_user")
#         password_reg = st.text_input("密码", type="password", key="reg_pass")
#         if st.button("注册", key="reg_btn"):
#             register_user(username_reg, password_reg)


# # ========================================================主页面=====================================================
def show():
    # 初始化 session_state
    if "model_save_list" not in st.session_state:
        st.session_state["model_save_list"] = []  # 初始化为空列表
    # 在侧边栏创建选择器
    page = st.sidebar.selectbox(
        "选择页面",
        ["数据读取", "数据预处理", "训练模型", "模型保存"]
    )

    if page == "数据读取":
        st.title("🏠 上传您的数据")
        st.session_state["files"] = st.file_uploader(
            "请选择上传数据",
            type=["txt", "csv"],
            accept_multiple_files=True
        )


        @st.cache_data
        def load_data(uploaded_file):
            return pd.read_csv(uploaded_file)

        if st.session_state["files"]:  # 只要有文件上传
            for file in st.session_state["files"]:
                st.subheader(f"文件: {file.name}")
                df = load_data(file)
                st.session_state["df"] = df
                st.dataframe(df.head())
    # ==========================================================================================
    elif page == "数据预处理":
        if "files" not in st.session_state or not st.session_state["files"]:
            st.warning("⚠️ 还未上传文件，请先前往文件上传页面上传数据")
        else:
            # 初始化预处理类
            preprocessing = data_preprocessing.DataPreprocessor(data=st.session_state["df"])

            def show_basic_info():
                """
                使用 st.session_state['basic'] 展示数据基本信息，
                支持通过下拉选择展示不同内容：
                - 形状
                - 空值统计柱状图
                - 重复行数量柱状图
                - 原始数据
                """
                if "basic" not in st.session_state:
                    st.warning(
                        "st.session_state['basic'] 不存在，请先调用 preprocessing.get_basic_info() 并保存到 session_state。")
                    return

                basic = st.session_state["basic"]

                # 下拉选择要展示的内容
                options = ["形状", "空值统计", "重复行数量", "数据样本"]
                choice = st.multiselect(
                    "📋 选择要展示的基础信息",
                    options,
                    default=options,
                    key=f"basic_info_multiselect_{uuid.uuid4()}"  # 自动生成唯一 key
                )

                # 基础信息展示区 - 多列布局优化
                if choice:
                    with st.container(border=True, height=400):
                        # 形状 + 重复行 一行展示
                        if "形状" in choice or "重复行数量" in choice:
                            col_shape, col_dup = st.columns(2)
                            # 形状
                            with col_shape:
                                if "形状" in choice:
                                    st.subheader("📏 数据形状")
                                    st.info(f"行数：{basic['形状'][0]} | 列数：{basic['形状'][1]}")
                            # 重复行数量
                            with col_dup:
                                if "重复行数量" in choice:
                                    st.subheader("🔄 重复行统计")
                                    duplicated_count = basic.get("重复行数量", 0)
                                    fig, ax = plt.subplots(figsize=(4, 3))
                                    ax.bar(["重复行"], [int(duplicated_count)], color="#FF6B6B")
                                    ax.set_ylabel("行数")
                                    st.pyplot(fig)

                        # 空值统计单独一行
                        if "空值统计" in choice:
                            st.subheader("❌ 空值分布统计")
                            null_counts = basic["空值数量"]
                            if null_counts:
                                fig, ax = plt.subplots(figsize=(12, 4))
                                bars = ax.bar(null_counts.keys(), [int(v) for v in null_counts.values()],
                                              color="#4ECDC4")
                                ax.set_xticklabels(null_counts.keys(), rotation=45, ha='right')
                                ax.set_ylabel("空值数量")
                                # 在每个柱子上显示具体数值
                                for bar in bars:
                                    yval = bar.get_height()
                                    ax.text(bar.get_x() + bar.get_width() / 2, yval, int(yval),
                                            ha='center', va='bottom', fontweight='bold')
                                st.pyplot(fig)
                            else:
                                st.success("✅ 当前数据无空值")

                        # 数据样本单独一行
                        if "数据样本" in choice and "df" in st.session_state:
                            st.subheader("📄 数据样本预览")
                            st.dataframe(st.session_state["df"].head(10), use_container_width=True)

            # 页面标题与说明
            st.header("🧹 数据预处理中心", divider="rainbow")
            st.caption("支持空值处理、重复值删除、数据编码等常见预处理操作，实时预览处理效果")

            # 第一区域：数据基础信息展示
            with st.container():
                st.subheader("📊 数据基础信息", anchor="basic-info")
                st.divider()
                # 1.先展示数据的空值情况
                st.session_state["null_sum"] = preprocessing.get_data_null()
                st.session_state["basic"] = preprocessing.get_basic_info()
                show_basic_info()

            # 第二区域：列选择模式（单列/多列）
            st.divider()
            with st.container(border=True):
                st.subheader("🎯 操作列选择", anchor="column-selection")
                col_mode, col_cols = st.columns([1, 2])

                with col_mode:
                    # 选择单列/多列模式
                    column_choose_mode = st.selectbox(
                        "选择操作模式",
                        ["单列", "多列"],
                        format_func=lambda x: "📌 单列操作" if x == "单列" else "📌 多列操作"
                    )

                with col_cols:
                    # 根据模式展示不同的列选择器
                    if column_choose_mode == "单列":
                        # 构建列列表（包含“全部”选项）
                        columns = ["全部"] + st.session_state["null_sum"].index.to_list()
                        select_option = st.selectbox("选择目标列", columns)
                        # 记录选择的列至全局状态
                        if select_option == "全部":
                            st.session_state["column"] = [col for col in columns if col != "全部"]
                            st.caption(f"✅ 已选择所有列（共{len(st.session_state['column'])}列）")
                        else:
                            st.session_state["column"] = select_option
                            st.caption(f"✅ 已选择列：{select_option}")
                    else:
                        columns = st.session_state["null_sum"].index.to_list()
                        select_option = st.multiselect("选择目标列（可多选）", columns)
                        st.session_state["column"] = select_option
                        if select_option:
                            st.caption(f"✅ 已选择 {len(select_option)} 列：{', '.join(select_option)}")
                        else:
                            st.warning("⚠️ 请至少选择一列进行操作")

            # 第三区域：核心预处理操作（标签页分组）
            st.divider()
            st.subheader("⚙️ 预处理操作", anchor="preprocess-ops")
            # 创建三个主标签页：去空、去重、编码
            tb_drop_null, tb_drop_repeat, tb_get_encoding = st.tabs([
                "🗑️ 数据去空",
                "🔍 数据去重",
                "🔢 数据编码"
            ])

            # 1. 数据去空标签页
            with tb_drop_null:
                st.caption("选择空值处理方式，处理后将实时更新数据预览")
                # 去空方式子标签页
                tb_drop_null_1, tb_drop_null_2, tb_drop_null_3 = st.tabs([
                    "删除空值行",
                    "均值填充",
                    "删除空值列"
                ])

                with tb_drop_null_1:
                    st.info("ℹ️ 功能说明：删除包含空值的行（适用于空值占比低的场景）")
                    tb_drop_null_1_btn = st.button(
                        "🚀 开始删除空值行",
                        key="tb_drop_null_1_btn",
                        use_container_width=True,
                        type="primary"
                    )
                    if tb_drop_null_1_btn:
                        preprocessing.drop_null_rows(columns=st.session_state["column"])
                        st.session_state["df"] = preprocessing.data
                        st.success("✅ 空值行删除完成！")
                        # 再次展示更新后的数据
                        st.session_state["basic"] = preprocessing.get_basic_info()
                        show_basic_info()

                with tb_drop_null_2:
                    st.info("ℹ️ 功能说明：用该列均值填充空值（仅适用于数值型列）")
                    tb_drop_null_2_btn = st.button(
                        "🚀 开始均值填充",
                        key="tb_drop_null_2_btn",
                        use_container_width=True,
                        type="primary"
                    )
                    if tb_drop_null_2_btn:
                        preprocessing.fill_null_with_mean(columns=st.session_state["column"])
                        st.session_state["df"] = preprocessing.data
                        st.success("✅ 均值填充空值完成！")
                        # 再次展示更新后的数据
                        st.session_state["basic"] = preprocessing.get_basic_info()
                        show_basic_info()

                with tb_drop_null_3:
                    st.info("ℹ️ 功能说明：直接删除包含空值的列（适用于空值占比极高的场景）")
                    tb_drop_null_3_btn = st.button(
                        "🚀 开始删除空值列",
                        key="tb_drop_null_3_btn",
                        use_container_width=True,
                        type="primary"
                    )
                    if tb_drop_null_3_btn:
                        preprocessing.drop_null_columns(st.session_state["column"])
                        st.session_state["df"] = preprocessing.data
                        st.success("✅ 空值列删除完成！")
                        # 再次展示更新后的数据
                        st.session_state["basic"] = preprocessing.get_basic_info()
                        show_basic_info()

            # 2. 数据去重标签页
            with tb_drop_repeat:
                st.caption("删除重复行，默认按所选列判断重复（全列对比）")
                tb_drop_repeat_1, = st.tabs(["删除重复行"])

                with tb_drop_repeat_1:
                    st.info("ℹ️ 功能说明：删除完全重复的行，保留第一行非重复数据")
                    tb_drop_repeat_1_btn = st.button(
                        "🚀 开始删除重复行",
                        key="tb_drop_repeat_1_btn",
                        use_container_width=True,
                        type="primary"
                    )
                    if tb_drop_repeat_1_btn:
                        preprocessing.remove_duplicates(columns=st.session_state["column"])
                        st.session_state["df"] = preprocessing.data
                        st.success("✅ 重复行删除完成！")
                        # 再次展示更新后的数据
                        st.session_state["basic"] = preprocessing.get_basic_info()
                        show_basic_info()

            # 3. 数据编码标签页
            with tb_get_encoding:
                st.caption("将非数值型特征转换为数值型，适配模型训练需求")
                tb_get_encoding_1, tb_get_encoding_2, = st.tabs([
                    "独热编码",
                    "标签编码"
                ])

                with tb_get_encoding_1:
                    st.info("ℹ️ 功能说明：对分类特征做独热编码（无顺序关系，如颜色：红/蓝/绿）")
                    tb_get_encoding_btn_1 = st.button(
                        "🚀 开始独热编码",
                        key="tb_get_encoding_btn_1",
                        use_container_width=True,
                        type="primary"
                    )
                    if tb_get_encoding_btn_1:
                        preprocessing.get_dummy_data(st.session_state["column"])
                        st.session_state["df"] = preprocessing.data
                        st.success("✅ 独热编码处理完成！")
                        # 再次展示更新后的数据
                        st.session_state["basic"] = preprocessing.get_basic_info()
                        show_basic_info()

                with tb_get_encoding_2:
                    st.info("ℹ️ 功能说明：对分类特征做标签编码（有顺序关系，如等级：高/中/低）")
                    tb_get_encoding_btn_2 = st.button(
                        "🚀 开始标签编码",
                        key="tb_get_encoding_btn_2",
                        use_container_width=True,
                        type="primary"
                    )
                    if tb_get_encoding_btn_2:
                        preprocessing.Label_Encoding(st.session_state["column"])
                        st.session_state["df"] = preprocessing.data
                        st.success("✅ 标签编码处理完成！")
                        # 再次展示更新后的数据
                        st.session_state["basic"] = preprocessing.get_basic_info()
                        show_basic_info()
    # ==========================================================================================

    elif page == "训练模型":
        if "files" not in st.session_state or not st.session_state["files"]:
            st.warning("还未上传文件，请上传文件")
        else:
            with st.container():
                st.subheader("📊 模型训练设置")
                st.divider()
                # 模式选择（分类或回归）
                col1, col2 = st.columns(2)
                with col1:
                    mode = st.selectbox("选择模式", ["回归", "分类"],
                                        format_func=lambda x: "📈 " + x if x == "回归" else "🔍 " + x)
                # 模型选择
                with col2:
                    if mode == "回归":
                        select_option = st.selectbox("选择模型",
                                                     ["KNN", "线性回归", "决策树", "随机森林", "梯度提升树", "支持向量机"])
                    else:
                        select_option = st.selectbox("选择模型",
                                                     ["KNN", "逻辑回归", "决策树", "随机森林", "梯度提升树", "支持向量机"])
                # 检查mode或模型是否变化
                if ("selected_model" not in st.session_state
                        or st.session_state["selected_model"] != select_option
                        or st.session_state.get("mode") != mode):
                    st.session_state["selected_model"] = select_option
                    st.session_state["mode"] = mode
                    # 清空参数
                    if "param_values" in st.session_state:
                        del st.session_state["param_values"]
                st.divider()
                st.subheader("🔢 数据设置")
                # 列表选择
                columns_options = st.session_state["df"].columns.tolist()
                col_y, col_x = st.columns(2)
                # 确定y
                with col_y:
                    column_y = st.selectbox("选择目标列 y", columns_options)
                # 确定X
                with col_x:
                    column_X = st.multiselect("选择特征列 X（可多选）", columns_options,
                                              default=[col for col in columns_options if col != column_y])
                # 数据集划分设置
                with st.expander("⚙️ 数据集划分参数", expanded=True):
                    split_cols = st.columns(2)
                    with split_cols[0]:
                        test_size = st.number_input("测试集占比（0-1）", value=0.2)
                    with split_cols[1]:
                        random_state = st.number_input("随机种子 (整数)", min_value=0, step=1, value=42, format="%d")
                my_model = model.model_choice(select_option, st.session_state["df"], test_size=test_size,
                                              random_state=random_state
                                              , X_columns=column_X, y_column=column_y, mode=st.session_state["mode"])
                # 模型参数设置
                st.divider()
                st.subheader("🛠️ 模型参数设置")
                # 确定参数列表
                param_options = my_model.get_params()
                param_options_map = param.param_options_map()
                param_types = param.param_types()
                # 让用户先选择要调整的参数
                selected_params = st.multiselect("选择要调整的参数", param_options)
                # 不同的参数采取不同的组件，防止类型错误
                param_dict = {}
                if selected_params:
                    with st.container(border=True):
                        for i, p in enumerate(selected_params):
                            p_type = param_types.get(p, "text")
                            param_col = st.columns(len(selected_params))[i] if len(selected_params) <= 3 else st.columns(3)[
                                i % 3]
                            with param_col:
                                if p_type == "select":
                                    param_dict[p] = st.selectbox(f"{p}:", param_options_map[p], key=f"{p}_select")
                                elif p_type == "int":
                                    param_dict[p] = st.number_input(f"{p}:", step=1, format="%d", key=f"{p}_int")
                                elif p_type == "float":
                                    param_dict[p] = st.number_input(f"{p}:", format="%.4f", key=f"{p}_float")
                                elif p_type == "bool":
                                    param_dict[p] = st.checkbox(f"{p}:", key=f"{p}_bool")
                                else:
                                    param_dict[p] = st.text_input(f"{p}:", key=f"{p}_text")
                # 保存到 session_state
                st.session_state["param_values"] = param_dict
                if param_dict:
                    with st.expander("📋 当前参数值", expanded=False):
                        st.write(st.session_state["param_values"])
                my_model.update_params(st.session_state["param_values"])
                # 最后实例化模型
                my_model.init_model()
                st.divider()

                # 训练按钮
                if st.button("🚀 开始训练", use_container_width=True):
                    try:
                        with st.spinner("训练中，请等待..."):
                            my_model.train()
                        st.success("✅ 训练完成！")
                        with st.expander("📊 评估结果", expanded=True):
                            st.write("正在评估...")
                            name, ret_dict = my_model.evaluate(column_y)
                            st.write("模型名：", name)
                            st.write(ret_dict)
                        st.session_state["train_state"] = True
                        st.session_state["my_model"] = my_model
                        # 保存训练好的模型
                        st.session_state["model_save_list"].append({
                            "model": my_model.model,
                            "name": my_model.model_name})
                    except Exception as e:
                        st.warning("⚠️ 数据存在字符串或存在空值，请移步数据预处理界面进一步处理")
                # 可视化部分
                st.divider()
                st.subheader("📈 可视化分析")
                if mode == "回归":
                    viz_cols = st.columns(2)
                    with viz_cols[0]:
                        if st.button("真实值 vs 预测值", use_container_width=True):
                            if st.session_state.get("train_state"):
                                fig, ax = st.session_state["my_model"].plot_true_pred()
                                st.pyplot(fig, use_container_width=True)
                                # 绘制完成后不需要再绘制，关闭训练状态
                                st.session_state["train_state"] = False
                            else:
                                st.warning("请先进行训练，否则无法绘制图像")

                    with viz_cols[1]:
                        if st.button("预测值拟合真实值", use_container_width=True):
                            if st.session_state.get("train_state"):
                                fig, ax = st.session_state["my_model"].plot_fit_true()
                                st.pyplot(fig, use_container_width=True)
                                # 绘制完成后不需要再绘制，关闭训练状态
                                st.session_state["train_state"] = False
                            else:
                                st.warning("请先进行训练，否则无法绘制图像")

                if mode == "分类":
                    if st.button("绘制决策边界", use_container_width=True):
                        if st.session_state.get("train_state"):
                            with st.spinner("绘制决策边界中..."):
                                st.write("决策边界可视化：")
                                fig, ax = st.session_state["my_model"].plot_boundary()
                                st.pyplot(fig, use_container_width=True)
                            # 绘制完成后不需要再绘制，关闭训练状态
                            st.session_state["train_state"] = False
                        else:

                            st.warning("请先进行训练，否则无法绘制决策边界")

                if st.session_state["selected_model"] == "决策树":

                    if st.button("🌳 进行树可视化", use_container_width=True):
                        with st.spinner("绘制决策树中..."):
                            fig, ax = st.session_state["my_model"].plot_tree()

                            st.pyplot(fig, use_container_width=True)

    elif page == "模型保存":
        # 页面标题与视觉分割
        st.header("💾 模型保存中心", divider="rainbow")
        st.caption("管理、选择并下载已训练完成的模型，支持批量操作")

        # 第一区域：已保存模型列表
        with st.container():
            st.subheader("📋 已保存模型库", anchor="saved-models")
            st.divider()

            # 检查是否有已保存模型
            if st.session_state["model_save_list"]:
                # 模型列表卡片式展示
                for idx, model_info in enumerate(st.session_state["model_save_list"]):
                    model_name = model_info.get("name", f"未知模型_{idx + 1}")
                    # 单个模型卡片（带边框+阴影）
                    with st.container(border=True, height=100):
                        col_idx, col_info, col_status = st.columns([1, 3, 1])

                        with col_idx:
                            st.markdown(f"**模型 {idx + 1}**")
                            st.markdown(f"<span style='color:#6c757d; font-size:0.8rem'>ID: {idx + 1}</span>",
                                        unsafe_allow_html=True)

                        with col_info:
                            st.markdown(f"📦 **模型名称**: {model_name}")
                            st.markdown(f"<span style='color:#28a745; font-size:0.8rem'>✅ 已就绪</span>",
                                        unsafe_allow_html=True)

                        with col_status:
                            st.metric(label="状态", value="可下载", delta="", delta_color="normal")

            else:
                # 无模型时的空状态提示
                with st.container(border=True, height=150, align="center"):
                    st.empty()  # 占位用
                    col_icon, col_text = st.columns([1, 3])
                    with col_icon:
                        st.markdown("<div style='font-size:3rem; color:#adb5bd'>📁</div>", unsafe_allow_html=True)
                    with col_text:
                        st.subheader("暂无已保存模型")
                        st.warning("⚠️ 请先前往「训练模型」页面训练并保存模型")
                    st.empty()  # 占位用
                return  # 没有模型则不显示后续内容

        # 第二区域：模型选择与下载
        st.divider()
        with st.container(border=True):
            st.subheader("🎯 选择并下载模型", anchor="download-models")
            st.caption("支持多选，点击下载按钮获取模型文件（.pkl格式）")

            # 构建模型选择选项（带名称预览）
            model_options = [
                f"模型 {idx + 1} | {model_info.get('name', '未知模型')}"
                for idx, model_info in enumerate(st.session_state["model_save_list"])
            ]

            # 多选框选择要下载的模型
            models_to_save = st.multiselect(
                "请选择需要下载的模型",
                options=model_options,
                key="models_to_save",
                help="按住Ctrl键可多选模型"
            )

            # 下载按钮区域（批量展示）
            if models_to_save:
                st.markdown("### 📥 下载选中模型")
                st.divider()
                # 每行一个下载按钮（全屏宽度+主色调）
                for model_label in models_to_save:
                    try:
                        # 解析模型索引（从"模型 X | 名称"中提取X）
                        model_index = int(model_label.split(" | ")[0].split(" ")[1]) - 1
                        model_info = st.session_state["model_save_list"][model_index]
                        model_obj = model_info.get("model", None)
                        model_name = model_info.get("name", f"模型_{model_index + 1}")

                        if model_obj is not None:
                            # 序列化模型
                            model_bytes = pickle.dumps(model_obj)
                            # 下载按钮（主色调+图标+全屏）
                            st.download_button(
                                label=f"🚀 下载模型：{model_name}",
                                data=model_bytes,
                                file_name=f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                                mime="application/octet-stream",
                                use_container_width=True,
                                type="primary",
                                help=f"点击下载 {model_name} 模型文件",
                                key=f"{uuid.uuid4()}"
                            )
                        else:
                            # 模型对象不存在的错误提示
                            with st.container(border=True):
                                st.markdown(f"❌ 模型 {model_name}")
                                st.error("模型对象不存在，无法下载")

                    except Exception as e:
                        # 其他错误提示
                        with st.container(border=True):
                            st.markdown(f"❌ 处理失败：{model_label}")
                            st.error(f"错误信息：{str(e)[:50]}...")  # 截断长错误信息
            else:
                # 未选择模型时的提示
                st.info("ℹ️ 请从上方列表中选择至少一个模型进行下载")
# 运行命令
# streamlit run "F:\py_MLA_Learning_Hub\MLA_Learning_Hub\UI.py"
import pickle
import sqlite3
import uuid
import model
import param
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import data_preprocessing


# 设置显示中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
plt.rcParams["axes.unicode_minus"] = False
st.title("机器学习训练平台")

# ========================================================登录页面=====================================================

# SQLite 初始化
conn = sqlite3.connect("users.db")
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
''')
conn.commit()

# 初始化登录状态
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''


# 登录功能
def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    data = c.fetchone()
    if data:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.success(f"登录成功，欢迎 {username}!")
    else:
        st.error("用户名或密码错误")


# 注册功能
def register_user(username, password):
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        st.success("注册成功！请登录")
    except sqlite3.IntegrityError:
        st.error("用户名已存在，请换一个用户名")


# 登录/注册页面
def login_page():
    st.title("🔒 登录 / 注册")
    tab = st.tabs(["登录", "注册"])

    # 登录
    with tab[0]:
        username_login = st.text_input("用户名", key="login_user")
        password_login = st.text_input("密码", type="password", key="login_pass")
        if st.button("登录", key="login_btn"):
            login_user(username_login, password_login)

    # 注册
    with tab[1]:
        username_reg = st.text_input("用户名", key="reg_user")
        password_reg = st.text_input("密码", type="password", key="reg_pass")
        if st.button("注册", key="reg_btn"):
            register_user(username_reg, password_reg)


# # ========================================================主页面=====================================================
def main_page():
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
            st.warning("还未上传文件，请上传文件")
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
                    "选择要展示的内容",
                    options,
                    default=options,
                    key=f"basic_info_multiselect_{uuid.uuid4()}"  # 自动生成唯一 key
                )

                # 形状
                if "形状" in choice:
                    with st.expander("数据形状", expanded=True):
                        st.write(basic["形状"])

                # 空值统计柱状图
                if "空值统计" in choice:
                    with st.expander("空值统计", expanded=False):
                        null_counts = basic["空值数量"]
                        if null_counts:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            bars = ax.bar(null_counts.keys(), [int(v) for v in null_counts.values()])
                            ax.set_xticklabels(null_counts.keys(), rotation=45, ha='right')
                            ax.set_ylabel("空值数量")

                            # 在每个柱子上显示具体数值
                            for bar in bars:
                                yval = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width() / 2, yval, int(yval),
                                        ha='center', va='bottom')
                            st.pyplot(fig)
                        else:
                            st.write("无空值。")

                # 重复行数量柱状图
                if "重复行数量" in choice:
                    with st.expander("重复行数量", expanded=False):
                        duplicated_count = basic.get("重复行数量", 0)
                        fig, ax = plt.subplots(figsize=(4, 3))
                        ax.bar(["重复行"], [int(duplicated_count)])
                        ax.set_ylabel("行数")
                        st.pyplot(fig)

                # 数据样本
                if "数据样本" in choice and "df" in st.session_state:
                    with st.expander("🔹 数据样本", expanded=False):
                        st.dataframe(st.session_state["df"])

            # 1.先展示数据的空值情况
            st.write("空值分布情况")
            st.session_state["null_sum"] = preprocessing.get_data_null()
            st.session_state["basic"] = preprocessing.get_basic_info()
            show_basic_info()

            # 2.选择需要去空的列
            #
            st.write("单列操作或多列操作？")
            column_choose_mode = st.selectbox("选择列", ["单列", "多列"])
            if column_choose_mode == "单列":
                # 构建列列表

                columns = ["全部"] + st.session_state["null_sum"].index.to_list()

                # 显示下拉框并获取选择列
                select_option = st.selectbox("选择列", columns)
                if select_option == "全部":  # 如果是全部设置column了
                    st.session_state["column"] = [col for col in columns if col != "全部"]
                else:
                    # 记录选择的列至全局状态
                    st.session_state["column"] = select_option
            else:
                columns = st.session_state["null_sum"].index.to_list()
                select_option = st.multiselect("选择列", columns)
                st.session_state["column"] = select_option

            # 3. 选择需要进行的操作
            # 创建两个标签页
            tb_drop_null, tb_drop_repeat, tb_get_encoding = st.tabs(["数据去空", "数据去重", "数据编码"])
            # 去空页面
            with tb_drop_null:
                # 创建三个用于不同去空方式的页面
                tb_drop_null_1, tb_drop_null_2, tb_drop_null_3 = st.tabs(["删除空值行", "插入均值", "删除列"])
                with tb_drop_null_1:
                    # 必须要指定key，因为两个文本一样，不指定key会报错
                    tb_drop_null_1_btn = st.button("开始去空", key="tb_drop_null_1_btn")
                    if tb_drop_null_1_btn:
                        preprocessing.drop_null_rows(columns=st.session_state["column"])
                        # 更新 st.session_state["df"] 为处理后的数据
                        st.session_state["df"] = preprocessing.data
                        # 再次展示数据
                        st.session_state["basic"] = preprocessing.get_basic_info()
                        show_basic_info()
                with tb_drop_null_2:
                    # 必须要指定key，因为两个文本一样，不指定key会报错
                    tb_drop_null_2_btn = st.button("开始去空", key="tb_drop_null_2_btn")
                    if tb_drop_null_2_btn:
                        preprocessing.fill_null_with_mean(columns=st.session_state["column"])
                        # 更新 st.session_state["df"] 为处理后的数据
                        st.session_state["df"] = preprocessing.data
                        # 再次展示数据
                        st.session_state["basic"] = preprocessing.get_basic_info()
                        show_basic_info()
                with tb_drop_null_3:
                    tb_drop_null_3_btn = st.button("删除列")
                    if tb_drop_null_3_btn:
                        preprocessing.drop_null_columns(st.session_state["column"])
                        # 更新 st.session_state["df"] 为处理后的数据
                        st.session_state["df"] = preprocessing.data
                        # 再次展示数据
                        st.session_state["basic"] = preprocessing.get_basic_info()
                        show_basic_info()

            with tb_drop_repeat:
                # 创建三个用于不同去空方式的页面
                tb_drop_repeat_1, = st.tabs(["删除重复行"])
                with tb_drop_repeat_1:
                    # 必须要指定key，因为两个文本一样，不指定key会报错
                    tb_drop_repeat_1_btn = st.button("开始去重", key="tb_drop_repeat_1_btn")
                    if tb_drop_repeat_1_btn:
                        preprocessing.remove_duplicates(columns=st.session_state["column"])
                        # 更新 st.session_state["df"] 为处理后的数据
                        st.session_state["df"] = preprocessing.data
                        # 再次展示数据
                        st.session_state["basic"] = preprocessing.get_basic_info()
                        show_basic_info()
            with tb_get_encoding:
                tb_get_encoding_1, tb_get_encoding_2, = st.tabs(["独热编码", "标签编码"])
                with tb_get_encoding_1:
                    # 必须要指定key，因为两个文本一样，不指定key会报错
                    tb_get_encoding_btn_1 = st.button("开始处理选择的非数值列", key="tb_get_encoding_btn_1")
                    if tb_get_encoding_btn_1:
                        preprocessing.get_dummy_data(st.session_state["column"])
                        # 更新 st.session_state["df"] 为处理后的数据
                        st.session_state["df"] = preprocessing.data
                        # 再次展示数据
                        st.session_state["basic"] = preprocessing.get_basic_info()
                        show_basic_info()
                with tb_get_encoding_2:
                    tb_get_encoding_btn_2 = st.button("开始编码选择的列", key="tb_get_encoding_btn_2")
                    if tb_get_encoding_btn_2:
                        preprocessing.Label_Encoding(st.session_state["column"])
                        # 更新 st.session_state["df"] 为处理后的数据
                        st.session_state["df"] = preprocessing.data
                        # 再次展示数据
                        st.session_state["basic"] = preprocessing.get_basic_info()
                        show_basic_info()
    # ==========================================================================================

    elif page == "训练模型":
        if "files" not in st.session_state or not st.session_state["files"]:
            st.warning("还未上传文件，请上传文件")
        else:
            # 模式选择（分类或回归）
            mode = st.selectbox("选择分类或回归模式", ["回归", "分类"])
            # 模型选择
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

            # 列表选择
            columns_options = st.session_state["df"].columns.tolist()
            # 确定y
            column_y = st.selectbox("选择目标列y", columns_options)
            # 确定X
            column_X = st.multiselect("选择特征列X（可多选）", columns_options,
                                      default=[col for col in columns_options if col != column_y])

            # 确定test_size
            test_size = st.number_input("请输入测试集占比（0-1）", value=0.2)
            # 确定random_state
            random_state = st.number_input("请输入 random_state (整数)", min_value=0, step=1, value=42, format="%d")
            my_model = model.model_choice(select_option, st.session_state["df"], test_size=test_size,
                                          random_state=random_state
                                          , X_columns=column_X, y_column=column_y, mode=st.session_state["mode"])

            # 确定参数列表
            param_options = my_model.get_params()

            param_options_map = param.param_options_map()
            param_types = param.param_types()

            # 让用户先选择要调整的参数
            selected_params = st.multiselect("选择要调整的参数", param_options)
            # 不同的参数采取不同的组件，防止类型错误
            param_dict = {}
            for p in selected_params:
                p_type = param_types.get(p, "text")
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
            st.write("当前参数值:", st.session_state["param_values"])

            my_model.update_params(st.session_state["param_values"])
            # 最后实例化模型
            my_model.init_model()

            # 划分测试集训练集

            train_btn = st.button("开始训练")

            if train_btn:
                try:
                    st.write("训练中，请等待")

                    my_model.train()
                    st.write("训练完成，正在评估")
                    name,ret_dict = my_model.evaluate(column_y)
                    st.write("评估结果：")
                    st.write("模型名：",name)
                    st.write(ret_dict)
                    st.session_state["train_state"] = True
                    st.session_state["my_model"] = my_model

                    # 保存训练好的模型
                    st.session_state["model_save_list"].append({
                        "model": my_model.model,
                        "name": my_model.model_name})
                except Exception as e:
                    st.warning("数据存在字符串或存在空值，请移步数据预处理界面进一步处理")
            if mode == "回归":
                if st.button("绘制真实值vs预测值"):
                    if st.session_state.get("train_state"):
                        fig, ax = st.session_state["my_model"].plot_true_pred()
                        st.pyplot(fig, use_container_width=True)
                        # 绘制完成后不需要再绘制，关闭训练状态
                        st.session_state["train_state"] = False

                    else:
                        st.warning("请先进行训练，否则无法绘制图像")
                if st.button("绘制预测值拟合真实值图像"):
                    if st.session_state.get("train_state"):
                        fig, ax = st.session_state["my_model"].plot_fit_true()
                        st.pyplot(fig, use_container_width=True)
                        # 绘制完成后不需要再绘制，关闭训练状态
                        st.session_state["train_state"] = False

                    else:
                        st.warning("请先进行训练，否否则无法绘制图像")

            if mode == "分类":
                if st.button("绘制决策边界"):
                    if st.session_state.get("train_state"):

                        st.write("绘制决策边界")
                        fig, ax = st.session_state["my_model"].plot_boundary()
                        st.pyplot(fig, use_container_width=True)
                        # 绘制完成后不需要再绘制，关闭训练状态
                        st.session_state["train_state"] = False

                    else:
                        st.warning("请先进行训练，否则无法绘制决策边界")

            if st.session_state["selected_model"] == "决策树":
                if st.button("进行树可视化"):
                    fig, ax = st.session_state["my_model"].plot_tree()
                    st.pyplot(fig, use_container_width=True)

    elif page == "模型保存":

        st.title("模型保存")
        # 显示已保存的模型列表
        if st.session_state["model_save_list"]:
            st.write("已保存的模型列表：")
            for idx, model_info in enumerate(st.session_state["model_save_list"]):
                st.write(f"模型 {idx + 1}: {model_info.get('name', '未知模型')}")
        else:
            st.warning("还没有保存任何模型。")
            return  # 没有模型就不显示下面的选择和下载

        # 用户选择需要保存的模型
        model_options = [f"模型 {idx + 1}" for idx in range(len(st.session_state["model_save_list"]))]
        models_to_save = st.multiselect(
            "选择需要保存的模型",
            model_options,
            key="models_to_save"
        )

        # 提供下载按钮
        for model_label in models_to_save:
            try:
                model_index = int(model_label.split(" ")[1]) - 1
                model_info = st.session_state["model_save_list"][model_index]
                model_obj = model_info.get('model', None)
                model_name = model_info.get('name', f"模型_{model_index + 1}")
                if model_obj is not None:
                    model_bytes = pickle.dumps(model_obj)
                    st.download_button(
                        label=f"下载 {model_name}",
                        data=model_bytes,
                        file_name=f"{model_name}.pkl",
                        mime="application/octet-stream"
                    )
                else:
                    st.error(f"{model_name} 对象不存在，无法下载。")
            except Exception as e:
                st.error(f"处理 {model_label} 时发生错误: {e}")


# ================== 页面控制 ==================
if not st.session_state.logged_in:
    login_page()
else:
    main_page()

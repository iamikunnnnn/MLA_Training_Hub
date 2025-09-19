# 运行命令
# streamlit run "F:\py_MLA_Learning_Hub\MLA_Learning_Hub\app.py"
# 重建模型参数

import streamlit as st

# 1. 子模块导入增加容错，避免 app 直接崩溃
try:
    from Traditional_ML import UI as ml_ui

    has_ml = True
except (ImportError, ModuleNotFoundError) as e:
    has_ml = False
    ml_error = str(e)

try:
    from Deep_Learning import UI as dl_ui

    has_dl = True
except (ImportError, ModuleNotFoundError) as e:
    has_dl = False
    dl_error = str(e)

try:
    from Deep_Learning_CNN import UI as dl_cnn_ui

    has_dl_cnn = True
except (ImportError, ModuleNotFoundError) as e:
    has_dl_cnn = False
    dl_cnn_error = str(e)

# 2. 初始化状态：用明确的标识替代 None，降低歧义
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "select"  # "select" = 选择模式，"ml" = 传统ML，"dl" = 深度学习


# 3. 定义模式配置（新增模式只需加一行，提升扩展性）
MODE_CONFIG = {
    "ml": {
        "name": "传统机器学习",
        "ui_module": ml_ui if has_ml else None,
        "available": has_ml,
        "error": ml_error if not has_ml else ""
    },
    "dl": {
        "name": "深度学习-基本学习",
        "ui_module": dl_ui if has_dl else None,
        "available": has_dl,
        "error": dl_error if not has_dl else ""
    },
    "dl_cnn": {
        "name": "深度学习-CNN",
        "ui_module": dl_cnn_ui if has_dl_cnn else None,
        "available": has_dl_cnn,
        "error": dl_cnn_error if not has_dl_cnn else ""
    }
}

# ----------------- 1. 模式选择界面（仅当 current_mode 为 "select" 时渲染） -----------------
if st.session_state.current_mode == "select":
    st.title("请选择模式")

    # 过滤可用的模式（隐藏不可用的）
    available_modes = [(key, cfg["name"]) for key, cfg in MODE_CONFIG.items() if cfg["available"]]
    if not available_modes:
        st.error("所有模式的依赖模块均未找到，请检查文件结构！")
    else:
        # 模式选择（用 key 绑定状态，避免重复选择）
        selected_mode_key = st.radio(
            "选择一个模式：",
            options=[item[0] for item in available_modes],
            format_func=lambda x: MODE_CONFIG[x]["name"],  # 显示友好名称
            key="mode_radio"  # 唯一 key，避免组件复用冲突
        )

        # 进入按钮（无需 rerun，点击后仅更新状态，页面自动重新渲染）
        if st.button("进入所选模式"):
            st.session_state.current_mode = selected_mode_key  # 直接切换状态，无刷新

# ----------------- 2. 子模式界面（根据 current_mode 渲染，无需 rerun） -----------------
else:
    # 获取当前模式的配置
    current_cfg = MODE_CONFIG[st.session_state.current_mode]
    # st.title(f"当前模式：{current_cfg['name']}")

    # 子界面渲染（增加容错，避免子模块报错崩溃）
    try:
        current_cfg["ui_module"].show()
    except Exception as e:
        st.error(f"子界面加载失败：{str(e)}")

    # 返回按钮（仅更新状态，页面自动重新渲染）
    if st.button("返回模式选择"):
        st.session_state.current_mode = "select"

        # （可选）清空子界面的临时状态（避免返回后残留数据）
        for key in st.session_state.keys():
            if key not in ["current_mode"]:  # 保留核心状态
                del st.session_state[key]
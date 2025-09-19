
import time
import traceback

import requests
import streamlit as st


def show():

        page = st.sidebar.selectbox(
            "选择页面",
            ['历史网络'],
        )

        if page == "历史网络":
            try:
                network = st.sidebar.selectbox('网络结构', ['LeNet'])

                if st.button("开始训练"):  # 新增一个开始训练的按钮（原逻辑可能自动触发，这里改为手动触发更合理）
                    # 1. 通知 Flask 后端开始训练
                    start_response = requests.post(
                        "http://localhost:5000/api/train",
                        json={"conv1_out": 6,
                              "conv2_out":16,
                              "num_epochs":10,
                              "batch_size":16}
                    )

                    if start_response.json().get("status") == "success":
                        st.success("训练已开始，正在获取过程数据...")

                        # 创建占位符用于实时更新
                        plot_placeholder = st.empty()
                        step = 1

                        # 2. 循环获取训练过程数据（轮询后端）
                        while True:
                            next_response = requests.get("http://localhost:5000/api/train/next")
                            result = next_response.json()

                            if result["status"] == "running":
                                # 显示当前步骤的损失和参数（和原逻辑一致）
                                with plot_placeholder.container():
                                    st.write(f"步骤 {step}：")
                                    st.write(f"损失：{result['loss']}")
                                    # st.write(f"参数：{result['params']}")
                                step += 1
                                time.sleep(0.5)  # 控制轮询频率，避免请求过于频繁

                            elif result["status"] == "completed":
                                st.success("训练已完成！")
                                break

                            else:
                                # 关键修改：打印完整响应，避免遗漏错误信息
                                st.error(f"训练出错：后端返回完整结果 → {result}")
                                # 同时单独提取错误信息（兼容后端可能的字段差异）
                                error_msg = result.get('message') or result.get('error') or "未知错误"
                                st.error(f"错误详情：{error_msg}")
                                break
            except (ValueError, ZeroDivisionError) as e:
                # 获取异常的详细信息
                print(f"捕获异常: {e}")
                # 打印异常发生的位置（文件名、行号等）
                print("异常发生在:")
                traceback.print_exc()  # 打印完整的异常追踪信息
                return e
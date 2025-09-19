# backend.py（完整的 Flask 后端，整合训练逻辑）
from flask import Flask, request, jsonify
from flask_cors import CORS  # 解决跨域问题
import torch
import get_dataset  # 导入同目录的 get_dataset
from Deep_Learning_CNN.page1 import LeNet, TrainNet  # 假设你的训练逻辑在 LeNet_Train.py 中（如果当前文件就是，可删除这行，直接用下面的类）

# ---------------------- 1. 初始化 Flask 服务 ----------------------
app = Flask(__name__)
CORS(app)  # 允许跨域（Streamlit 和 Flask 端口不同必须加）

# 全局变量：存储训练迭代器（供轮询接口使用）
trainer_iterator = None



# ---------------------- 2. 定义 Flask API 接口 ----------------------
@app.route('/api/train', methods=['POST'])
def start_train():
    """接口1：接收前端参数，启动训练"""
    global trainer_iterator  # 引用全局迭代器

    try:
        # 1. 获取前端传来的训练参数（如卷积层通道数、迭代次数等）
        data = request.json
        conv1_out = data.get('conv1_out', 6)  # 默认值6
        conv2_out = data.get('conv2_out', 16)  # 默认值16a
        num_epochs = data.get('num_epochs', 10)  # 默认10轮（避免1000轮太长）
        batch_size = data.get('batch_size', 16)  # 默认批次16

        # 2. 初始化训练类（调用你的 TrainNet）
        trainer = TrainNet(
            conv1_out_channels=conv1_out,
            conv2_out_channels=conv2_out,
            num_epochs=num_epochs,
            batch_size=batch_size
        )

        # 3. 启动训练生成器，获取迭代器（不直接执行，供后续轮询）
        trainer_iterator = iter(trainer.train())

        # 4. 返回启动成功的响应
        return jsonify({
            "status": "success",
            "message": "训练已启动，可调用 /api/train/next 获取进度"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/train/next', methods=['GET'])
def get_train_step():
    """接口2：轮询接口，获取每一步训练结果（损失、参数）"""
    global trainer_iterator

    try:
        # 1. 从迭代器获取下一步结果（即训练的一轮数据）
        loss, params = next(trainer_iterator)

        # 2. 返回结果给前端（JSON 可序列化）
        return jsonify({
            "status": "running",
            "loss": loss,  # 损失值（float）
            "params": params  # 模型参数（列表字典）
        })

    except StopIteration:
        # 3. 训练结束（迭代器耗尽）
        return jsonify({"status": "completed", "message": "训练全部结束"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------------- 3. 启动 Flask 服务 ----------------------
if __name__ == '__main__':
    # 启动服务，端口5000（必须和Streamlit调用的端口一致）
    app.run(
        host='0.0.0.0',  # 允许局域网访问
        port=5000,  # 端口号，和前端调用地址对应
        debug=True  # 开发模式（生产环境关闭）
    )

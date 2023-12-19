import os
import argparse

# 文件路径
current_path = os.path.abspath(os.path.dirname(__file__))  # src
dir_path = os.path.dirname(current_path)  # 项目
data_path = os.path.join(dir_path, "data")  # 数据集
model_path = os.path.join(dir_path, "pretrained_model")  # 预训练的模型文件

# 模型名
bert_base = os.path.join(model_path, "hfl-chinese-bert-wwm-ext")  # 基线模型
LTP = os.path.join(model_path, "LTP-base1")

# 模型保存
save_model_path = os.path.join(dir_path, "save_models")
# 训练日志记录
save_log_path = os.path.join(dir_path, "logs")

parser = argparse.ArgumentParser()

parser.add_argument('--save_model_path', type=str, default=save_model_path)
parser.add_argument('--save_log_path', type=str, default=save_log_path)

parser.add_argument('--pretrained_model', type=str, default=bert_base)
parser.add_argument('--LTP', type=str, default=LTP)
parser.add_argument('--data_path', type=str, default=data_path)


parser.add_argument('--input_dim', type=int, default=768)
parser.add_argument('--output_dim', type=int, default=768)
parser.add_argument('--head', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--class_nums', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=5)
# 随机种子
parser.add_argument('--seed', type=int, default=2023)

hparams = parser.parse_args()

import argparse
import configparser
import os

from texttable import Texttable


def get_parser():
    parser = argparse.ArgumentParser(description="Graph Neural Network Parameter Parser")

    """
    Dataset相关配置
    """
    parser.add_argument('--config', type=str, default='../../config/CGEDN.ini', help='配置文件路径')
    parser.add_argument('--dataset', type=str, default='AIDS_700', help='使用的数据集')
    parser.add_argument('--data-location', type=str, default='../../data', help='数据集文件夹')
    parser.add_argument('--training-set', type=str, default='real', help='训练使用真实数据集还是合成数据集[real, syn]')
    parser.add_argument('--testing-set', type=str, default='real', help='测试使用真实数据集还是合成数据集[real, syn]')
    parser.add_argument('--syn-num', type=int, default=100, help='训练集中的每个图合成多少个图对')

    """
    Training相关配置
    """
    # 超参
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability. Default is 0.5.")
    parser.add_argument('--learning-rate', type=float, default=0.001, help='学习率')
    parser.add_argument("--weight-decay", type=float, default=5 * 10 ** -4,
                        help="Adam weight decay. Default is 5*10^-4.")
    # 训练参数
    parser.add_argument("--use-gpu", type=bool, default=False, help='是否使用GPU')
    parser.add_argument("--epoch-start", type=int, default=0, help="加载已经训练的epoch")
    parser.add_argument("--epoch-end", type=int, default=20, help="要训练多少epoch")
    parser.add_argument('--batch-size', type=int, default=128, help='每批次的样本数量')
    parser.add_argument("--model-train", type=bool, default=True, help='是否训练')
    
    """
    Model相关配置
    """
    parser.add_argument("--model-name", type=str, default='SimGNN', help="模型名称")
    parser.add_argument('--model-path', type=str, default='./model_save', help='模型文件的保存路径')
    parser.add_argument("--gnn-filters", type=str, default='64-32-16', help="GNN模块的filters")
    parser.add_argument("--tensor-neurons", type=int, default=16, help="NTN模块的输出维度.")
    parser.add_argument("--reg-neurons", type=str, default='16-8-4', help="预测相似度分数的MLP各层的维度")
    parser.add_argument("--target-mode", type=str, default='exp', help="计算相似度分数的方式，有[linear, exp].")
    # SimGNN模型参数
    parser.add_argument("--histogram", type=bool, default=True, help="是否使用直方图")
    parser.add_argument("--bins", type=int, default=16, help="直方图bin的数量")
    # CGEDN模型参数
    parser.add_argument("--lamb", type=float, default=0.01, help="mapping loss所占的权重")
    parser.add_argument("--max-iter", type=int, default=10, help="Sinkhorn的最大迭代次数")
    parser.add_argument("--tau", type=float, default=1., help="超参，越小则Sinkhorn的结果越接近Hungarian")
    # GEDGNN模型参数
    parser.add_argument("--value-loss-weight", type=float, default=1.0, help="value loss的权重")
    parser.add_argument("--hidden-dim", type=int, default=16, help="交叉矩阵模块的层数")

    return parser


def parse_config_file(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    # 返回一个字典，包含配置文件中的所有部分和参数
    return {key: value for section in config.sections() for key, value in config.items(section)}


def update_args_with_config(args, config):
    # 更新命令行参数，如果在配置文件中定义了相同的参数
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, type(getattr(args, key))(value))


def tab_printer(args):
    """
    打印模型的参数
    :param args:
    :return:
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_row(keys)
    t.add_row([args[k] for k in keys])
    t.set_max_width(1000)
    print(t.draw())

    # 检查目录是否存在，如果不存在则创建
    directory_path = f'{args["model_path"]}/{args["model_name"]}/{args["dataset"]}-{args["training_set"]}_{args["testing_set"]}/'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    with open(f'{directory_path}/results.txt', 'a') as f:
        print(t.draw(), file=f)

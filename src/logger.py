import os
import time
import argparse
import logging
import yaml

import argparse


def str2bool(value):
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')





def reset_log(log_path):
    import logging
    fileh = logging.FileHandler(log_path, 'a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileh.setFormatter(formatter)
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)
    log.setLevel(logging.DEBUG)

def make_logger(train_time):
    # 加载默认配置
    config = load_config('config.yaml')

    # 解析命令行参数
    args = cmdline_args()
    args = merge_config_with_args(config, args)

    # 计算日志文件夹路径
    log_dir =os.path.join(args.log_file,args.model,args.dataset)
    # log_dir = os.path.abspath(args.log_file + args.model + args.dataset)

    # 检查并创建文件夹
    if not os.path.exists(args.log_file):
        print(f"Creating base log directory: {args.log_file}")
        os.makedirs(args.log_file)
    if not os.path.exists(log_dir):
        print(f"Creating dataset-specific log directory: {log_dir}")
        os.makedirs(log_dir)

    # 打印路径调试信息
    # print(f"Log directory: {log_dir}")

    # 设置日志文件的完整路径
    log_file_path = os.path.join(log_dir, str(train_time) + str(args.description)+ '.log')
    print(f"Log file path: {log_file_path}")
    reset_log(log_file_path)
    # 设置日志配置
    # logging.basicConfig(level=logging.INFO,
    #                     filename=log_file_path,
    #                     datefmt='%Y/%m/%d %H:%M:%S',
    #                     format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s',
    #                     filemode='w')

    logger = logging.getLogger(__name__)

    # 测试日志输出
    # logger.info("This is a test log message.")
    return logger, args

def cmdline_args():
    # 创建 argparse 解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Dataset name: toys, amazon_beauty, steam, ml-1m')
    parser.add_argument('--log_file', help='log dir path')
    parser.add_argument('--random_seed', type=int, help='Random seed')
    parser.add_argument('--max_len', type=int, help='The max length of sequence')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0', 'cuda:1'], help='Device: cpu, cuda:0, cuda:1')
    parser.add_argument('--num_gpu', type=int, help='Number of GPU')
    parser.add_argument('--batch_size', type=int, help='Batch Size')
    parser.add_argument("--hidden_size", type=int, help="hidden size of model")
    parser.add_argument('--dropout', type=float, help='Dropout of representation')
    parser.add_argument('--emb_dropout', type=float, help='Dropout of item embedding')
    parser.add_argument("--hidden_act", type=str, help="Activation function: gelu or relu")
    # parser.add_argument('--num_blocks', type=int, help='Number of denoised decoder blocks')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--decay_step', type=int, help='Decay step for StepLR')
    parser.add_argument('--gamma', type=float, help='Gamma for StepLR')
    parser.add_argument('--metric_ks', nargs='+', type=int, help='ks for Metric@k')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], help='Optimizer choice: SGD or Adam')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--loss_lambda', type=float, help='loss weight for diffusion')
    parser.add_argument('--weight_decay', type=float, help='L2 regularization')
    parser.add_argument('--momentum', type=float, help='SGD momentum')
    parser.add_argument('--schedule_sampler_name', type=str, help='Diffusion for t generation')
    parser.add_argument('--diffusion_steps', type=int, help='Diffusion step')
    parser.add_argument('--lambda_uncertainty', type=float, help='uncertainty weight')
    parser.add_argument('--lambda_schedule', type=str2bool, help='use lambda schedule')
    parser.add_argument('--lambda_beta_a', type=float, help='uncertainty weight')
    parser.add_argument('--lambda_beta_b', type=float, help='uncertainty weight')
    parser.add_argument('--noise_schedule', type=str, help='Noise schedule')
    parser.add_argument('--beta_a', type=float)
    parser.add_argument('--beta_b', type=float)
    parser.add_argument('--rescale_timesteps', help='rescale timesteps')
    parser.add_argument('--eval_interval', type=int, help='the number of epoch to eval')
    parser.add_argument('--patience', type=int, help='the number of epoch to wait before early stop')
    parser.add_argument('--long_head', type=str2bool, help='Long and short sequence, head and long-tail items')
    parser.add_argument('--diversity_measure', type=str2bool, help='Measure the diversity of recommendation results')
    parser.add_argument('--epoch_time_avg', type=str2bool, help='Calculate the average time of one epoch training')
    parser.add_argument('--dif_decoder', type=str, choices=['att', 'mlp'], help='Choose denoised decoder')
    parser.add_argument('--split_onebyone', type=str2bool, help='Split sequence one by one')
    parser.add_argument('--parallel_ag', type=str2bool, help='Train in a per token auto-aggressive manner')
    parser.add_argument('--is_causal', type=str2bool, help='Use causal attention')
    parser.add_argument('--dif_objective', type=str, choices=['pred_noise', 'pred_x0', 'pred_v'],
                        help='Choose diffusion loss objective')
    parser.add_argument('--pretrained', type=str2bool, help='use pretrained embedding weight')
    parser.add_argument('--freeze_emb', type=str2bool, help='freezing embedding weight')
    parser.add_argument('--model', type=str)
    parser.add_argument('--loss', type=str, choices=['ce', 'mse'])
    parser.add_argument('--loss_scale', type=float)
    parser.add_argument('--cfg_scale', type=float)
    parser.add_argument('--description', type=str)
    parser.add_argument('--pcgrad', type=str2bool)
    parser.add_argument('--geodesic', type=str2bool)
    # 解析命令行参数
    args = parser.parse_args()

    return args
def load_config(config_file):
    # 模拟加载配置
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def merge_config_with_args(config, args):
    # 将 YAML 配置字典转为 Namespace 对象
    config_namespace = argparse.Namespace(**config)

    # 使用命令行参数覆盖配置字典中的值
    for key, value in vars(args).items():
        if value is not None:
            setattr(config_namespace, key, value)

    return config_namespace

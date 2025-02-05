

from logger import make_logger
import torch
import pprint
import pickle
from trainer import model_train, LSHT_inference,load_data,choose_model,item_num_create
from utils import *
# import yaml
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from utils import Data_Train,Data_Val,Data_Test



def main():
    start_time = time.time()
    train_time=time.strftime("%Y-%m-%d_%H-%M-%S",  time.localtime())
    logger,args = make_logger(train_time)
    fix_random_seed_as(args.random_seed)
    args = item_num_create(args)
    model = choose_model(args)
    tra_data_loader, val_data_loader, test_data_loader = load_data(args)

    # cold_hot_long_short(data_raw, args.dataset)
    print(args.description)
    logger.info(args.description)
    print(args)
    formatted_args = "\n".join(f"{key}: {value}" for key, value in vars(args).items())
    logger.info("Arguments:\n%s", formatted_args)
    # print(args)
    best_model, test_results = model_train(model,tra_data_loader, val_data_loader, test_data_loader, args, logger,train_time)
    training_duration_seconds = time.time()-start_time
    minutes = training_duration_seconds // 60
    seconds = training_duration_seconds % 60
    logger.info(f"Training duration: {minutes} minutes and {seconds} seconds")

if __name__ == '__main__':
    main()

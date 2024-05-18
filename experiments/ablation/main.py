from experiments.ablation.model.trainer import Trainer
from utils.parameter_parser import *


def main():
    # 示例用法
    parser = get_parser()
    args = parser.parse_args()

    # args.__setattr__("config", "../../config/CGEDN/CGEDN_no_crs-AIDS_700-real_real.ini")
    # args.__setattr__("config", "../../config/CGEDN/CGEDN_no_crs-AIDS_small-real_real.ini") 
    # args.__setattr__("config", "../../config/CGEDN/CGEDN_no_crs-Linux-real_real.ini") 
    args.__setattr__("config", "../../config/CGEDN/CGEDN_no_crs-IMDB_small-real_real.ini") 

    # args.__setattr__("config", "../../config/CGEDN/CGEDN_no_multi_view-AIDS_700-real_real.ini")
    # args.__setattr__("config", "../../config/CGEDN/CGEDN_no_multi_view-AIDS_small-real_real.ini")
    # args.__setattr__("config", "../../config/CGEDN/CGEDN_no_multi_view-Linux-real_real.ini")


    # 如果提供了配置文件路径，从配置文件中读取参数并更新
    if args.config is not None:
        config = parse_config_file(args.config)
        update_args_with_config(args, config)

    tab_printer(args)

    trainer = Trainer(args)

    if args.epoch_start > 0:
        trainer.load(args.epoch_start)
    if args.model_train:
        for epoch in range(args.epoch_start, args.epoch_end):
            trainer.fit()
            trainer.save(epoch + 1)
            trainer.score()
            # if args.model_name in ['CGEDN', 'GEDGNN']:
            #     trainer.score_best_k(best_k=100)
    else:
        # trainer.score()
        if args.model_name in ['CGEDN', 'GEDGNN']:
            trainer.score_best_k(best_k=100)


def ablation_on_inter_gconv():
    configs = [
        "../../config/CGEDN/CGEDN_no_InterGConv-AIDS_small-real_real.ini",
        "../../config/CGEDN/CGEDN_no_InterGConv-AIDS_700-real_real.ini",
        "../../config/CGEDN/CGEDN_no_InterGConv-Linux-real_real.ini",
        "../../config/CGEDN/CGEDN_no_InterGConv-IMDB_small-real_real.ini"
    ]

    for cfg in configs:
        parser = get_parser()
        args = parser.parse_args()
        args.__setattr__("config", cfg) 
        config = parse_config_file(args.config)
        update_args_with_config(args, config)
        tab_printer(args)

        trainer = Trainer(args)

        for epoch in range(args.epoch_start, args.epoch_end):
            trainer.fit()
            trainer.save(epoch + 1)
            trainer.score()

def ablation_on_multi_view_matching():
    configs = [
        # "../../config/CGEDN/CGEDN_no_MultiViewMatching-AIDS_small-real_real.ini",
        # "../../config/CGEDN/CGEDN_no_MultiViewMatching-AIDS_700-real_real.ini",
        # "../../config/CGEDN/CGEDN_no_MultiViewMatching-Linux-real_real.ini",
        "../../config/CGEDN/CGEDN_no_MultiViewMatching-IMDB_small-real_real.ini"
    ]

    for cfg in configs:
        parser = get_parser()
        args = parser.parse_args()
        args.__setattr__("config", cfg) 
        config = parse_config_file(args.config)
        update_args_with_config(args, config)
        tab_printer(args)

        trainer = Trainer(args)

        for epoch in range(args.epoch_start, args.epoch_end):
            trainer.fit()
            trainer.save(epoch + 1)
            trainer.score()

def ablation_on_bias():
    configs = [
        "../../config/CGEDN/CGEDN_no_bias-AIDS_small-real_real.ini",
        "../../config/CGEDN/CGEDN_no_bias-AIDS_700-real_real.ini",
        "../../config/CGEDN/CGEDN_no_bias-Linux-real_real.ini",
        "../../config/CGEDN/CGEDN_no_bias-IMDB_small-real_real.ini"
    ]

    for cfg in configs:
        parser = get_parser()
        args = parser.parse_args()
        args.__setattr__("config", cfg) 
        config = parse_config_file(args.config)
        update_args_with_config(args, config)
        tab_printer(args)

        trainer = Trainer(args)

        for epoch in range(args.epoch_start, args.epoch_end):
            trainer.fit()
            trainer.save(epoch + 1)
            trainer.score()

def default_model_train():
    configs = [
        "../../config/CGEDN/CGEDN-AIDS_small-real_real.ini",
        "../../config/CGEDN/CGEDN-AIDS_700-real_real.ini",
        "../../config/CGEDN/CGEDN-Linux-real_real.ini",
        "../../config/CGEDN/CGEDN-IMDB_small-real_real.ini"
    ]

    for cfg in configs:
        parser = get_parser()
        args = parser.parse_args()
        args.__setattr__("config", cfg) 
        config = parse_config_file(args.config)
        update_args_with_config(args, config)
        tab_printer(args)

        trainer = Trainer(args)

        for epoch in range(args.epoch_start, args.epoch_end):
            trainer.fit()
            trainer.save(epoch + 1)
            trainer.score()



def best_k_evaluation():
    configs = [
        "../../config/CGEDN/CGEDN-AIDS_small-real_real.ini",
        "../../config/CGEDN/CGEDN-AIDS_700-real_real.ini",
        "../../config/CGEDN/CGEDN-Linux-real_real.ini",
        "../../config/CGEDN/CGEDN-IMDB_small-real_real.ini"
    ]

    best_ks = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for cfg in configs:
        parser = get_parser()
        args = parser.parse_args()
        args.__setattr__("config", cfg) 
        config = parse_config_file(args.config)
        update_args_with_config(args, config)
        tab_printer(args)

        trainer = Trainer(args)

        for epoch in range(args.epoch_start, args.epoch_end):
            trainer.fit()
            trainer.save(epoch + 1)
            trainer.score()
    

if __name__ == "__main__":
    # ablation_on_bias()
    # ablation_on_multi_view_matching()
    # ablation_on_inter_gconv()
    default_model_train()

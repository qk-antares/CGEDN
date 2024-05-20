from experiments.compare.model.trainer import Trainer
from utils.parameter_parser import *


def main():
    # 示例用法
    parser = get_parser()
    args = parser.parse_args()

    # args.__setattr__("model_train", False)
    # args.__setattr__("epoch_start", 20)

    # SimGNN
    # args.__setattr__("config", "../../config/SimGNN/SimGNN-Linux-real_real.ini")
    # args.__setattr__("config", "../../config/SimGNN/SimGNN-AIDS_700-real_real.ini")
    # args.__setattr__("config", "../../config/SimGNN/SimGNN-IMDB_small-real_real.ini")

    # GEDGNN
    # args.__setattr__("config", "../../config/GEDGNN/GEDGNN-Linux-real_real.ini")
    # args.__setattr__("config", "../../config/GEDGNN/GEDGNN-AIDS_700-real_real.ini")
    # args.__setattr__("config", "../../config/GEDGNN/GEDGNN-IMDB_small-real_real.ini")

    # TaGSim
    # args.__setattr__("config", "../../config/TaGSim/TaGSim-AIDS_700-real_real.ini")
    # args.__setattr__("config", "../../config/TaGSim/TaGSim-AIDS_small-real_real.ini")
    # args.__setattr__("config", "../../config/TaGSim/TaGSim-Linux-real_real.ini")
    # args.__setattr__("config", "../../config/TaGSim/TaGSim-IMDB_small-real_real.ini")

    # CGEDN
    # args.__setattr__("config", "../../config/CGEDN/CGEDN-IMDB_small-real_real.ini")
    # args.__setattr__("config", "../../config/CGEDN/CGEDN-AIDS_700-real_real.ini")
    # args.__setattr__("config", "../../config/CGEDN/CGEDN-Linux-real_real.ini") 
    # args.__setattr__("config", "../../config/CGEDN/CGEDN-AIDS_small-real_real.ini") 
    args.__setattr__("config", "../../config/CGEDN/CGEDN-IMDB_large-syn_syn.ini") 

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


def CGEDN_train():
    configs = [
        # "../../config/CGEDN/CGEDN-IMDB_large-syn_syn.ini",
        # "../../config/CGEDN/CGEDN-IMDB_small-real_real.ini",
        # "../../config/CGEDN/CGEDN-AIDS_large-syn_syn.ini",
        # "../../config/CGEDN/CGEDN-AIDS_small-real_real.ini",
        "../../config/CGEDN/CGEDN-AIDS_700-real_real.ini",
        "../../config/CGEDN/CGEDN-Linux-real_real.ini",
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


def TaGSim_train():
    configs = [
        # "../../config/CGEDN/CGEDN-AIDS_small-real_real.ini",
        # "../../config/CGEDN/CGEDN-AIDS_700-real_real.ini",
        # "../../config/CGEDN/CGEDN-Linux-real_real.ini",
        # "../../config/CGEDN/CGEDN-IMDB_small-real_real.ini",
        "../../config/TaGSim/TaGSim-IMDB_large-syn_syn.ini",
        "../../config/TaGSim/TaGSim-AIDS_large-syn_syn.ini",
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



def GEDGNN_train():
    configs = [
        # "../../config/GEDGNN/GEDGNN-Linux-real_real.ini",
        # "../../config/GEDGNN/GEDGNN-AIDS_700-real_real.ini",
        # "../../config/GEDGNN/GEDGNN-IMDB_small-real_real.ini",
        "../../config/GEDGNN/GEDGNN-IMDB_large-syn_syn.ini",
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


def SimGNN_train():
    configs = [
        # "../../config/SimGNN/SimGNN-Linux-real_real.ini",
        # "../../config/SimGNN/SimGNN-AIDS_700-real_real.ini",
        # "../../config/SimGNN/SimGNN-IMDB_small-real_real.ini",
        "../../config/SimGNN/SimGNN-IMDB_large-syn_syn.ini",
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


if __name__ == "__main__":
    CGEDN_train()
    # SimGNN_train()
    # TaGSim_train()
    # GEDGNN_train()

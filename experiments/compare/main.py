from experiments.compare.model.trainer import Trainer
from utils.parameter_parser import *


def CGEDN_train():
    configs = [
        "../../config/CGEDN/CGEDN-AIDS_700-real_real.ini",
        # "../../config/CGEDN/CGEDN-IMDB_large-syn_syn.ini",
        # "../../config/CGEDN/CGEDN-IMDB_small-real_real.ini",
        # "../../config/CGEDN/CGEDN-AIDS_large-syn_syn.ini",
        # "../../config/CGEDN/CGEDN-AIDS_small-real_real.ini",
        # "../../config/CGEDN/CGEDN-Linux-real_real.ini",
    ]

    for cfg in configs:
        parser = get_parser()
        args = parser.parse_args()
        args.__setattr__("config", cfg) 
        config = parse_config_file(args.config)
        update_args_with_config(args, config)
        tab_printer(args)

        trainer = Trainer(args)

        if args.epoch_start > 0:
            trainer.load(args.epoch_start)

        for epoch in range(args.epoch_start, args.epoch_end):
            trainer.fit()
            trainer.save(epoch + 1)
            trainer.score()


def TaGSim_train():
    configs = [
        # "../../config/CGEDN/CGEDN-AIDS_small-real_real.ini",
        "../../config/CGEDN/CGEDN-AIDS_700-real_real.ini",
        # "../../config/CGEDN/CGEDN-Linux-real_real.ini",
        # "../../config/CGEDN/CGEDN-IMDB_small-real_real.ini",
        # "../../config/TaGSim/TaGSim-IMDB_large-syn_syn.ini",
        # "../../config/TaGSim/TaGSim-AIDS_large-syn_syn.ini",
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
        "../../config/GEDGNN/GEDGNN-AIDS_700-real_real.ini",
        # "../../config/GEDGNN/GEDGNN-IMDB_small-real_real.ini",
        # "../../config/GEDGNN/GEDGNN-IMDB_large-syn_syn.ini",
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
        "../../config/SimGNN/SimGNN-AIDS_700-real_real.ini",
        # "../../config/SimGNN/SimGNN-IMDB_small-real_real.ini",
        # "../../config/SimGNN/SimGNN-IMDB_large-syn_syn.ini",
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

def CGEDN_best_k_evaluation():
    configs = [
        # ("../../config/CGEDN/CGEDN-IMDB_large-syn_syn.ini", [19, 20]),
        # ("../../config/CGEDN/CGEDN-IMDB_small-real_real.ini", [17, 18, 19, 20]),
        # ("../../config/CGEDN/CGEDN-AIDS_large-syn_syn.ini", [19]),
        # ("../../config/CGEDN/CGEDN-AIDS_small-real_real.ini", [17, 18, 19, 20]),
        ("../../config/CGEDN/CGEDN-AIDS_700-real_real.ini", [17, 18, 19, 20]),
        # ("../../config/CGEDN/CGEDN-Linux-real_real.ini", [17, 18, 19, 20])
    ]

    for cfg in configs:
        parser = get_parser()
        args = parser.parse_args()
        args.__setattr__("config", cfg[0]) 
        config = parse_config_file(args.config)
        update_args_with_config(args, config)
        tab_printer(args)

        trainer = Trainer(args)

        for epoch in cfg[1]:
            trainer.load(epoch)
            trainer.score_best_k(best_k=100)
        
def GEDGNN_best_k_evaluation():
    configs = [
        ("../../config/GEDGNN/GEDGNN-IMDB_large-syn_syn.ini", [17, 18, 19, 20]),
        ("../../config/GEDGNN/GEDGNN-IMDB_small-real_real.ini", [17, 18, 19, 20]),
        ("../../config/GEDGNN/GEDGNN-AIDS_700-real_real.ini", [17, 18, 19, 20]),
        ("../../config/GEDGNN/GEDGNN-Linux-real_real.ini", [17, 18, 19, 20])
    ]

    for cfg in configs:
        parser = get_parser()
        args = parser.parse_args()
        args.__setattr__("config", cfg[0]) 
        config = parse_config_file(args.config)
        update_args_with_config(args, config)
        tab_printer(args)

        trainer = Trainer(args)

        for epoch in cfg[1]:
            trainer.load(epoch)
            trainer.score_best_k(best_k=100)

def different_k_evaluation():
    configs = [
        # ("../../config/CGEDN/CGEDN-IMDB_large-syn_syn.ini", [19, 20]),
        # ("../../config/CGEDN/CGEDN-IMDB_small-real_real.ini", [17, 18, 19, 20]),
        # ("../../config/CGEDN/CGEDN-AIDS_large-syn_syn.ini", [17, 18, 19, 20]),
        # ("../../config/CGEDN/CGEDN-AIDS_small-real_real.ini", [17, 18, 19, 20]),
        # ("../../config/CGEDN/CGEDN-AIDS_700-real_real.ini", [17, 18, 19, 20]),
        # ("../../config/CGEDN/CGEDN-Linux-real_real.ini", [17, 18, 19, 20])
    ]

    for cfg in configs:
        parser = get_parser()
        args = parser.parse_args()
        args.__setattr__("config", cfg[0]) 
        config = parse_config_file(args.config)
        update_args_with_config(args, config)
        tab_printer(args)

        trainer = Trainer(args)

        for epoch in cfg[1]:
            trainer.load(epoch)
            trainer.score_best_k(best_k=100)

if __name__ == "__main__":
    CGEDN_train()
    # GEDGNN_best_k_evaluation()
    # CGEDN_best_k_evaluation()
    # SimGNN_train()
    # GEDGNN_train()
    # TaGSim_train()

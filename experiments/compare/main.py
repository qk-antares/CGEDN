from experiments.compare.model.trainer import Trainer
from utils.parameter_parser import *


def main():
    # 示例用法
    parser = get_parser()
    args = parser.parse_args()

    args.__setattr__("model_train", False)
    args.__setattr__("epoch_start", 20)

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
    args.__setattr__("config", "../../config/CGEDN/CGEDN-Linux-real_real.ini") 
    # args.__setattr__("config", "../../config/CGEDN/CGEDN-AIDS_small-real_real.ini") 

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

if __name__ == "__main__":
    main()

import argparse

from mindspore import context

from .engine import Trainer
from .model import vqa_model
from .utils import Tokenizer
from .data import build_dataset
from .config import get_default_cfg

def reset_cfg(cfg, args):
    if args.question_dir:
        cfg["que_path"] = args.question_dir

    if args.image_path:
        cfg["img_path"] = args.image_path

    if args.answer_dir:
        cfg["ans_path"] = args.answer_dir

    if args.glove_dir:
        cfg["glove_path"] = args.glove_dir

    if args.embd_dir:
        cfg["embd_path"] = args.embd_dir

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir


def setup_cfg(args):
    cfg = get_default_cfg()

    # 1. From the dataset config file
    if args.dataset_config_file:
        #cfg.merge_from_file(args.dataset_config_file)
        pass
    # 2. From the method config file
    if args.config_file:
        #cfg.merge_from_file(args.config_file)
        pass
    # 3. From input arguments
    reset_cfg(cfg, args)

    return cfg


def main(args):
    cfg = {}
    
    data_parser = Tokenizer(cfg)
    data_parser.parse()


    model = vqa_model(cfg)
    data_parser = Tokenizer(cfg)
    train_loader = build_dataset(cfg, data_parser, "train")
    val_loader = build_dataset(cfg, data_parser, "val")
    test_loader = build_dataset(cfg, data_parser, "test")
    trainer = Trainer(model, cfg, train_loader, val_loader, test_loader)

    if args.eval_only:
        ##trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:  
        trainer.train()

    

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--question-dir", type=str, default="", help="path to question data"
    )
    parser.add_argument(
        "--image-path", type=str, default="", help="path to image file"
    )
    parser.add_argument(
        "--answer-dir", type=str, default="", help="path to answer data"
    )
    parser.add_argument(
        "--glove-dir", type=str, default="", help="path to GloVe data"
    )
    parser.add_argument(
        "--embd-dir", type=str, default="", help="output directory of the embedding"
    )
    parser.add_argument(
        "--output-dir", type=str, default="", help="output directory"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="evaluation only"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)

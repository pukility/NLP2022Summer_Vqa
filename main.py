from mindspore import context

from .engine import Trainer
from .model import vqa_model


def main(args):
    cfg = {}
    
    model = vqa_model(cfg)
    ################
    #这一行是loader#
    ################
    trainer = Trainer(model, cfg, _, _, _)

    if args.eval_only:
        ##trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    """
    然后是巴拉巴拉一大堆parse
    """
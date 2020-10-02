import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from models.gpt2_pl import GPT2

if __name__ == '__main__':

    # arguments
    parser = ArgumentParser()

    parser.add_argument('--lr_range_test', type=bool, default=False)
    parser.add_argument('--fine_tune_on_petases', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=777)

    parser.add_argument('--min_delta', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=10)

    parser = GPT2.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # seed everything
    if args.random_seed is not None:
        pl.trainer.seed_everything(args.random_seed)
        args.deterministic = True

    # make save directory
    os.makedirs(args.default_root_dir, exist_ok=True)

    logger = pl_loggers.TensorBoardLogger(save_dir=args.default_root_dir)
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=args.min_delta,
                                                patience=args.patience,
                                                verbose=True,
                                                mode='min')
    ckpt_path = args.default_root_dir + "{epoch}-{val_loss:.4f}"
    ckpting = pl.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                           monitor='val_loss',
                                           verbose=True,
                                           save_top_k=3,
                                           mode='min')

    trainer = pl.Trainer.from_argparse_args(
        args,
        checkpoint_callback=ckpting,
        early_stop_callback=early_stopping,
    )
    model = GPT2(args)

    if args.lr_range_test:
        # when running this, I removed the scheduler from the GPT2 code,
        # although I'm not sure if it matters...

        lr_finder = trainer.lr_find(model)
        fig = lr_finder.plot(suggest=True)

        save_path = os.path.join(args.default_root_dir, 'lr-range-test')
        fig.savefig(save_path, format='png')
    else:
        trainer.fit(model)

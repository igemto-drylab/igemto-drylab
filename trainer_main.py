import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from track1.gpt2_pl import GPT2

if __name__ == '__main__':

    # path
    curr_dir = os.path.dirname(__file__)

    parser = ArgumentParser()

    default_data_path = os.path.join(
        curr_dir, 'track1/dataset/ec_3_1_1_seqs_cleaned.txt'
    )

    parser.add_argument('--data_path', type=str, default=default_data_path)
    parser.add_argument('--session_name', type=str, default='gpt2')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--min_delta', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--auto_lr', type=bool, default=False)

    parser = GPT2.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    dir_name = input("Save Dir Name?")
    save_dir = os.path.join(curr_dir, f"results/{input()}/")

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    logger = pl_loggers.TensorBoardLogger(save_dir=save_dir,
                                          name=args.session_name)

    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=args.min_delta,
                                                patience=args.patience,
                                                verbose=True,
                                                mode='min')

    checkpoint_path = save_dir + "{epoch}-{val_loss:.2f}"
    checkpointing = pl.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 monitor='val_loss',
                                                 verbose=True,
                                                 save_top_k=3,
                                                 mode='min')

    trainer = pl.Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpointing,
        early_stop_callback=early_stopping,
        default_root_dir=save_dir,
        gradient_clip_val=args.grad_clip,
        progress_bar_refresh_rate=50,
        auto_lr_find=args.auto_lr,
        deterministic=True
    )
    model = GPT2(args)
    trainer.fit(model)

from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import transformers
from pytorch_lightning.core.decorators import auto_move_data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR

from track1.seq_dataset import AA_TO_IDX, SeqDataset, pad_batch


class GPT2(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model arguments
        parser.add_argument('--n_embd', type=int, default=20)
        parser.add_argument('--n_layer', type=int, default=4)
        parser.add_argument('--n_head', type=int, default=4)

        # training arguments
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--base_lr', type=float, default=0.0001)
        parser.add_argument('--max_lr', type=float, default=0.0001)
        parser.add_argument('--step_size_up', type=int, default=3000)

        return parser

    def __init__(self, args):
        super().__init__()

        if isinstance(args, dict):
            args = Namespace(**args)

        self.hparams = args
        self.save_hyperparameters(args)

        self.lr = self.hparams.lr

        self.dataset = SeqDataset(args.data_path)
        self.train_dataset = None
        self.valid_dataset = None

        config = transformers.GPT2Config(
            vocab_size=len(AA_TO_IDX),
            n_positions=self.dataset.max_len,
            n_ctx=self.dataset.max_len,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
        )
        self.gpt2 = transformers.GPT2LMHeadModel(config=config)

    @auto_move_data
    def forward(self, x):
        return self.gpt2(input_ids=x, labels=x)

    def setup(self, stage):
        train_size = int(self.hparams.split_ratio[0] * len(self.dataset))
        valid_size = len(self.dataset) - train_size

        self.train_dataset, self.valid_dataset = \
            torch.utils.data.random_split(self.dataset,
                                          [train_size, valid_size])

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            collate_fn=pad_batch)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.valid_dataset,
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            collate_fn=pad_batch)
        return loader

    def training_step(self, batch, batch_idx):
        loss = self(batch)[0]

        logger_logs = {
            'train_loss': loss
        }
        return {'loss': loss, 'log': logger_logs}

    def validation_step(self, batch, batch_idx):
        loss = self(batch)[0]

        logger_logs = {
            'val_loss': loss,
        }
        return {'val_loss': loss, 'log': logger_logs}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        logger_logs = mean_of_logs(outputs)

        return {'val_loss': val_loss_mean, 'log': logger_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = CyclicLR(optimizer=optimizer,
                             base_lr=self.hparams.base_lr,
                             max_lr=self.hparams.max_lr,
                             step_size_up=self.hparams.step_size_up,
                             cycle_momentum=False)
        return [optimizer], [scheduler]

    def generate(self, **kwargs):
        return self.gpt2.generate(**kwargs)


# Helper Methods

def mean_of_logs(outputs, log_key='log'):
    logs = [x[log_key] for x in outputs]

    log_means = {}
    for k in logs[0].keys():
        log_means[k] = torch.stack([x[k] for x in logs]).mean()
    return log_means

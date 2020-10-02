import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import transformers
from pytorch_lightning.core.decorators import auto_move_data
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader

from utils.seq_dataset import AA_TO_IDX, SeqDataset, pad_batch


class GPT2(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model arguments
        parser.add_argument('--seq_len', type=int, default=(310 + 2))
        parser.add_argument('--n_embd', type=int, default=198)
        parser.add_argument('--n_head', type=int, default=6)
        parser.add_argument('--n_layer', type=int, default=5)
        parser.add_argument('--dropout', type=float, default=0.2368447359)

        # training arguments
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--lr', type=float, default=0.0014804104)
        parser.add_argument('--base_lr', type=float, default=0.00003162277)
        parser.add_argument('--max_lr', type=float, default=0.0022894)

        return parser

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.hparams.aa_to_idx = AA_TO_IDX

        config = transformers.GPT2Config(
            vocab_size=len(AA_TO_IDX),
            n_positions=self.hparams.seq_len,
            n_ctx=self.hparams.seq_len,
            n_embd=self.hparams.n_embd,
            n_layer=self.hparams.n_layer,
            n_head=self.hparams.n_head,
            resid_pdrop=self.hparams.dropout,
            embd_pdrop=self.hparams.dropout,
            attn_pdrop=self.hparams.dropout
        )
        self.gpt2 = transformers.GPT2LMHeadModel(config=config)

        # variables that will be defined later
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @auto_move_data
    def forward(self, x):
        return self.gpt2(input_ids=x, labels=x)

    def setup(self, stage):

        # get path of dataset directory
        cwd = os.path.abspath(os.path.dirname(__file__))
        dataset_dir = os.path.abspath(
            os.path.join(cwd, "../../datasets/generative")
        )

        paths = {'train': [], 'val': [], 'test': []}
        for k, v in paths.items():
            v.append(f"{dataset_dir}/petases_cleaned_{k}.txt")

        if not self.hparams.fine_tune_on_petases:
            for k, v in paths.items():
                v.append(f"{dataset_dir}/ec_3_1_1_cleaned_{k}.txt")

        self.hparams.data_paths = paths
        self.train_dataset = SeqDataset(paths['train'])
        self.val_dataset = SeqDataset(paths['val'])
        self.test_dataset = SeqDataset(paths['test'])

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            collate_fn=pad_batch,
                            shuffle=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset,
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            collate_fn=pad_batch)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset,
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

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        logger_logs = mean_of_logs(outputs)

        tqdm_dict = {'val_loss': val_loss_mean}

        results = {
            'progress_bar': tqdm_dict,
            'log': logger_logs
        }
        return results

    def test_step(self, batch, batch_idx):
        loss = self(batch)[0]

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        logger_logs = mean_of_logs(outputs)

        tqdm_dict = {'test_loss': test_loss_mean}

        results = {
            'progress_bar': tqdm_dict,
            'log': logger_logs
        }
        return results

    def configure_optimizers(self):
        step_size = 3 * len(self.train_dataset) / self.hparams.batch_size
        self.hparams.step_size_up = step_size

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
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

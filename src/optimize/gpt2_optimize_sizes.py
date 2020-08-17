import os

import optuna
import pytorch_lightning as pl
import torch
import transformers
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.core.decorators import auto_move_data
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

SEED = 777
pl.seed_everything(SEED)

EPOCHS = 40
SEQ_LEN = 310 + 2
BATCH_SIZE = 64
LIMIT_TEST_BATCHES = 1
LIMIT_VAL_BATCHES = 1

NUM_WORKERS = 0

VOCAB = ['<pad>', '<s>', '</s>',
         'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
AA_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_AA = {i: c for c, i in AA_TO_IDX.items()}


class SeqDataset(Dataset):

    def __init__(self, file_paths):

        self.sequences = []
        for path in file_paths:

            with open(path, 'r') as seq_file:
                contents = seq_file.readlines()
                self.sequences.extend(list(map(int_encode_and_process,
                                               contents)))
        self.sequences.sort(key=len)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        return self.sequences[item]


# Helpers

def int_encode_and_process(seq):
    seq = seq.rstrip()

    int_encoded = [AA_TO_IDX[char] for char in seq]
    int_encoded.insert(AA_TO_IDX['<s>'], 0)
    int_encoded.append(AA_TO_IDX['</s>'])
    return torch.tensor(int_encoded)


def pad_batch(batch):
    return pad_sequence(batch, True, AA_TO_IDX['<pad>'])


class GPT2(pl.LightningModule):

    def __init__(self, trial):
        super().__init__()

        n_embd = trial.suggest_int('n_embd', 100, 240)
        n_head = trial.suggest_int('n_head', 3, 10)
        n_layer = trial.suggest_int('n_layer', 3, 8)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        self.lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)

        # make n_embd divisible
        n_embd = n_head * round(n_embd / n_head)

        config = transformers.GPT2Config(
            vocab_size=len(AA_TO_IDX),
            n_positions=SEQ_LEN,
            n_ctx=SEQ_LEN,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout
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
        cwd = os.path.abspath(os.getcwd())
        dataset_dir = os.path.abspath(
            os.path.join(cwd, "../../datasets/ec_3_1_1_and_petases")
        )

        paths = {'train': [], 'val': [], 'test': []}

        for k, v in paths.items():
            v.append(f"{dataset_dir}/petases_cleaned_{k}.txt")

        self.train_dataset = SeqDataset(paths['train'])
        self.val_dataset = SeqDataset(paths['val'])
        self.test_dataset = SeqDataset(paths['test'])

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            collate_fn=pad_batch,
                            shuffle=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# Helper Methods

def mean_of_logs(outputs, log_key='log'):
    logs = [x[log_key] for x in outputs]

    log_means = {}
    for k in logs[0].keys():
        log_means[k] = torch.stack([x[k] for x in logs]).mean()
    return log_means


class MetricsCallback(pl.Callback):

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def objective(trial):
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=0.0001,
                                                patience=4,
                                                verbose=True,
                                                mode='min')
    prune_callback = PyTorchLightningPruningCallback(trial,
                                                     monitor="val_loss")
    metrics_callback = MetricsCallback()

    # noinspection PyTypeChecker
    trainer = pl.Trainer(
        logger=False,
        max_epochs=EPOCHS,
        early_stop_callback=early_stopping,
        checkpoint_callback=False,
        callbacks=[prune_callback, metrics_callback],
        limit_test_batches=LIMIT_TEST_BATCHES,
        limit_val_batches=LIMIT_VAL_BATCHES,
        gradient_clip_val=1.0,
        gpus=(1 if torch.cuda.is_available() else None),
        deterministic=True  # forgot to set flag in uploaded run (sorry)
    )

    model = GPT2(trial)
    trainer.fit(model)

    return metrics_callback.metrics[-1]["val_loss"].item()


if __name__ == '__main__':

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.PercentilePruner(percentile=75,
                                             n_startup_trials=10,
                                             n_warmup_steps=20,
                                             interval_steps=5)

    study = optuna.create_study(storage='sqlite:///gpt2_optimize_sizes.db',
                                sampler=sampler,
                                pruner=pruner,
                                study_name='gpt2_optimize_sizes',
                                direction='minimize',
                                load_if_exists=True)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(df)

    # study.enqueue_trial(  # best trial found in a previous buggy run
    #      {'n_embd': 232, 'n_head': 6, 'n_layer': 6,
    #      'dropout': 0.21649723315268476,
    #      'lr': 0.000411457312871047}
    # )

    # study.optimize(objective,
    #                n_trials=200,
    #                timeout=86400,
    #                catch=(RuntimeError,))

    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    best_trial = study.best_trial

    print(f"  Value: {best_trial.value}")

    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

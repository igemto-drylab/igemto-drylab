import math
import os

import optuna
import pytorch_lightning as pl
import torch
import transformers
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.core.decorators import auto_move_data
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR
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

        n_embd = 196
        n_head = 6
        n_layer = 5
        dropout = 0.23684473591088903

        self.trial = trial

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
            os.path.join(cwd, "datasets/ec_3_1_1_and_petases")
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

        trial = self.trial

        scheduler_type = trial.suggest_categorical(
            'scheduler_type',
            ['None', 'CyclicLR', 'OneCycleLR']
        )
        optimizer_type = trial.suggest_categorical(
            'optimizer_type',
            ['SGD', 'Adam', 'AdamW']
        )

        if scheduler_type == 'None':
            lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
            max_lr = 0.0014804104114356487  # dummy number
        else:
            lr = 0.0014804104114356487  # dummy number
            max_lr = trial.suggest_loguniform('max_lr', 1e-3, 1e-1)

        use_weight_decay = trial.suggest_categorical('use_weight_decay',
                                                     [True, False])
        if use_weight_decay:
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
        else:
            weight_decay = 0

        if optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=lr,
                                         weight_decay=weight_decay)
        elif optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=lr,
                                          weight_decay=weight_decay)
        else:
            momentum = trial.suggest_float('momentum', 0.0, 1.0)

            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=lr,
                                        momentum=momentum,
                                        weight_decay=weight_decay)

        if scheduler_type == 'CyclicLR':
            scheduler = CyclicLR(optimizer,
                                 base_lr=0.00003162277,
                                 max_lr=max_lr,
                                 step_size_up=(3 * round(761 / BATCH_SIZE)),
                                 cycle_momentum=(optimizer_type == 'SGD'))
            return [optimizer], [scheduler]

        elif scheduler_type == 'OneCycleLR':
            scheduler = OneCycleLR(optimizer,
                                   max_lr=max_lr,
                                   epochs=EPOCHS,
                                   steps_per_epoch=math.ceil(761 / BATCH_SIZE),
                                   cycle_momentum=(optimizer_type == 'SGD'))
            return [optimizer], [scheduler]

        else:
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
                                                patience=6,
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
        deterministic=True
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

    study = optuna.create_study(storage='sqlite:///gpt2_study_lr.db',
                                sampler=sampler,
                                pruner=pruner,
                                study_name='gpt2_optimize_lr',
                                direction='minimize',
                                load_if_exists=True)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(df)

    # Interested in trying the two below
    # study.enqueue_trial({
    #     'scheduler_type': 'CyclicLR',
    #     'optimizer_type': 'Adam',
    #     'use_weight_decay': False,
    #     'max_lr': 0.031622
    # })
    # study.enqueue_trial({
    #     'scheduler_type': 'OneCycleLR',
    #     'optimizer_type': 'Adam',
    #     'use_weight_decay': False,
    #     'max_lr': 0.031622
    # })
    #
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

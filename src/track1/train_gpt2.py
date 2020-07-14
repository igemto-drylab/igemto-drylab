import torch.utils.data
import transformers
from torch.utils.tensorboard import SummaryWriter

from src.track1.seq_dataset import get_dataset_and_collater

print("Loading Data...")

DATA_PATH = "dataset/ec_3_1_1_seqs_cleaned.txt"
VOCAB_PATH = "dataset/prot-bpe-vocab.json"
MERGES_PATH = "dataset/prot-bpe-merges.txt"

seq_len = 0
with open(DATA_PATH, 'r') as data_file:
    for row in data_file.readlines():
        seq_len = max(seq_len, len(row))
seq_len += 2  # add 2 to account for sos and eos tokens?

dataset, data_collator = get_dataset_and_collater(vocab_path=VOCAB_PATH,
                                                  merges_path=MERGES_PATH,
                                                  data_path=DATA_PATH,
                                                  seq_len=seq_len)

split_ratio = [0.9, 0.1]
split_lens = [int(len(dataset) * split_ratio[0]), None]
split_lens[1] = len(dataset) - split_lens[0]

train_set, valid_set = torch.utils.data.random_split(dataset, split_lens)

print("Loading Model...")

config = transformers.GPT2Config(
    vocab_size=261,
    n_positions=seq_len,
    n_ctx=seq_len,
    n_embd=30,
    n_layer=3,
    n_head=3
)

model = transformers.GPT2LMHeadModel(config=config)

print("Training Model...")

writer = SummaryWriter()

training_args = transformers.TrainingArguments(
    output_dir="models/gpt2/",
    do_train=True,
    do_eval=True,
    do_predict=True,
    evaluate_during_training=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    logging_first_step=True,
    save_steps=2000,
    save_total_limit=2,
)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_set,
    eval_dataset=valid_set,
    tb_writer=writer
)

trainer.train()

# Save Model
trainer.save_model("models/gpt2/")

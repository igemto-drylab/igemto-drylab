from src.generative.gpt2_pl import GPT2
import torch

from src.utils.seq_dataset import AA_TO_IDX, IDX_TO_AA

PATH = "../results/gpt2-finetune/epoch=58-val_loss=0.6654.ckpt"

gpt2 = GPT2.load_from_checkpoint(PATH)

N = 10

for _ in range(N):
    input_ids = torch.tensor([AA_TO_IDX['<s>'], AA_TO_IDX['M']]).long()
    input_ids = input_ids.unsqueeze(0)

    output = gpt2.generate(
        input_ids=input_ids,
        min_length=260,
        max_length=310,
        do_samples=True,
        early_stopping=True,
        num_beams=5,
        pad_token_id=AA_TO_IDX['<pad>'],
        bos_token_id=AA_TO_IDX['<s>'],
        eos_token_id=AA_TO_IDX['</s>'],
    )
    output = output.squeeze(0)

    protein_seq = ""
    for i in output:
        aa = IDX_TO_AA[i.item()]
        if aa in {'<s>', '<pad>', '</s>'}:
            continue
        protein_seq += aa

    print(protein_seq)
    print(len(protein_seq))
    print()

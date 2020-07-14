"""Script for building vocab.json and merges.txt
"""

from tokenizers import ByteLevelBPETokenizer

paths = ["../../datasets/ec_3_1_1_data/amino_acids.txt"]

tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
tokenizer.train(
    files=paths,
    min_frequency=2,
    special_tokens=['<s>', '</s>', '<unk>', '<pad>']
)

tokenizer.save_model(".", "prot-bpe")

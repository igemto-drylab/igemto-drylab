import transformers


def get_seq_tokenizer(vocab_path, merges_path):
    tokenizer = transformers.GPT2TokenizerFast(
        vocab_file=vocab_path,
        merges_file=merges_path,
        unk_token='<unk>',
        bos_token='<s>',
        eos_token='</s>',
        pad_token='<pad>',
        padding_side='right',
        add_prefix_space=True
    )
    return tokenizer


def get_dataset_and_collater(vocab_path, merges_path, data_path, seq_len):
    tokenizer = get_seq_tokenizer(vocab_path, merges_path)

    dataset = transformers.LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=data_path,
        block_size=seq_len
    )

    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    return dataset, data_collator

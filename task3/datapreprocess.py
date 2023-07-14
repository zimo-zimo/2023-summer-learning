import torch
import torch.nn as nn
from torchtext.datasets import IMDB
from torchtext.data import Field, LabelField, BucketIterator


def dataPreprocess(batch_size):
    # declare field: data type
    tokenize = lambda x: x.split()
    text_field = Field(sequential=True, tokenize=tokenize, lower=True, batch_first = True)
    label_field = LabelField(dtype = torch.float, batch_first = True)

    # load dataset
    train_data, test_data = IMDB.splits(text_field, label_field, root='.data')
    print("train data length: ", len(train_data))
    print("test data length: ", len(test_data))

    # print(vars(train_data.examples[0]))

    # build vocabulary
    text_field.build_vocab(train_data, max_size=10000)
    label_field.build_vocab(train_data)

    # print(text_field.vocab.freqs.most_common(20))

    # build iterator: shuffle and padding 
    train_iter, test_iter = BucketIterator.splits(
        (train_data, test_data),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.text),
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )

    return train_iter, test_iter, len(text_field.vocab), len(train_data), len(test_data)
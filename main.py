from torchtext import data
from torchtext import datasets

TEXT = data.Field()
LABEL = data.Field()

fields = [(None, None), (None, None), ('l', LABEL), ('t', TEXT)]
train_data, valid_data, test_data = data.TabularDataset.splits(
                                        path = '',
                                        train = 'train.tsv',
                                        validation = 'valid.tsv',
                                        test = 'test.tsv',
                                        format = 'tsv',
                                        fields = fields,
                                        skip_header = False
)
print(vars(train_data[0]))

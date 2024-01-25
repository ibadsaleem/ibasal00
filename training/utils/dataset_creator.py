from torch.utils.data import Dataset, DataLoader
class DatasetCreator(Dataset):
    def __init__(self, processed_data, train):
        self.data = processed_data
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data.iloc[index]
        if self.train:
            return {'text': line['text'], 'label': line['labels']}
        else:
            return {'text': line['text'], 'label': 0}

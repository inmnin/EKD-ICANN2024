import tqdm
from torch.utils.data import Dataset
import string
import torch
from transformers import BertTokenizer
from time import sleep
class Normal_dataset(Dataset):
    def __init__(self, corpus_path, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        self.contents = []
        self.labels = []

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.on_memory == True:
                self.lines = [line[:-1]
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                for line in self.lines:
                    content,label = self.split_at_last_colon(line)
                    if label == '' or content == '':
                        continue

                    self.contents.append(content)
                   
                    if(str(label) != '0' and str(label) != '-1' and str(label) != '1'):
                        continue
                    #There may be cases where strings cannot be converted to integers
                    self.labels.append(label) 

    def __getitem__(self, index):
        content = self.contents[index]
        label = self.labels[index]

        # Convert content to a sequence of token indices
        content = content.split()
        return content, label

    @staticmethod
    def split_at_last_colon(s):
        idx = s.rfind(':')
        if idx == -1:  # no comma found
            return s, ''
        return s[:idx], s[idx + 1:]

    def remove_punctuation(self,s):
        """
        Remove all punctuation from a string.
        """
        return ''.join(ch for ch in s if ch not in string.punctuation)

        return idxs

    def __len__(self):
        return len(self.contents)


#dataset of the BERT
class Bert_Dataset(Normal_dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        super().__init__(corpus_path, seq_len, encoding, corpus_lines, on_memory)
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        content = self.contents[index]
        label = int(self.labels[index])

        # Encoding content using a tokenizer
        inputs = self.tokenizer(content, return_tensors="pt", padding='max_length', truncation=True,
                                max_length=self.seq_len)

        inputs['original_text'] = self.contents[index]
        # Remove unnecessary batch dimensions
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):  # Check for tensor
                inputs[key] = value.squeeze(0)

        inputs['labels'] = torch.tensor(label)
        return inputs



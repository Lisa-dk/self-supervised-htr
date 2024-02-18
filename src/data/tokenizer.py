import unicodedata
from itertools import groupby
import numpy as np

class Tokenizer():
    """Manager tokens functions and charset/dictionary properties"""

    def __init__(self, chars, max_text_length=10):
        self.PAD_TK, self.UNK_TK = "¶", "¤"
        self.GO_TK, self.END_TK = "♂", "♀"
        self.chars = (self.GO_TK + self.GO_TK + self.PAD_TK + self.UNK_TK + chars)
        # print(self.chars)

        self.PAD = self.chars.find(self.PAD_TK)
        self.UNK = self.chars.find(self.UNK_TK)
        self.GO = self.chars.find(self.GO_TK)
        self.END = self.chars.find(self.END_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length + 2

    def encode(self, text):
        """Encode text to vector"""

        if isinstance(text, bytes):
            text = text.decode()

        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
        text = " ".join(text.split())

        # groups = ["".join(group) for _, group in groupby(text)]
        # text = "".join([self.UNK_TK.join(list(x)) if len(x) > 1 else x for x in groups])
        text = self.GO_TK + text + self.GO_TK
        encoded = []

        for item in text:
            index = self.chars.find(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        encoded = np.pad(encoded, (0, self.maxlen - len(encoded)), mode='constant', constant_values=self.PAD)

        return np.asarray(encoded).tolist()

    def decode(self, text):
        """Decode vector to text"""

        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        decoded = self.remove_tokens(decoded)

        return decoded


    def remove_tokens(self, text):
        """Remove tokens (PAD) from text"""

        return text.replace(self.PAD_TK, "").replace(self.UNK_TK, "").replace(self.GO_TK, "").replace(self.END_TK, "")

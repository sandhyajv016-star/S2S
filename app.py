import torch # [cite: 28]
import torch.nn as nn # [cite: 29]
import random # [cite: 30]

# 1. Dataset Definition
eng_sentences = [ # [cite: 31]
    ["i", "love", "india"], # [cite: 32]
    ["you", "like", "music","very","much"] # [cite: 33]
] # [cite: 34]

fr_sentences = [ # [cite: 35]
    ["", "je", "t'aime", "l'inde", ""], # [cite: 36]
    ["", "tu", "aimes", "la", "musique", ""] # [cite: 37]
] # [cite: 38]

# 2. Vocabulary Initialization
eng_vocab = {"": 0} # [cite: 39]
fr_vocab = {"": 0, "": 1} # [cite: 45]

# 3. Building the Vocabularies
for sent in eng_sentences: # [cite: 42]
    for w in sent: # [cite: 43]
        if w not in eng_vocab: eng_vocab[w] = len(eng_vocab) # [cite: 44]

for sent in fr_sentences: # [cite: 46]
    for w in sent: # [cite: 49]
        if w not in fr_vocab: fr_vocab[w] = len(fr_vocab) # [cite: 50]

# Create a reverse dictionary for the French vocabulary
idx2fr = {v:k for k,v in fr_vocab.items()} # [cite: 51]

# Special Tokens
SOS = fr_vocab[""] # [cite: 69]
EOS = fr_vocab[""] # [cite: 70]

# 4. Encoding Function
def encode(words, vocab): # [cite: 72]
    idxs = [vocab[w] for w in words] # [cite: 73]
    return torch.tensor(idxs).unsqueeze(1) # [cite: 74]

# Encoding the source and target sentences
src1 = encode(eng_sentences[0], eng_vocab) # [cite: 77]
src2 = encode(eng_sentences[1], eng_vocab) # [cite: 78]
trg1 = encode(fr_sentences[0], fr_vocab) # [cite: 79]
trg2 = encode(fr_sentences[1], fr_vocab) # [cite: 80]

# 5. Encoder Architecture
class Encoder(nn.Module): # [cite: 81]
    def __init__(self, vocab, emb, hid): # [cite: 82]
        super().__init__() # [cite: 83]
        self.embed = nn.Embedding(len(vocab), emb) # [cite: 84]
        self.rnn = nn.LSTM(emb, hid) # [cite: 85]
        
    def forward(self, src): # [cite: 86]
        emb = self.embed(src) # [T,1,E] # [cite: 87]
        outputs, hidden = self.rnn(emb) # [cite: 88]
        return hidden # [cite: 89]

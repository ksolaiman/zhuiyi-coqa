## Sourced from: https://github.com/pytorch/fairseq/tree/master/examples/roberta

import torch
# there are 4 pretrained models - base, large, large-mnli, large-wsc
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()  # disable dropout (or leave in train mode to finetune)

tokens = roberta.encode('Hello world!')#, 'test dual sentence input')
# assert tokens.tolist() == [0, 31414, 232, 328, 2]
print(roberta.decode(tokens))  # 'Hello world!'

print(tokens)
#print(tokens.offsets)

tokens = roberta.encode('Hello', 'world!')
print(roberta.decode(tokens)) 
print(tokens)

tokens = roberta.encode('Hello world! My name is Salvi. hi!')
print(roberta.decode(tokens)) 
print(tokens)

tokens = roberta.encode('Hello', 'world!','My', 'name', 'is', 'Salvi.', 'hi!')
print(roberta.decode(tokens)) 
print(tokens)

# https://pytorch.org/hub/pytorch_fairseq_translation/
# Roberta base from hub does not have tokenize or apply bpe or binarize :(

# Extract the last layer's features
last_layer_features = roberta.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 5, 1024])

# Extract all layer's features (layer 0 is the embedding layer)
all_layers = roberta.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 25
assert torch.all(all_layers[-1] == last_layer_features)

print(roberta)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import Tuple, List
from collections import Counter, defaultdict

def align_features_to_words(roberta, features, alignment, diff_score):
    """
    Align given features to words.
    Args:
        roberta (RobertaHubInterface): RoBERTa instance
        features (torch.Tensor): features to align of shape `(T_bpe x C)`
        alignment: alignment between BPE tokens and words returned by
            func:`align_bpe_to_words`.
    """
    assert features.dim() == 2

    bpe_counts = Counter(j for bpe_indices in alignment for j in bpe_indices)
    assert bpe_counts[0] == 0  # <s> shouldn't be aligned
    denom = features.new([bpe_counts.get(j, 1) for j in range(len(features))])
    weighted_features = features / denom.unsqueeze(-1)

    output = [weighted_features[0]]
    largest_j = -1
    for bpe_indices in alignment:
        output.append(weighted_features[bpe_indices].sum(dim=0))
        largest_j = max(largest_j, *bpe_indices)
    for j in range(largest_j + 1, len(features)):
        output.append(weighted_features[j])
    output = torch.stack(output)
    assert torch.all(torch.abs(output.sum(dim=0) - features.sum(dim=0)) < diff_score)
    return output

def add_SEP_token(sep, tokens: List[str], tokens2=None, single=True):
    full_tokens = list()
    full_tokens.append("<s>")
    full_tokens.extend(tokens)
    full_tokens.append("</s>")
    if not single:
        full_tokens.append("</s>")# if not no_separator else ""
        full_tokens.extend(tokens2)
        full_tokens.append("</s>")
    return full_tokens

def extract_aligned_roberta_multiple(roberta, sentence: str, sentence2,
                            tokens: List[str], tokens2,
                            return_all_hiddens=False):
    
    from fairseq.models.roberta import alignment_utils
    full_tokens = add_SEP_token("<s>", tokens, tokens2, single=False)
    
    # tokenize both with GPT-2 BPE and get alignment with given tokens
    bpe_toks = roberta.encode(sentence, sentence2)
    alignment = alignment_utils.align_bpe_to_words(roberta, bpe_toks, full_tokens) # tokens came from spacy, for this func. from golden tokens
    features = roberta.extract_features(bpe_toks, return_all_hiddens=return_all_hiddens)
    sent_features = features
    if return_all_hiddens:
        features = features[-1].squeeze(0)   #Batch-size = 1 #-1 is the last layer, right?
    else:
        features = features.squeeze(0)   #Batch-size = 1           
    # aligned_feats = alignment_utils.align_features_to_words(roberta, features, alignment)
    aligned_feats = align_features_to_words(roberta, features, alignment, 1e-3)
   
    return aligned_feats[1:-1], sent_features  #exclude <s> and </s> tokens

def extract_aligned_roberta(roberta, sentence: str, 
                            tokens: List[str], 
                            return_all_hiddens=False):
    ''' Code inspired from: 
       https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py
    
    Aligns roberta embeddings for an input tokenization of words for a sentence
    
    Inputs:
    1. roberta: roberta fairseq class
    2. sentence: sentence in string
    3. tokens: tokens of the sentence in which the alignment is to be done
    
    Outputs: Aligned roberta features 
    '''
    
    from fairseq.models.roberta import alignment_utils
    
    full_tokens = add_SEP_token("<s>", tokens, single=True)

    # tokenize both with GPT-2 BPE and get alignment with given tokens
    bpe_toks = roberta.encode(sentence)
    alignment = alignment_utils.align_bpe_to_words(roberta, bpe_toks, tokens) # tokens came from spacy, for this func. from golden tokens
    
    # extract features and align them
    # LM heads are only used when masked_tokens are involved, not in this case
    features = roberta.extract_features(bpe_toks, return_all_hiddens=return_all_hiddens)
    sent_features = features
    print("Feature size: " + str(len(sent_features)))  # 1 vs 25, when all layers are returned, when return all hidden is true just features are returned, one variable, with the hidden
    # features is the last layer, extra is all layers with hidden ones
    if return_all_hiddens:
        features = features[-1].squeeze(0)   #Batch-size = 1 #-1 is the last layer, right?
    else:
        features = features.squeeze(0)   #Batch-size = 1           
    # aligned_feats = alignment_utils.align_features_to_words(roberta, features, alignment)
    aligned_feats = align_features_to_words(roberta, features, alignment, 1e-3)
   
    return aligned_feats[1:-1], sent_features  #exclude <s> and </s> tokens

# Copied from transformers.modeling_bert.BertPooler
class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size']) # hidden_size = 1024
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        print("entered pooler...")
        print(hidden_states.shape)
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] # 16*364*1024 -> 16*1*1024
        print(first_token_tensor.shape)
        pooled_output = self.dense(first_token_tensor) 
        pooled_output = self.activation(pooled_output)
        print(pooled_output)
        pooled_output = pooled_output.unsqueeze(1)
        print(pooled_output)
        print(pooled_output.shape)
        print("leaving...")
        return pooled_output
    
class Model(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, config, train_set=None):
        # Book-keeping.
        self.config = config
        print(self.config)
        
        # there are 4 pretrained models - base, large, large-mnli, large-wsc
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli') #'roberta.large')
        # print(self.roberta)
        prev_num_classes = self.roberta.model.classification_heads.mnli.out_proj.out_features
        prev_inner_dim = self.roberta.model.classification_heads.mnli.dense.out_features
        print(prev_num_classes, prev_inner_dim)
        self.span_fc = nn.Linear(prev_inner_dim, 2)
        self.other_fc = nn.Linear(prev_inner_dim, 3) # FC layer for yes/no/unk, input: roberta_pooled_output_dimension, output: 3
        
        self.pooler = RobertaPooler(config) # if add_pooling_layer else None
        
        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        feature_extract = True
        self._init_optimizer()
        
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
            
    def _init_optimizer(self):
        parameters = [p for p in self.roberta.parameters() if p.requires_grad]
        # print(parameters)
        for name, param in self.roberta.named_parameters():
            if param.requires_grad:
                print(name) #, param.data
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.config['learning_rate'],
                                       momentum=self.config['momentum'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(parameters,
                                          weight_decay=self.config['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.config['optimizer'])
            
    def predict(self, ex, update=True, out_predictions=False):
        
        print([len(item['question']['word'][-512:]) for item in ex])
        print([len(item['evidence']['word'][0:512]) for item in ex])
        print([item['answers'] for item in ex])
        print([item['targets'] for item in ex])
        
        # Train/Eval mode
        if not update: # eval mode
            self.roberta.eval()
        # gittu how should ex look like
        from fairseq.data.data_utils import collate_tokens
        batch_of_pairs = [[" ".join(item['question']['word'][-512:]), " ".join(item['evidence']['word'][0:512])] for item in ex]
        # print(batch_of_pairs)
        batch = collate_tokens([self.roberta.encode(pair[0], pair[1]) for pair in batch_of_pairs], pad_idx=1)
        
        # self.roberta.cuda()
        extra = self.roberta.extract_features(batch, return_all_hiddens=True)
        # features, sent_features = extract_aligned_roberta_multiple(self.roberta, " ".join(ex[0]['question']['word'][0:512]), " ".join(ex[0]['evidence']['word'][0:512]), ex[0]['question']['word'][0:512], ex[0]['evidence']['word'][0:512], return_all_hiddens=True)
        #features, sent_features = extract_aligned_roberta(self.roberta, " ".join(ex[0]['question']['word'][0:512]), ex[0]['question']['word'][0:512], return_all_hiddens=True)
        sent_features = extra
        print(len(sent_features))
        # print(sent_features.size)
        print(sent_features[0].shape)
        print(sent_features[1].shape)
        print(sent_features[2].shape)
        print(sent_features[-1].shape)
        features = extra[-1]
        
        print(features[:, 0, :].shape)
        
        # self.roberta.register_classification_head('span', num_classes=2)
        # logits = self.roberta.model.classification_heads['span'](sent_features[-1])
        # print(logits)
        
        print("normal FC: ")
        logits = self.span_fc(features)
        print(logits.shape)
        '''
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        print(start_logits)
        print(end_logits)
        input("wait")
        '''
        """
        from transformers import BertForQuestionAnswering
        from transformers import BertTokenizer

        #Model
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        #Tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        
        question = '''Why was the student group called "the Methodists?"'''

        paragraph = ''' The movement which would become The United Methodist Church began in the mid-18th century within the Church of England.
                    A small group of students, including John Wesley, Charles Wesley and George Whitefield, met on the Oxford University campus.
                    They focused on Bible study, methodical study of scripture and living a holy life.
                    Other students mocked them, saying they were the "Holy Club" and "the Methodists", being methodical and exceptionally detailed in their Bible study, opinions and disciplined lifestyle.
                    Eventually, the so-called Methodists started individual societies or classes for members of the Church of England who wanted to live a more religious life. '''

        encoding = tokenizer.encode_plus(text=question,text_pair=paragraph,add_special_tokens=True)

        inputs = encoding['input_ids']  #Token embeddings
        sentence_embedding = encoding['token_type_ids']  #Segment embeddings
        tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens
        
        print(tokens)
        print(len(tokens))
        # print(model)
        print(model.num_labels)
        ans = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))
        
        print(ans)
        print(ans['start_logits'].shape)
        print(ans['end_logits'].shape)
        
        input("wait")
        #"""
        ## try different pooling, right now self.pooler just takes the hidden state corresponding to first token <[CLS]>
        #self.roberta.register_classification_head('ynunk', num_classes=3)
        
        #logits_2 = self.roberta.model.classification_heads['ynunk'](self.pooler(sent_features[-1]))
        #print(logits_2)
        
        print("ynunk FC: ")
        logits_2 = self.other_fc(self.pooler(features))
        print(logits_2.shape)
        
        # pooled output/ yes/no/unk processing - 
        # 1. [16, 364, 2] - start and end for 364 tokens
        # torch.Size([16, 3]); batch size = 16, repeat for each token, and take the max
        # 2. in pooling function, do not take just the first token hidden state, do something else, or just pass through linear layer
        # trying 1. 
        print(features.size(1))
        logits_2 = logits_2.repeat(1, features.size(1), 1)
        print(logits_2.shape)
        
        indices = torch.tensor([0]) # torch.tensor([0])
        start_span = torch.index_select(logits, 2, indices) # torch.index_select(logits, 1, indices)
        print("start_span_size: ")
        print(start_span.shape)
        
        start_logits = torch.cat([start_span, 
                            torch.index_select(logits_2, 2, torch.tensor([0])), # dim 2 instead of 1
                            torch.index_select(logits_2, 2, torch.tensor([1])), # dim 2 instead of 1
                            torch.index_select(logits_2, 2, torch.tensor([2]))], 1) # dim 2 instead of 1
        
        indices = torch.tensor([1]) # torch.tensor([1])
        end_span = torch.index_select(logits, 2, indices) # torch.index_select(logits, 1, indices)
        
        print(start_logits)
        end_logits = torch.cat([end_span, 
                            torch.index_select(logits_2, 2, torch.tensor([0])), # dim 2 instead of 1
                            torch.index_select(logits_2, 2, torch.tensor([1])), # dim 2 instead of 1
                            torch.index_select(logits_2, 2, torch.tensor([2]))], 1) # dim 2 instead of 1
        print(end_logits)
        
        slog = F.log_softmax(start_logits, dim=-1)
        elog = F.log_softmax(end_logits, dim=-1)
        
        print(torch.argmax(slog, 1))
        print(torch.argmax(elog, 1))
        
        print(ex[0]['targets'])
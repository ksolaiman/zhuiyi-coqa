# -*- coding: utf-8 -*-
"""
Module to handle getting data loading classes and helper functions.
"""

import argparse
import json
import io
import torch
import numpy as np

from collections import Counter, defaultdict
from torch.utils.data import Dataset
import utils.constants as Constants
from utils.timer import Timer


################################################################################
# Dataset Prep #
################################################################################

def prepare_datasets(config):
    train_set = None if config['trainset'] is None else CoQADataset(config['trainset'], config)
    dev_set = None if config['devset'] is None else CoQADataset(config['devset'], config)
    test_set = None if config['testset'] is None else CoQADataset(config['testset'], config)
    return {'train': train_set, 'dev': dev_set, 'test': test_set}
    # return {'dev': dev_set}

################################################################################
# Dataset Classes #
################################################################################


class CoQADataset(Dataset):
    """SQuAD dataset."""

    def __init__(self, filename, config):
        timer = Timer('Load %s' % filename)
        self.filename = filename
        self.config = config
        paragraph_lens = []
        question_lens = []
        self.paragraphs = []
        self.examples = []
        self.vocab = Counter()
        dataset = read_json(filename)
        for paragraph in dataset['data']:
            history = []
            for qas in paragraph['qas']:
                qas['paragraph_id'] = len(self.paragraphs)
                temp = []
                non_annotated_temp = ""
                n_history = len(history) if config['n_history'] < 0 else min(config['n_history'], len(history))
                if n_history > 0:
                    for i, (q, a, naq, naa) in enumerate(history[-n_history:]): # say, n =10; starts from 10th item from last 
                        d = n_history - i
                        # temp.append('<Q{}>'.format(d))
                        temp.append('<Q>')
                        temp.extend(q)
                        #temp.append('<A{}>'.format(d))
                        temp.append('<A>')
                        temp.extend(a)
                        
                        
                        # non_annotated_temp.append('<Q{}>{}'.format(d, naq))
                        # non_annotated_temp.append(naq)
                        # non_annotated_temp.append('<A{}>{}'.format(d, naa))
                        # non_annotated_temp.append(naa)
                        non_annotated_temp += '<Q> {} <A> {} '.format(naq, naa)
                temp.append('<Q>')
                # non_annotated_temp.append('<Q>')
                temp.extend(qas['annotated_question']['word'])
                non_annotated_temp = non_annotated_temp + '<Q> ' + qas['question']
                history.append((qas['annotated_question']['word'], qas['annotated_answer']['word'], 
                               qas['question'], qas['answer']))
                qas['annotated_question']['word'] = temp               
                qas['question'] = non_annotated_temp
                self.examples.append(qas)
                question_lens.append(len(qas['annotated_question']['word']))
                paragraph_lens.append(len(paragraph['annotated_context']['word']))
                '''
                for w in qas['annotated_question']['word']:
                    self.vocab[w] += 1
                for w in paragraph['annotated_context']['word']:
                    self.vocab[w] += 1
                for w in qas['annotated_answer']['word']:
                    self.vocab[w] += 1
                '''
            self.paragraphs.append(paragraph)
        print('Load {} paragraphs, {} examples.'.format(len(self.paragraphs), len(self.examples)))
        print('Paragraph length: avg = %.1f, max = %d' % (np.average(paragraph_lens), np.max(paragraph_lens)))
        print('Question length: avg = %.1f, max = %d' % (np.average(question_lens), np.max(question_lens)))
        timer.finish()

    def __len__(self):
        return 50 if self.config['debug'] else len(self.examples)

    def __getitem__(self, idx):
        qas = self.examples[idx]
        paragraph = self.paragraphs[qas['paragraph_id']]
        annnotated_question = qas['annotated_question']
        answers = [qas['answer']]
        question_string = qas['question']
        if 'additional_answers' in qas:
            answers = answers + qas['additional_answers']

        sample = {'id': (paragraph['id'], qas['turn_id']),
                  'question': annnotated_question,
                  'answers': answers,
                  'evidence': paragraph['annotated_context'],
                  #'input': '[CLS]' + qas['question'] +'[SEP]' + paragraph['context'] + '[SEP]',
                  'question_string': " ".join(annnotated_question['word'][(-1)*self.config['question_token_length']:]), # limit on the number of question tokens, truncate from the previous questions
                  'targets': qas['answer_span']}

        if self.config['predict_raw_text']:
            sample['raw_evidence'] = paragraph['context']
        return sample

################################################################################
# Read & Write Helper Functions #
################################################################################


def write_json_to_file(json_object, json_file, mode='w', encoding='utf-8'):
    with io.open(json_file, mode, encoding=encoding) as outfile:
        json.dump(json_object, outfile, indent=4, sort_keys=True, ensure_ascii=False)


def log_json(data, filename, mode='w', encoding='utf-8'):
    with io.open(filename, mode, encoding=encoding) as outfile:
        outfile.write(json.dumps(data, indent=4, ensure_ascii=False))


def get_file_contents(filename, encoding='utf-8'):
    with io.open(filename, encoding=encoding) as f:
        content = f.read()
    f.close()
    return content


def read_json(filename, encoding='utf-8'):
    contents = get_file_contents(filename, encoding=encoding)
    return json.loads(contents)


def get_processed_file_contents(file_path, encoding="utf-8"):
    contents = get_file_contents(file_path, encoding=encoding)
    return contents.strip()

################################################################################
# DataLoader Helper Functions #
################################################################################


def sanitize_input(sample_batch, config, vocab, feature_dict, training=True):
    """
    Reformats sample_batch for easy vectorization.
    Args:
        sample_batch: the sampled batch, yet to be sanitized or vectorized.
        vocab: word embedding dictionary.
        feature_dict: the features we want to concatenate to our embeddings.
        train: train or test?
    """
    sanitized_batch = defaultdict(list)
    for ex in sample_batch:
        question = ex['question']['word']
        evidence = ex['evidence']['word']
        offsets = ex['evidence']['offsets']

        processed_q, processed_e = [], []
        for w in question:
            processed_q.append(vocab[w] if w in vocab else vocab[Constants._UNK_TOKEN])
        for w in evidence:
            processed_e.append(vocab[w] if w in vocab else vocab[Constants._UNK_TOKEN])

        # Append relevant index-structures to batch
        sanitized_batch['question'].append(processed_q)
        sanitized_batch['evidence'].append(processed_e)

        if config['predict_raw_text']:
            sanitized_batch['raw_evidence_text'].append(ex['raw_evidence'])
            sanitized_batch['offsets'].append(offsets)
        else:
            sanitized_batch['evidence_text'].append(evidence)

        # featurize evidence document:
        sanitized_batch['features'].append(featurize(ex['question'], ex['evidence'], feature_dict))
        sanitized_batch['targets'].append(ex['targets'])
        sanitized_batch['answers'].append(ex['answers'])
        if 'id' in ex:
            sanitized_batch['id'].append(ex['id'])
    return sanitized_batch


def vectorize_input(batch, config, training=True, device=None):
    """
    - Vectorize question and question mask
    - Vectorize evidence documents, mask and features
    - Vectorize target representations
    """
    # Check there is at least one valid example in batch (containing targets):
    if not batch:
        return None

    # Relevant parameters:
    batch_size = len(batch['question'])

    # Initialize all relevant parameters to None:
    targets = None

    # Part 1: Question Words
    # Batch questions ( sum_bs(n_sect), len_q)
    max_q_len = max([len(q) for q in batch['question']])
    xq = torch.LongTensor(batch_size, max_q_len).fill_(0)
    xq_mask = torch.ByteTensor(batch_size, max_q_len).fill_(1)
    for i, q in enumerate(batch['question']):
        xq[i, :len(q)].copy_(torch.LongTensor(q))
        xq_mask[i, :len(q)].fill_(0)

    # Part 2: Document Words
    max_d_len = max([len(d) for d in batch['evidence']])
    xd = torch.LongTensor(batch_size, max_d_len).fill_(0)
    xd_mask = torch.ByteTensor(batch_size, max_d_len).fill_(1)
    xd_f = torch.zeros(batch_size, max_d_len, config['num_features']) if config['num_features'] > 0 else None

    # 2(a): fill up DrQA section variables
    for i, d in enumerate(batch['evidence']):
        xd[i, :len(d)].copy_(torch.LongTensor(d))
        xd_mask[i, :len(d)].fill_(0)
        if config['num_features'] > 0:
            xd_f[i, :len(d)].copy_(batch['features'][i])

    # Part 3: Target representations
    if config['sum_loss']:  # For sum_loss "targets" acts as a mask rather than indices.
        targets = torch.ByteTensor(batch_size, max_d_len, 2).fill_(0)
        for i, _targets in enumerate(batch['targets']):
            for s, e in _targets:
                targets[i, s, 0] = 1
                targets[i, e, 1] = 1
    else:
        targets = torch.LongTensor(batch_size, 2)
        for i, _target in enumerate(batch['targets']):
            targets[i][0] = _target[0]
            targets[i][1] = _target[1]

    torch.set_grad_enabled(training)
    example = {'batch_size': batch_size,
               'answers': batch['answers'],
               'xq': xq.to(device) if device else xq,
               'xq_mask': xq_mask.to(device) if device else xq_mask,
               'xd': xd.to(device) if device else xd,
               'xd_mask': xd_mask.to(device) if device else xd_mask,
               'xd_f': xd_f.to(device) if device else xd_f,
               'targets': targets.to(device) if device else targets}

    if config['predict_raw_text']:
        example['raw_evidence_text'] = batch['raw_evidence_text']
        example['offsets'] = batch['offsets']
    else:
        example['evidence_text'] = batch['evidence_text']
    return example


def featurize(question, document, feature_dict):
    doc_len = len(document['word'])
    features = torch.zeros(doc_len, len(feature_dict))
    q_cased_words = set([w for w in question['word']])
    q_uncased_words = set([w.lower() for w in question['word']])
    for i in range(doc_len):
        d_word = document['word'][i]
        if 'f_qem_cased' in feature_dict and d_word in q_cased_words:
            features[i][feature_dict['f_qem_cased']] = 1.0
        if 'f_qem_uncased' in feature_dict and d_word.lower() in q_uncased_words:
            features[i][feature_dict['f_qem_uncased']] = 1.0
        if 'pos' in document:
            f_pos = 'f_pos={}'.format(document['pos'][i])
            if f_pos in feature_dict:
                features[i][feature_dict[f_pos]] = 1.0
        if 'ner' in document:
            f_ner = 'f_ner={}'.format(document['ner'][i])
            if f_ner in feature_dict:
                features[i][feature_dict[f_ner]] = 1.0
    return features

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    
from typing import Tuple, List

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

    # tokenize both with GPT-2 BPE and get alignment with given tokens
    bpe_toks = roberta.encode(sentence)
    alignment = alignment_utils.align_bpe_to_words(roberta, bpe_toks, tokens) # tokens came from spacy, for this func. from golden tokens
    
    # extract features and align them
    # LM heads are only used when masked_tokens are involved, not in this case
    if not return_all_hiddens:
        features, x = roberta.extract_features(bpe_toks, return_all_hiddens=return_all_hiddens), None
    else:
        features, x = roberta.extract_features(bpe_toks, return_all_hiddens=return_all_hiddens)
    sent_features = features
    features = features.squeeze(0)   #Batch-size = 1
    # aligned_feats = alignment_utils.align_features_to_words(roberta, features, alignment)
    aligned_feats = align_features_to_words(roberta, features, alignment, 1e-3)
   
    return aligned_feats[1:-1], x, sent_features  #exclude <s> and </s> tokens

if __name__ == '__main__':
    set_random_seed(24)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--devset', type=str, default=None, help='Dev set')
    parser.add_argument('--trainset', type=str, default=None, help='Train set')
    parser.add_argument('--testset', type=str, default=None, help='Train set')
    parser.add_argument('--n_history', type=int, default=0)
    parser.add_argument('--question_token_length', type=int, default=512)
    # Optimizer
    group = parser.add_argument_group('training_spec')
    group.add_argument('--predict_raw_text', type=bool, default=True,
                       help='Whether to use raw text and offsets for prediction.')
    args = vars(parser.parse_args())
    config = args
    datasets = prepare_datasets(config)
    #train_set = datasets['train']
    dev_set = datasets['dev']
    #test_set = datasets['test']
    print(dev_set[11])
    
    import torch
    # there are 4 pretrained models - base, large, large-mnli, large-wsc
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli') #'roberta.large')
    roberta.eval()  # disable dropout (or leave in train mode to finetune)
    
    # print(roberta)
    print(roberta.model.classification_heads.mnli.dropout)
    # Encode a pair of sentences and make a prediction
    tokens = roberta.encode(dev_set[11]['question_string'],  dev_set[11]['raw_evidence'][0:512]) #encode can break it up
    print(roberta.predict(head='mnli', tokens=tokens, return_logits=True))#.argmax())  # 0: contradiction
    
    roberta.register_classification_head('new_task', num_classes=3)
    logprobs = roberta.predict('new_task', tokens, return_logits=True)  # tensor([[-1.1050, -1.0672, -1.1245]], grad_fn=<LogSoftmaxBackward>)
    print(logprobs)
    
    roberta.register_classification_head('new_task_2', num_classes=3)
    logprobs = roberta.predict('new_task_2', tokens, return_logits=True)  # tensor([[-1.1050, -1.0672, -1.1245]], grad_fn=<LogSoftmaxBackward>)
    print(logprobs)
    
    print("Token size: " + str(tokens.size()))
    features = roberta.extract_features(tokens.to(device='cpu'))
    x = features[:, 0, :]  # take <s> token (equiv. to [CLS]) # supposed to be 1024, 512+512
    print(x.size())
    x = roberta.model.classification_heads.mnli.dropout(x)
    x = roberta.model.classification_heads.mnli.dense(x)
    x = roberta.model.classification_heads.mnli.activation_fn(x) # this is pooled output - https://huggingface.co/transformers/_modules/transformers/modeling_roberta.html#RobertaForSequenceClassification
    # x = roberta.model.classification_heads.mnli.dropout(x)
    
    # 2 separate FC layers - one for start and end logit, 2 class labels, another for yes/no/unk, 3 class labels
    
    print(x)
    print(x.size())
    
    print(len(dev_set[11]['evidence']['word']))
    doc = roberta.extract_features_aligned_to_words(dev_set[11]['raw_evidence'][0:200]) # For rationale tagging
    print(len(doc))
    
    doc = extract_aligned_roberta(roberta, " ".join(dev_set[11]['evidence']['word'][0:512]), dev_set[11]['evidence']['word'][0:512], return_all_hiddens=False) # dev_set[11]['raw_evidence'][0:200] is a string, and it takes first 200 indices of it, not first 200 tokens, so joining them may solve the problem from tokens
    print(len(doc))
    print(doc.shape)
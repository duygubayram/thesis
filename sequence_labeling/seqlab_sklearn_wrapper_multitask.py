import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import tempfile
from abc import ABCMeta, abstractmethod
from copy import copy
from functools import partial
from pathlib import Path
import os, pickle
from typing import List

import datasets
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, DatasetDict
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    TextClassificationPipeline
from transformers import AutoTokenizer, set_seed
from transformers import DataCollatorWithPadding

from classif_experim.pynvml_helpers import print_gpu_utilization, print_cuda_devices
#from pynvml_helpers import print_gpu_utilization, print_cuda_devices


def set_torch_np_random_rseed(rseed):
    np.random.seed(rseed)
    random.seed(rseed)
    torch.manual_seed(rseed)
    torch.cuda.manual_seed(rseed)
    torch.cuda.manual_seed_all(rseed)

class SklearnTransformerBase(metaclass=ABCMeta):
    def __init__(self, hf_model_label, lang:str, eval=0.1,
                 learning_rate=2e-5, num_train_epochs=3, weight_decay=0.01, batch_size=16, warmup=0.1, gradient_accumulation_steps=1,
                 max_seq_length=128, device=None, rnd_seed=381757, tmp_folder=None):
        '''
        :param hf_model_label: hugginface repo model identifier
        :param tmp_folder: Folder for saving model checkpoints, can be used for resuming the training.
            If None, temporary folder will be used and resuming is not possible.
        :param eval: A proportion of the train set used for model evaluation, or the number of train exapmples used.
        If None, no evaluation will be performed - model will be trained on a fixed number of epochs.
        '''
        self._hf_model_label = hf_model_label
        self._learning_rate = learning_rate; self._num_train_epochs = num_train_epochs
        self._weight_decay = weight_decay
        self._eval = eval; self._lang = lang
        if device: self._device = device
        else: self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._max_seq_length = max_seq_length
        self._tmp_folder = tmp_folder
        self._rnd_seed = rnd_seed
        self._batch_size = batch_size
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._warmup = warmup
        self.tokenizer = None
        self.model = None
        #set_seed(rnd_seed)
        set_torch_np_random_rseed(rnd_seed)

    def _init_temp_folder(self):
        '''
        Initialize temporary folder for the model training checkpoints.
        '''
        if self._tmp_folder is None:
            self._tmp_folder_object = tempfile.TemporaryDirectory()
            self._tmp_folder = self._tmp_folder_object.name
        else:
            assert Path(self._tmp_folder).exists() # todo do assert alway, create exception
        print(f'Temporary folder: {self._tmp_folder}')

    def _cleanup_temp_folder(self):
        if hasattr(self, '_tmp_folder_object'):
            self._tmp_folder_object.cleanup()
            del self._tmp_folder_object
            self._tmp_folder = None # new fit will initiate new tmp. folder creation
        else: # leave the user-specified tmp folder intact
            pass

    def _init_train_args(self):
        ''' Initialize huggingface TrainingArguments. '''
        if self._eval is None:
            save_params = {
                'save_strategy' : 'no',
                'evaluation_strategy' : 'no',
                'output_dir': self._tmp_folder,
            }
        else:
            save_params = {
                'output_dir' : self._tmp_folder,
                'save_strategy' : 'epoch',
                'evaluation_strategy' : 'epoch',
                'save_total_limit' : 2,
                'load_best_model_at_end' : True
            }
        self._training_args = TrainingArguments(
            do_train=True, do_eval=self._eval is not None,
            learning_rate=self._learning_rate, num_train_epochs=self._num_train_epochs,
            warmup_ratio=self._warmup, weight_decay=self._weight_decay,
            per_device_train_batch_size=self._batch_size,
            per_device_eval_batch_size=self._batch_size,
            gradient_accumulation_steps=self._gradient_accumulation_steps,
            overwrite_output_dir=True, resume_from_checkpoint=False,
            **save_params
        )

    @abstractmethod
    def fit(self, X, y):
        '''
        :param X: list-like of texts
        :param y: list-like of labels
        :return:
        '''
        pass

    @abstractmethod
    def predict(self, X):
        '''
        :param X: list-like of texts
        :return: array of label predictions
        '''
        pass

    def __del__(self):
        if hasattr(self, 'model') and self.model is not None: del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            del self.tokenizer_params
        self._cleanup_temp_folder()
        torch.cuda.empty_cache()

    @property
    def device(self): return self._device

    @device.setter
    def device(self, dev):
        self._device = dev

    def save(self, output_dir):
        """
        Save the model, tokenizer, and class configuration to the output directory.
        """
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        # save model and tokenizer
        #model_path = os.path.join(output_dir, 'pytorch_model.bin')
        #torch.save(self.model.state_dict(), model_path)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.save_class_attributes(output_dir)

    ATTRIBUTES_FILE_NAME = 'class_attributes.pkl'

    def save_class_attributes(self, output_dir):
        """
        Save the class attributes to the output directory, excluding 'model' and 'tokenizer'.
        """
        attributes_path = os.path.join(output_dir, self.ATTRIBUTES_FILE_NAME)
        # Save class attributes except 'model' and 'tokenizer'
        # TODO add non-serializable attributes to the list, enable sub-class customization
        with open(attributes_path, 'wb') as attributes_file:
            attributes_to_save = self.__dict__.copy()
            attributes_to_save.pop('model', None)
            attributes_to_save.pop('tokenizer', None)
            pickle.dump(attributes_to_save, attributes_file)

    @classmethod
    def load_class_attributes(cls, input_dir):
        """
        Load class attributes from the specified directory, excluding 'model' and 'tokenizer'.
        """
        attributes_path = os.path.join(input_dir, cls.ATTRIBUTES_FILE_NAME)
        with open(attributes_path, 'rb') as attributes_file:
            attributes = pickle.load(attributes_file)
        instance = cls.__new__(cls)
        instance.__dict__.update(attributes)
        return instance

class SklearnTransformerClassif(SklearnTransformerBase):
    '''
    Adapter of hugginface transformers to scikit-learn interface.
    The workflow is load model, fine-tune, apply and/or save.
    '''

    def _init_tokenizer_params(self):
        if not hasattr(self, 'tokenizer_params'):
            self.tokenizer_params = {'truncation': True}
            if self._max_seq_length is not None: self.tokenizer_params['max_length'] = self._max_seq_length

    def _init_model(self, num_classes):
        '''
        Load model and tokenizer for classification fine-tuning.
        :return:
        '''
        # load and init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self._hf_model_label)
        self._init_tokenizer_params()
        # load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
                        self._hf_model_label, num_labels=num_classes).to(self._device)

    def set_string_labels(self, labels: List[str]):
        ''' Set 1-1 mapping between string labels and corresponding integer indices.
        For binary classification, the labels therefore should be ['NEGATIVE LABEL', 'POSITIVE LABEL'].'''
        assert len(labels) == len(set(labels)) # assert that the labels are unique
        self._str_labels = labels

    def _init_classes(self, labels):
        ''' Init class label data from the labels of the training set. '''
        if not hasattr(self, '_str_labels'): # induce labels from the input list of (train) labels
            self._class_labels = sorted(list(set(l for l in labels)))
        else:
            self._class_labels = copy(self._str_labels)
        self._num_classes = len(self._class_labels)
        self._cls_ix2label = { ix: l for ix, l in enumerate(self._class_labels) }
        self._cls_label2ix = { l: ix for ix, l in enumerate(self._class_labels) }

    def _labels2indices(self, labels):
        ''' Map class labels in input format to numbers in [0, ... , NUM_CLASSES] '''
        return np.array([ix for ix in map(lambda l: self._cls_label2ix[l], labels)])

    def _indices2labels(self, indices):
        ''' Map class indices in [0,...,NUM_CLASSES] to original class labels '''
        return np.array([l for l in map(lambda ix: self._cls_ix2label[ix], indices)])

    def _prepare_dataset(self, X, y):
        '''
        Convert fit() params to hugginface-compatible datasets.Dataset
        '''
        int_labels = self._labels2indices(y)
        df = pd.DataFrame({'text': X, 'label': int_labels})
        if self._eval:
            train, eval = \
                train_test_split(df, test_size=self._eval, random_state=self._rnd_seed, stratify=df[['label']])
            dset = DatasetDict(
                {'train': datasets.Dataset.from_pandas(train), 'eval': datasets.Dataset.from_pandas(eval)})
        else:
            dset = datasets.Dataset.from_pandas(df)
        return dset

    def fit(self, X, y):
        '''
        :param X: list-like of texts
        :param y: list-like of labels
        :return:
        '''
        # delete old model from tmp folder, if it exists
        self._init_classes(y)
        # model and tokenizer init
        self._init_model(self._num_classes)
        self._init_temp_folder()
        self._do_training(X, y)
        self._cleanup_temp_folder()
        # input txt formatting and tokenization
        # training

    def predict(self, X):
        '''
        :param X: list-like of texts
        :return: array of label predictions
        '''
        #todo X 2 pandas df, df to Dataset.from_pandas dset ? or simply from iterable ?
        dset = datasets.Dataset.from_list([{'text': txt} for txt in X])
        pipe = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, device=self._device,
                                          max_length=self._max_seq_length, truncation=True, batch_size=32)
        result = pipe(dset['text'], function_to_apply='softmax')
        del pipe
        torch.cuda.empty_cache()
        # parse predictions, map to original labels
        #todo regex-based extraction of integers from the specific format
        pred = [int(r['label'][-1]) for r in result] # assumes *LABEL$N format
        return self._indices2labels(pred)

    def tokenize(self, txt, **kwargs):
        self._init_tokenizer_params()
        # joint self.tokenizer_params and kwargs
        params = self.tokenizer_params.copy()
        for k, v in kwargs.items(): params[k] = v
        return self.tokenizer(txt, **params)

    def _do_training(self, X, y):
        torch.manual_seed(self._rnd_seed)
        def preprocess_function(examples):
            return self.tokenizer(examples['text'], **self.tokenizer_params)
        dset = self._prepare_dataset(X, y)
        tokenized_dset = dset.map(preprocess_function, batched=True)
        self._init_train_args()
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        if self._eval: train, eval = tokenized_dset['train'], tokenized_dset['eval']
        else: train, eval = tokenized_dset, None
        trainer = Trainer(
            model=self.model,
            args=self._training_args,
            train_dataset=train,
            eval_dataset=eval,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        trainer.train()
        if self.model is not trainer.model: # just in case
            del self.model
            self.model = trainer.model
        del trainer
        torch.cuda.empty_cache()

    @classmethod
    def load(cls, input_dir, device=None):
        """
        Load the model, tokenizer, and class configuration from the input directory.
        """
        instance = cls.load_class_attributes(input_dir)
        # load tokenizer and model
        # TODO move tokenizer loading to superclass?
        #tokenizer_path = os.path.join(input_dir, 'tokenizer_config.json')
        instance.tokenizer = AutoTokenizer.from_pretrained(input_dir)
        model_path = os.path.join(input_dir, 'pytorch_model.bin')
        if device is None: device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        instance.model = AutoModelForSequenceClassification.from_pretrained(input_dir).to(device)
        return instance


def test_hf_wrapper(test_dset, model='bert-base-uncased', device='cuda:0', subsample=500, rnd_seed=4140, eval=0.1):
    # prepare test dataset
    dset = load_dataset(test_dset)
    texts = np.array(dset['train']['text'])
    labels = np.array(dset['train']['label'])
    if subsample:
        random.seed(rnd_seed)
        ixs = random.sample(range(len(texts)), subsample)
        texts, labels = texts[ixs], labels[ixs]
    txt_trdev, txt_test, lab_trdev, lab_test = \
        train_test_split(texts, labels, test_size=0.8, random_state=rnd_seed, stratify=labels)
    # train model, evaluate
    tr = SklearnTransformerClassif(num_train_epochs=5, eval=eval, hf_model_label=model, rnd_seed=rnd_seed, device=device,
                                   lang='en')
    tr.fit(txt_trdev, lab_trdev)
    lab_pred = tr.predict(txt_test)
    f1 = partial(f1_score, average='binary')
    acc = accuracy_score
    print(f'f1: {f1(lab_test, lab_pred):.3f}, acc: {acc(lab_test, lab_pred):.3f}')

if __name__ == '__main__':
    test_hf_wrapper(test_dset='imdb', subsample=100, eval=None)

  
  
  
  

########



#import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
from typing import List, Tuple, Dict, Union

import datasets
import pandas as pd
import torch
import transformers
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from spacy.tokens import Doc
from torch import tensor as TT
from transformers import AutoTokenizer, DataCollatorForTokenClassification, Trainer

#from classif_experim.hf_skelarn_wrapper import SklearnTransformerBase
from data_tools.spacy_utils import get_doc_id, get_annoation_tuples_from_doc
from sequence_labeling.multi_task_model import MultiTaskModel, Task
from sequence_labeling.spanannot2hf import extract_spans, convert_to_hf_format, labels_from_predictions, \
    align_labels_with_tokens, extract_span_ranges


class OppSequenceLabelerMultitask(SklearnTransformerBase):
    '''
    'Oppositional Sequence Labeler', wraps the data transformation functionality and the multitask HF model
    for sequence labeling into a sklearn-like interface.
    '''

    def __init__(self, task_labels, task_indices, empty_label_ratio=0.2, loss_freq_weights=False, task_importance=None, **kwargs):
        super().__init__(**kwargs)
        self._empty_label_ratio = empty_label_ratio
        self._loss_freq_weights = loss_freq_weights
        self._task_importance = task_importance
        self.task_labels, self.task_indices = task_labels, task_indices

    def dataset_stats(self, docs: List[Doc]):
        '''
        Calculate and print statistics of the transformed dataset that will be used for training.
        :return:
        '''
        old_eval = self._eval
        span_labels = [get_annoation_tuples_from_doc(doc) for doc in docs]
        self._eval = None # entire dataset will be the train set
        self._init_tokenizer()
        #self._init_model()
        #self._init_temp_folder()
        self._construct_train_eval_raw_datasets(docs, span_labels)
        self._calc_dset_stats(verbose=True)
        #self._do_training()
        #self._cleanup_temp_folder()
        self._eval = old_eval

    def _calc_dset_stats(self, verbose=False):
        ''' Calculate and print dataset statistics using self._raw_train '''
        if verbose: print(f'Seqlabel train dataset statistics for {self._lang}:')
        # use self._raw_train
        # for each label, count and print the number of instances, as well as the number of instances with no spans
        # print these statistics, use fixed ordering of the labels
        num = sum([len(self._raw_train[label]) for label in self.task_labels])
        if verbose: print(f'Total: {num} instances')
        self._task_frequencies = {}
        for label in self.task_labels:
            dset = self._raw_train[label]
            num_instances = len(dset)
            num_empty = len([1 for i in range(num_instances) if list(set(dset[i]['ner_tags'])) == [0]])
            self._task_frequencies[label] = num_instances/num
            if verbose:
                print(f'{label:<3}: {num_instances:<5} instances ({num_instances/num*100:5.3f}%) '
                    f'{num_empty:<5} empty ({num_empty/num*100:5.3f}%)')

    def fit(self, docs: List[Doc], span_labels: List[Tuple[str, int, int, str]]):
        '''
        :param docs: spacy Docs
        :param span_labels: list of lists of spans, span is a tuple of (label, start, end, text),
                where start and end are token indices, ie, doc[start:end] is the span text
        :return:
        '''
        self._init_tokenizer()
        self._construct_datasets_for_inference(docs, span_labels)
        self._init_model()
        self._init_temp_folder()
        self._do_training()
        self._cleanup_temp_folder()
        # input txt formatting and tokenization
        # training

    def fit_(self, docs: List[Doc]):
        '''
        Helper to enable fitting without previously extracting spans a separate list.
        docs: spacy Docs, with annotated spans, in the format defined in 'create_spacy_span_dataset.py'
        :return:
        '''
        span_labels = [get_annoation_tuples_from_doc(doc) for doc in docs]
        self.fit(docs, span_labels)

    def _do_training(self):
        train_dataset = self._dataset['train']
        eval_dataset = self._dataset['eval'] if self._eval else None
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        self._init_train_args()
        trainer = Trainer(model=self.model, args=self._training_args,
            train_dataset=train_dataset, eval_dataset=eval_dataset if self._eval else None,
            tokenizer=self.tokenizer, data_collator=data_collator)
        trainer.train()
        if self.model is not trainer.model: # just in case
            del self.model
            self.model = trainer.model
        del trainer
        torch.cuda.empty_cache()

    def _construct_datasets_for_inference(self, docs, spans):
        self._construct_train_eval_raw_datasets(docs, spans)
        self._calc_dset_stats(verbose=False)
        #self._inspect_data(self._raw_train, self.span_labels, num_samples=5)
        self._hf_tokenize_task_dataset()

    def _construct_train_eval_raw_datasets(self, docs, spans):
        ''' Construct 'raw' datasets, but separately for train end eval. '''
        if self._eval:
            docs_train, docs_eval, spans_train, spans_eval = \
                train_test_split(docs, spans, test_size=self._eval, random_state=self._rnd_seed)
            self._raw_train = self._construct_raw_hf_dataset(docs_train, spans_train, downsample=self._empty_label_ratio)
            self._raw_eval = self._construct_raw_hf_dataset(docs_eval, spans_eval, downsample=self._empty_label_ratio)
        else:
            self._raw_train = self._construct_raw_hf_dataset(docs, spans, downsample=self._empty_label_ratio)
            self._raw_eval = None

    def _construct_raw_hf_dataset(self, docs, span_labels, downsample) -> Dict[str, Dataset]:
        '''
        Convert the data to HF format: create one HF dataset per label, in BIO format.
        '''
        data_by_label = extract_spans(docs, span_labels, downsample_empty=downsample, rnd_seed=self._rnd_seed,
                                      label_set=self.task_labels)
        datasets = {}
        for label, data in data_by_label.items():
            datasets[label] = convert_to_hf_format(label, data)
        return datasets

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self._hf_model_label)
        if isinstance(self.tokenizer, transformers.RobertaTokenizerFast):
            self.tokenizer.add_prefix_space = True
        self.tokenizer_params = {'truncation': True}
        if self._max_seq_length is not None: self.tokenizer_params['max_length'] = self._max_seq_length

    def _get_narrative_tasks(self):
        '''
        Definition of tasks for this sequence labeling problem, compatible with MultiTaskModel.
        '''
        return [
            Task(id=self.task_indices[task_id], name=None, num_labels=3, type="token_classification")
            for task_id in self.task_labels
        ]

    def _calculate_task_weights(self):
        ''' Calculate task weights from task frequencies and importance weights. '''
        def normalize_weight_map(weights):
            sum_weights = sum(weights.values())
            for label in self.task_labels: weights[self.task_indices[label]] /= sum_weights
        if not self._loss_freq_weights and not self._task_importance:
            self._task_weights = None
        else: # calculate freq. weights, importance weights, or their combination if both are provided
            self._task_weights = {}
            if self._loss_freq_weights: # use loss frequency weights
                for label in self.task_labels: self._task_weights[self.task_indices[label]] = 1/self._task_frequencies[label]
                normalize_weight_map(self._task_weights)
                if self._task_importance:
                    for label in self.task_labels: self._task_weights[self.task_indices[label]] *= self._task_importance[label]
                    normalize_weight_map(self._task_weights)
            else:
                for label in self.task_labels: self._task_weights[self.task_indices[label]] = self._task_importance[label]
                normalize_weight_map(self._task_weights)

    def _init_model(self):
        self._calculate_task_weights()
        self._is_roberta = 'roberta' in self._hf_model_label.lower()
        self.model = MultiTaskModel(self._hf_model_label, self._get_narrative_tasks(), task_weights=self._task_weights).to(self._device)

    def _hf_tokenize_task_dataset(self):
        '''
        Given a HF dataset for a single task (label) produced by _construct_hf_datasets,
        tokenize it, add taks labels, and return the tokenized dataset.
        '''
        # for each label, perform hf tokenization
        raw_dset_per_label = {}
        for label in self.task_labels:
            if not self._eval: raw_dataset = DatasetDict({'train': self._raw_train[label]})
            else: raw_dataset = DatasetDict({'train': self._raw_train[label], 'eval': self._raw_eval[label]})
            #label_list = raw_dataset['train'].features['ner_tags'].feature.names
            tokenized_dataset = self._tokenize_token_classification_dataset(
                raw_datasets=raw_dataset, tokenizer=self.tokenizer, task_id=self.task_indices[label])
            raw_dset_per_label[label] = tokenized_dataset
        # merge per-label datasets into one, for multi-task training
        dset_splits = ['train', 'eval'] if self._eval else ['train']
        datasets_df = { split: None for split in dset_splits }
        for label, raw_dset in raw_dset_per_label.items(): # merge datasets as pandas dataframes
            for split in dset_splits:
                if datasets_df[split] is None:
                    datasets_df[split] = raw_dset[split].to_pandas()
                else:
                    datasets_df[split] = pd.concat([datasets_df[split], raw_dset[split].to_pandas()], ignore_index=True)
        # convert dataframes backt to HF datasets, shuffle, and create final dataset dict
        merged_datasets = { split: datasets.Dataset.from_pandas(datasets_df[split]) for split in dset_splits }
        for split in dset_splits:
            merged_datasets[split].shuffle(seed=self._rnd_seed)
        if self._eval is None:
            self._dataset = datasets.DatasetDict({'train': merged_datasets['train']})
        else:
            self._dataset = datasets.DatasetDict({'train': merged_datasets['train'], 'eval': merged_datasets['eval']})

    def _tokenize_token_classification_dataset(self, raw_datasets: Union[DatasetDict, Dataset], tokenizer, task_id,
                                               text_column_name='tokens', label_column_name='ner_tags'):
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples[text_column_name],
                padding=True,
                truncation=True,
                max_length=self._max_seq_length,
                is_split_into_words=True,
            )
            labels = []
            for i, label in enumerate(examples[label_column_name]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None: label_ids.append(-100) # for special token set label to -100 (to ignore in loss)
                    elif word_idx != previous_word_idx: # set the label only for the first token of a word
                        label_ids.append(label[word_idx])
                    else: label_ids.append(-100) # for consecutive tokens of multi-token words, set the label to -100
                    previous_word_idx = word_idx
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            tokenized_inputs["task_ids"] = [task_id] * len(tokenized_inputs["labels"])
            return tokenized_inputs
        if isinstance(raw_datasets, DatasetDict):
            tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True, num_proc=1,
                                load_from_cache_file=False)
        elif isinstance(raw_datasets, Dataset): # helper code that enables to tokenize a single Dataset as DatasetDict
            dset = DatasetDict({'dset': raw_datasets})
            tokenized_datasets = dset.map(tokenize_and_align_labels, batched=True, num_proc=1,
                                load_from_cache_file=False)
            tokenized_datasets = tokenized_datasets['dset']
        else: raise ValueError(f'Unknown dataset type: {type(raw_datasets)}')
        return tokenized_datasets

    def _inspect_data(self, datasets, label_list, num_samples=10):
        def print_aligned_tokens_and_tags(tokens, ner_tags):
            token_str = ' '.join([f"{token:<{len(token) + 2}}" for token in tokens])
            print(token_str)
            ner_tag_str = ' '.join([f"{str(tag):<{len(tokens[i]) + 2}}" for i, tag in enumerate(ner_tags)])
            print(ner_tag_str)
            print("\n" + "-" * 40)

        for label in label_list:
            dataset = datasets[label]
            sampled_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
            print(f"===== ORIGINAL TEXT FOR LABEL: {label} =====")
            for idx in sampled_indices:
                tokens = dataset[idx]['tokens']
                ner_tags = dataset[idx]['ner_tags']
                print_aligned_tokens_and_tags(tokens, ner_tags)
            tokenized_dataset = \
            self._tokenize_token_classification_dataset(DatasetDict({'train': dataset}),
                                                        self.tokenizer, self.task_indices[label])['train']
            print(f"\n===== TOKENIZED TEXT FOR LABEL: {label} =====")
            for idx in sampled_indices:
                tokens = self.tokenizer.convert_ids_to_tokens(tokenized_dataset[idx]['input_ids'])
                labels = tokenized_dataset[idx]['labels']
                print_aligned_tokens_and_tags(tokens, labels)
            print("\n\n")

    def _construct_predict_dataset(self, docs, spans=None):
        if spans == None: # no spans provided, create a list of #docs empty lists for compatibility
            spans = [[] for _ in range(len(docs))]
        raw_dset_per_label = self._construct_raw_hf_dataset(docs, spans, downsample=None)
        tokenized_dset_per_label = {}
        for label in self.task_labels:
            tokenized_dataset = self._tokenize_token_classification_dataset(
                raw_datasets=raw_dset_per_label[label], tokenizer=self.tokenizer, task_id=self.task_indices[label])
            tokenized_dset_per_label[label] = tokenized_dataset
        return tokenized_dset_per_label
    
    # my addition    
#    def save_model(self, save_dir):

#        if not os.path.exists(save_dir):
#            os.makedirs(save_dir)
    
#        encoder_path = os.path.join(save_dir, "encoder")
#        self.encoder.save_pretrained(encoder_path)
    
#        config_path = os.path.join(save_dir, "config.json")
#        with open(config_path, "w") as f:
#            json.dump(self.config, f)
    
#        state_path = os.path.join(save_dir, "model_state.pth")
#        torch.save(self.state_dict(), state_path)
    
#        logger.info(f"Model saved successfully to {save_dir}.")


    def predict(self, X: List[Doc]) -> List[List[Tuple[str, int, int, str]]]:
        '''
        :return: list of lists of spans (full annotations for one document); each span is a tuple of (label, start, end, author),
            for the data to be in the same format as in the original spacy data
        '''
        text_labels_pred = {} # intermediate map with output, and the helper function for adding labels to it
        def add_labels_to_map(label_map, text_id, task_label, labels: List[str]):
            if text_id not in label_map: label_map[text_id] = {}
            if task_label not in label_map[text_id]: label_map[text_id][task_label] = labels
            else: raise ValueError(f'Label map already contains labels for text id {text_id} and task {task_label}')
        # tokenize input, predict, transform data
        tokenized_dset_per_label = self._construct_predict_dataset(X)
        for label in self.task_labels:
            dset = tokenized_dset_per_label[label]
            for t in dset:
                if not self._is_roberta: ttids = t['token_type_ids']
                else: ttids = [0]
                ids, att, tti, tsk = [t['input_ids']], [t['attention_mask']], [ttids], [self.task_indices[label]]
                ids, att, tti, tsk = TT(ids, device=self.device), TT(att, device=self.device), \
                                     TT(tti, device=self.device), TT(tsk, device=self.device)
                res, _ = self.model(ids, att, tti, task_ids=tsk)
                preds = res[0].cpu().detach().numpy()
                orig_tokens = t['tokens']
                pred_labels = labels_from_predictions(preds, t['labels'], label)
                pred_labels = align_labels_with_tokens(orig_tokens, pred_labels)
                add_labels_to_map(text_labels_pred, t['text_ids'], label, pred_labels)
        # convert the map to the format of the original spacy data
        id2doc = {get_doc_id(doc): doc for doc in X}
        span_labels_pred = {text_id:[] for text_id in text_labels_pred.keys()}
        for text_id in text_labels_pred.keys():
            for task_label in text_labels_pred[text_id].keys():
                span_bio_tags = text_labels_pred[text_id][task_label]
                if len(span_bio_tags) == 0: continue
                doc = id2doc[text_id]
                # assert that doc has the same number of tokens as there are bio tags
                assert len(doc) == len(span_bio_tags)
                tokens = [token.text for token in doc]
                spans = extract_span_ranges(tokens, span_bio_tags, allow_hanging_itag=True)
                span_labels_pred[text_id].extend([(task_label, start, end, self._hf_model_label) for start, end in spans])
        return [span_labels_pred[get_doc_id(doc)] for doc in X]


if __name__ == '__main__':
    pass
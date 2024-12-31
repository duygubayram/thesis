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



import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from copy import copy
import logging
import time
import datetime
import pandas as pd
import sentencepiece
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import shutil


from classif_experim.classif_utils import classif_scores
#from classif_experim.hf_sklearn_wrapper import SklearnTransformerClassif
#from hf_wrapper.hf_sklearn_wrapper import SklearnTransformerClassif
from data_tools.dataset_loaders import load_dataset_classification

##
#import neptune.new as neptune
#from neptune.new.integrations.transformers import NeptuneCallback
##

##
#run = neptune.init_run(
#    project="duygu/thesis",
#    api_token=""
#)
##


def build_transformer_model(model_label, model_hparams, rnd_seed):
    ''' Factory method for building a sklearn-wrapped transformer model.'''
    return SklearnTransformerClassif(hf_model_label=model_label, **model_hparams, rnd_seed=rnd_seed)

def run_classif_crossvalid(lang, model_label, model_params, positive_class='critical', num_folds=5,
                           rnd_seed=3154561, test=False, pause_after_fold=0):
    '''
    Run x-fold crossvalidation for a given model, and report the results.
    '''
    logger.info(f'RUNNING crossvalid. for model: {model_label}')
    score_fns = classif_scores('all')
    texts, classes, txt_ids = load_dataset_classification(lang, positive_class=positive_class)
    if test: texts, classes, txt_ids = texts[:test], classes[:test], txt_ids[:test]
    foldgen = StratifiedKFold(n_splits=num_folds, random_state=rnd_seed, shuffle=True)
    fold_index = 0
    results_df = pd.DataFrame(columns=score_fns.keys())
    conf_mx = None; rseed = rnd_seed
    pred_res = {} # map text_id -> class prediction
    
    best_mcc = -float('inf')
    best_model_dir = None
    
    for train_index, test_index in foldgen.split(texts, classes):
        logger.info(f'Starting Fold {fold_index+1}')
        model = build_transformer_model(model_label, model_params, rseed)
        logger.info(f'model built')
        # split data
        txt_tr, txt_tst = texts[train_index], texts[test_index]
        cls_tr, cls_tst = classes[train_index], classes[test_index]
        id_tst = txt_ids[test_index]
        # train model
        model.fit(txt_tr, cls_tr)
        #model_dir = f"saved_model_{model_label}_fold_{fold_index}"
        #model.save(model_dir)
        #logger.info(f"Model saved to {model_dir}")
        # evaluate model
        cls_pred = model.predict(txt_tst)
        for txt_id, pred in zip(id_tst, cls_pred):
            assert txt_id not in pred_res
            pred_res[txt_id] = pred
        # del model # clear memory
        scores = pd.DataFrame({fname: [f(cls_tst, cls_pred)] for fname, f in score_fns.items()})
        # log scores
        logger.info(f'Fold {fold_index+1} scores:')
        logger.info("; ".join([f"{fname:10}: {f(cls_tst, cls_pred):.3f}" for fname, f in score_fns.items()]))
        # formatted_values = [f"{col:10}: {scores[col].iloc[0]:.3f}" for col in scores.columns]
        results_df = pd.concat([results_df, scores], ignore_index=True)
        
        if "MCC" in scores.columns:
            fold_mcc = scores["MCC"].iloc[0]
        else:
            logger.error("MCC not found in scores. Terminating...")
            sys.exit("Error: MCC metric is missing in the scores DataFrame!")

        if fold_mcc == -float('inf'):
            logger.error(f"Invalid MCC value: {fold_mcc}. Terminating...")
            sys.exit(f"Error: Invalid MCC value detected: {fold_mcc}")
        
        #fold_mcc = scores.get("mcc", pd.Series([-float('inf')])).iloc[0]
        if fold_mcc > best_mcc:
            if best_model_dir and os.path.exists(best_model_dir):
                logger.info(f"Deleting old model directory: {best_model_dir}")
                try:
                    shutil.rmtree(best_model_dir)
                except Exception as e:
                    logger.error(f"Failed to delete old model directory: {e}")
                    raise SystemExit(f"Critical Error: Unable to delete old model directory. {e}")
        
            best_mcc = fold_mcc
            parent_folder = os.path.dirname(os.path.abspath(__file__))
            os.makedirs(parent_folder, exist_ok=True)
            best_model_dir = os.path.join(parent_folder, f"classification_model_{model_label}_fold_{fold_index+1}_rseed[{rnd_seed}]")
            if not os.path.exists(best_model_dir):
                os.makedirs(best_model_dir)
            try:
                logger.info(f"Saving model to: {best_model_dir}")
                model.save(best_model_dir)
                
                if not os.listdir(best_model_dir):
                    raise RuntimeError(f"Model save failed. Directory is empty: {best_model_dir}")

                logger.info(f"New best model saved with MCC: {best_mcc:.4f} at {best_model_dir}")
            except Exception as e:
                logger.error(f"Error saving model: {e}")
                raise SystemExit(f"Critical Error: Unable to save model. {e}")
            #model.save(best_model_dir)  # Save the best model
            #logger.info(f"New best model saved with MCC: {best_mcc:.4f} at {best_model_dir}")
            
        del model
        conf_mx_tmp = confusion_matrix(cls_tst, cls_pred)
        if conf_mx is None: conf_mx = conf_mx_tmp
        else: conf_mx += conf_mx_tmp
        if pause_after_fold and fold_index < num_folds - 1:
            logger.info(f'Pausing for {pause_after_fold} minutes...')
            time.sleep(pause_after_fold * 60)
        rseed += 1; fold_index += 1
    conf_mx = conf_mx.astype('float64')
    conf_mx /= num_folds
    logger.info('CROSSVALIDATION results:')
    for fname in score_fns.keys():
        logger.info(f'{fname:10}: ' + '; '.join(f'{nm}: {val:.3f}' for nm, val in results_df[fname].describe().items()))
    logger.info('Per-fold scores:')
    # for each score function, log all the per-fold results
    for fname in score_fns.keys():
        logger.info(f'{fname:10}: [{", ".join(f"{val:.3f}" for val in results_df[fname])}]')
    logger.info('Confusion matrix:')
    for r in conf_mx:
        logger.info(', '.join(f'{v:7.2f}' for v in r))
    assert set(pred_res.keys()) == set(txt_ids)
    
    logger.info(f"Best model saved to: {best_model_dir} with MCC: {best_mcc:.4f}")
    
    return pred_res

MAX_SEQ_LENGTH = 256

HF_MODEL_LIST = {
    'en': [
           #'bert-base-cased',
           'microsoft/deberta-v3-large',
           #'roberta-large',
           #'digitalepidemiologylab/covid-twitter-bert',
          ],
    'es': [
            #'dccuchile/bert-base-spanish-wwm-cased',
          ],
}

# default reasonable parameters for SklearnTransformerBase
HF_CORE_HPARAMS = {
    'learning_rate': 2e-5,
    'num_train_epochs': 3,
    'warmup': 0.1,
    'weight_decay': 0.01,
    'batch_size': 16,
}

# try as list
#HF_CORE_HPARAMS = {
#    'learning_rate': [1e-6, 2e-6, 1e-5, 2e-5, 1e-3, 2e-3],
#    'num_train_epochs': [2, 3, 4],
#    'warmup': 0.1,
#    'weight_decay': 0.01,
#    'batch_size': [4, 8, 16, 32],
#}

DEFAULT_RND_SEED = 564671

logger = None
def setup_logging(log_filename):
    global logger
    logging.basicConfig(
        level=logging.INFO,  # Log INFO level and above
        handlers=[
            logging.FileHandler(log_filename),  # Log to a file with timestamp in its name
            logging.StreamHandler()  # Log to console
        ]
    )
    logger = logging.getLogger('')

def run_classif_experiments(lang, num_folds, rnd_seed, test=False, experim_label=None,
                            pause_after_fold=0, pause_after_model=0, max_seq_length=MAX_SEQ_LENGTH,
                            positive_class='critical', model_list=None):
    '''
    :param positive_class: 'critical' or 'conspiracy'
    :return:
    '''
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experim_label = f'{experim_label}_rseed_{rnd_seed}' if experim_label else f'rseed_{rnd_seed}'
    log_filename = f"classification_experiments_{experim_label}_{timestamp}.log"
    setup_logging(log_filename)
    models = HF_MODEL_LIST[lang] if model_list is None else model_list
    params = copy(HF_CORE_HPARAMS)
    params['lang'] = lang
    params['eval'] = None
    params['max_seq_length'] = max_seq_length
    logger.info(f'RUNNING classif. experiments: lang={lang.upper()}, num_folds={num_folds}, '
                f'max_seq_len={max_seq_length}, eval={params["eval"]}, rnd_seed={rnd_seed}, test={test}')
    logger.info(f'... HPARAMS = {"; ".join(f"{param}: {val}" for param, val in HF_CORE_HPARAMS.items())}')
    init_batch_size = params['batch_size']
    pred_res = {}
    for model in models:
        try_batch_size = init_batch_size
        grad_accum_steps = 1
        while try_batch_size >= 1:
            try:
                params['batch_size'] = try_batch_size
                params['gradient_accumulation_steps'] = grad_accum_steps
                res = run_classif_crossvalid(lang=lang, model_label=model, model_params=params, num_folds=num_folds,
                                             rnd_seed=rnd_seed, test=test, pause_after_fold=pause_after_fold,
                                             positive_class=positive_class)
                pred_res[model] = res
                break
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logging.warning(
                        f"GPU out of memory using batch size {try_batch_size}. Halving batch size and doubling gradient accumulation steps.")
                    try_batch_size //= 2
                    grad_accum_steps *= 2
                else:
                    raise e
            if try_batch_size < 1:
                logging.error("Minimum batch size reached and still encountering memory errors. Exiting.")
                break
        if pause_after_model:
            logger.info(f'Pausing for {pause_after_model} minutes...')
            time.sleep(pause_after_model * 60)
    return pred_res

def run_all_critic_conspi(seed=DEFAULT_RND_SEED, langs=['en']):
    for lang in langs:
        run_classif_experiments(lang=lang, num_folds=5, rnd_seed=seed, test=None,
                                positive_class='critical', pause_after_fold=1,
                                pause_after_model=2)

if __name__ == '__main__':
    run_all_critic_conspi()


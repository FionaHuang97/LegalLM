import os
import pickle
import torch
import json
import random
import math
import numpy as np
import yaml
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset_build import make_raw_citation_vocab, normalize_vocabulary, make_thresholded_vocabulary
from dataset_vocab import CitationPredictionDataset, WrappedCitationTokenizer
from transformers import RobertaConfig, AutoConfig, TrainingArguments, Trainer, RobertaTokenizerFast
from bilstm import BiLSTM, CheckpointEveryNSteps, bilstm_eval_class, bilstm_eval_idx, bilstm_test_n_fold
from roberta import RobertaForSequenceClassification, roberta_evaluate_class, roberta_evaluate_idx, roberta_test_n_fold, roberta_model_analysis_val_data
from preprocessing import MetadataProcessor, DataPreprocessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
    generateCitation() takes the customize input texts and metadata, extracts citations from texts, clean texts
    and padding(or truncation depends on input length) before forwarding input to model, top n generated citations
    will be printed, n is configurable with default=1
    input_text: customize input texts (input in config file)
    input_meta: customize input metadata (input in config file)
    n: the number of citations desire to generate
"""


def generate_citation(input_text, input_meta='-1,-1,-1', n=1):

    context_length = config_file["context_length"]
    pp = DataPreprocessor()
    processed_input = pp.process_input_texts(input_text)
    pad_token_id = wrappedTokenizor.wrapped_tokenizer.pad_token_id
    citation_indices = [cvx.citation_indices_from_raw(cit)\
                        for cit in processed_input['citation_texts']]
    encoded = wrappedTokenizor.encode(processed_input["txt"], citation_indices)
    attention = torch.tensor([1] * len(encoded))
    if len(encoded) < context_length:
        pre_padding = torch.tensor([pad_token_id] * (context_length - len(encoded)))
        pre_attention = torch.tensor([0] * (context_length - len(encoded)))
        encoded = torch.cat([torch.tensor(pre_padding, encoded)])
        attention = torch.cat([pre_attention, attention])
    else:
        encoded = torch.tensor(encoded[len(encoded) - context_length:])
        attention = attention[len(encoded) - context_length:]
    metadata = torch.tensor([int(meta) for meta in input_meta.split(",")])
    attention_padding = torch.tensor([0] * len(metadata))
    encoded = torch.cat([encoded, metadata]).unsqueeze(0)
    attention = torch.cat([attention, attention_padding]).unsqueeze(0)
    logits = trainer.prediction_step(model, {"input_ids": encoded, "attention_mask": attention})[1].detach().tolist()[0]
    idxs = sorted(range(len(logits)), key=lambda sub: logits[sub], reverse=True)[:n]
    for idx in idxs:
        cit_str = list(cvx.citation_counts)[idx]
        print(f"Predicted citation string is {cit_str}, cit index is {idx}")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file used for training', required=True)
    args = parser.parse_args()

    # load the config file
    with open(args.config) as file:
        config_file = yaml.safe_load(file)

    # sanity checks for metadata settings
    assert config_file['add_case_meta'] or (not config_file['enable_meta_year'] and not config_file['enable_meta_issarea'] and not config_file['enable_meta_judge'])

    # load cvx (reduced and thresholded citation vocab)
    if not os.path.exists(config_file['cv_norm_path']):
        print('no normalized citation vocabulary found')
        # load cv(citation vocab)
        if not os.path.exists(config_file['cv_path']):
            print('Building Citation Vocabulary')
            cv = make_raw_citation_vocab(config_file['preprocessed_dir'], config_file['train_ids_fpath'])
            with open(config_file['cv_path'], 'wb') as f:
                pickle.dump(cv, f)
        else:
            print('cv loaded')
            with open(config_file['cv_path'], 'rb') as f:
                cv = pickle.load(f)
        print('Raw Citation Vocabulary available:')
        cv.vocab_report()
        print("Reducing Citation Vocabulary")
        cvx = normalize_vocabulary(cv, config_file['citation_dict_fpaths'])
        cvx.vocab_report()
        print("Thresholding Citation Vocabulary")
        cvx = make_thresholded_vocabulary(cvx, 20)
        cvx.vocab_report()
        print('Normalized Citation Vocabulary available:')
        cvx.vocab_report()
        #with open(config_file['cv_norm_path'], 'wb') as f:
        #    pickle.dump(cvx, f)
    else:
        print('loading normalized vocabulary from '+config_file['cv_norm_path'])
        with open(config_file['cv_norm_path'], 'rb') as f:
            cvx = pickle.load(f)
            print('Reduced & Thresholded Citation Vocabulary:')
            cvx.vocab_report()

    # load metadata
    meta_cache_fpath = config_file['meta_fpath'].rsplit('.', 1)[0] + "_processed.pickle"
    meta = None
    if config_file['add_case_meta'] and not os.path.exists(meta_cache_fpath):
        print("Building metadata")
        mp = MetadataProcessor()
        meta = mp.build_metadata(metadata_fpath=config_file['meta_fpath'])
        with open(meta_cache_fpath, 'wb') as f:
            pickle.dump(meta, f)
    elif config_file['add_case_meta']:
        print("Loading metadata")
        with open(meta_cache_fpath, 'rb') as f:
            meta = pickle.load(f)

    # set num_meta
    if config_file['add_case_meta']:
        num_meta = 3
    else:
        num_meta = 0

    # set num_labels based on task
    num_labels = -1
    negative_sample_prob = None
    if config_file['task'].startswith("binary"):
        num_labels = 2
        negative_sample_prob = None
    elif config_file['task'].startswith("cit_class"):
        num_labels = 5
        negative_sample_prob = 0.25
    elif config_file['task'].startswith("cit_idx"):
        num_labels = len(cvx)
        negative_sample_prob = 0
    else:
        print("Error: unknown task")
        exit()
    print(f"num labels {num_labels}")

    # set num_folds
    num_folds = 6
    #set learning rate
    lr = float(config_file['learning_rate'])

    if config_file['model_type'].startswith('bilstm'):
        dataset_return_type = 'lightning'
    elif config_file['model_type'].startswith('roberta'):
        dataset_return_type = 'features'
    else:
        print("Error: invalid model name")
        exit()

    tokenizer = RobertaTokenizerFast.from_pretrained(config_file['pretrain_name'])
    wrappedTokenizor = WrappedCitationTokenizer(tokenizer, cvx)
    train_dataset = CitationPredictionDataset(
        config_file['preprocessed_dir'],
        cvx,
        case_ids=None,
        case_ids_fpath=config_file['train_ids_fpath'],
        tokenizer=wrappedTokenizor,
        target_mode=config_file['task'],
        ignore_unknown=True,
        negative_sample_prob=negative_sample_prob,
        add_case_meta=config_file['add_case_meta'],
        meta=meta,
        forecast_length=config_file['forecast_length'],
        context_length=config_file['context_length'],
        pre_padding=False,
        return_type=dataset_return_type,
    )
    dev_dataset = CitationPredictionDataset(
        config_file['preprocessed_dir'],
        cvx,
        case_ids=None,
        case_ids_fpath=config_file['dev_ids_fpath'],
        tokenizer=wrappedTokenizor,
        target_mode=config_file['task'],
        ignore_unknown=True,
        negative_sample_prob=negative_sample_prob,
        add_case_meta=config_file['add_case_meta'],
        meta=meta,
        forecast_length=config_file['forecast_length'],
        context_length=config_file['context_length'],
        pre_padding=False,
        return_type=dataset_return_type,
    )

    if config_file['model_type'].startswith('bilstm'):
        train_loader = DataLoader(
            train_dataset,
            batch_size=config_file['batch_size'],
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=config_file['batch_size'],
            shuffle=False,
            num_workers=4,
            drop_last=True,
        )
        model = BiLSTM(
            embedding_dim=768,
            hidden_dim=3072,
            vocab_size=len(wrappedTokenizor),
            batch_size=config_file['batch_size'],
            label_size=num_labels,
            task=config_file['task'],
            num_meta=num_meta,
            enable_meta_year=config_file['enable_meta_year'],
            enable_meta_issarea=config_file['enable_meta_issarea'],
            enable_meta_judge=config_file['enable_meta_judge'],
            device=device,
            lr=lr,
        ).to(device)
        if config_file['mode'] == "train":
            logger = TensorBoardLogger(
                config_file['output_dir'],
                name="bilstm",
            )
            checkpoint_callback = ModelCheckpoint(
                monitor='avg_val_loss',
                save_top_k=20,
            )
            trainer = pl.Trainer(
                weights_save_path=config_file['output_dir'],
                checkpoint_callback=checkpoint_callback,
                # callbacks=[CheckpointEveryNSteps(save_step_frequency=400)],
                gpus=int(device.type == "cuda"),
                logger=logger,
                log_every_n_steps=10,
                flush_logs_every_n_steps=10,
                val_check_interval=1.0,
                resume_from_checkpoint=config_file['load_checkpoint_path'],
                accumulate_grad_batches=config_file['gradient_accumulation_steps'],
            )
            trainer.fit(model, train_loader, dev_loader)
        elif config_file['mode'] == "test":
            ckpt = torch.load(config_file['load_checkpoint_path'])
            model.load_state_dict(ckpt['state_dict'])
            model.freeze()
            del ckpt

            random.seed(42)
            bilstm_test_n_fold(num_folds, model, config_file, cvx, wrappedTokenizor, negative_sample_prob, meta, dataset_return_type)
        else:
            print("Error: invalid training mode")
            exit()
    elif config_file['model_type'].startswith('roberta'):
        if config_file['load_checkpoint_path'] is not None:
            print("Load checkpoint model: ", config_file['load_checkpoint_path'].split('/')[-1])
            config_path = os.path.join(config_file['load_checkpoint_path'], 'config.json')
            config = RobertaConfig.from_json_file(config_path)
            pretrained_model_name_or_path = config_file['load_checkpoint_path']
            scheduler = torch.load(os.path.join(config_file['load_checkpoint_path'], 'scheduler.pt'))
            # load last learning rate of the scheduler checkpoint
            lr = scheduler['_last_lr'][0]
        else:
            config = AutoConfig.from_pretrained(
                config_file['pretrain_name'],
                is_decoder=False,  # disable causal encoding
            )
            pretrained_model_name_or_path = config_file['pretrain_name']
        config.update(
            {'num_labels': num_labels,
             'num_meta': num_meta,
             'enable_meta_year': config_file['enable_meta_year'],
             'enable_meta_issarea': config_file['enable_meta_issarea'],
             'enable_meta_judge': config_file['enable_meta_judge'],
             'task': config_file['task'],
             'run_config': config_file
             }
        )
        model = RobertaForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            from_tf=False,
            config=config
        )
        if config_file['judge_embedding_export_file']:
            judge_embeddings = torch.clone(model.embedding.weight)
            je = judge_embeddings.detach().numpy()
            np.save(config_file['judge_embedding_export_file'],
                    je,
                    allow_pickle=False)
            print('saved judge embedding model')

        # change embedding size for vocabulary expansion
        model.roberta.resize_token_embeddings(len(wrappedTokenizor))

        training_args = TrainingArguments(
            output_dir=config_file['output_dir'],
            evaluation_strategy='epoch',
            do_train=True,
            do_eval=True,
            fp16=False,
            save_steps=200,
            per_device_train_batch_size=config_file['batch_size'],
            per_device_eval_batch_size=config_file['batch_size'],
            logging_first_step=False,
            logging_steps=9,
            learning_rate=lr,
            save_total_limit=5,
            dataloader_num_workers=2,
            gradient_accumulation_steps=config_file['gradient_accumulation_steps'],
            num_train_epochs=8,
            load_best_model_at_end = True
        )

        writer = SummaryWriter() if config_file['mode'] == "train" else None
        if config_file['task'].startswith("cit_class"):
            evaluate_metrics = roberta_evaluate_class
        elif config_file['task'].startswith("cit_idx"):
            evaluate_metrics = roberta_evaluate_idx
        else:
            evaluate_metrics = None
        no_decay = ["bias", "LayerNorm.weight"]
        # little hack to disable huggingface default scheduler, weight decay is off by defaul
        optimizer_grouped_parameters = [
                {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                            "weight_decay": 0.0,
                },
                {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                         "weight_decay": 0.0,
                },
                ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            #tb_writer=writer,
            compute_metrics=evaluate_metrics,
            optimizers=(optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.))
        )

        if config_file['mode'] == "train":
            trainer.train()
            trainer.save_model()
        elif config_file['mode'] in ['test', 'analysis']:
            if config_file['input_text'] is not None:
                generate_citation(config_file['input_text'], config_file['input_meta'])
                exit()
            random.seed(42)
            ## export prediction analysis data
            if ((config_file['predictions_analysis_file'] is not None)
                and config_file['task'] == 'cit_idx_predictions'):
                with open(config_file['predictions_analysis_file'], 'w') as f:
                    f.write('year;issarea;judge;label;pred;position;context;forecast;preds\n')
            ## test: produce performance metrics on test data
            if config_file['mode'] == 'test':
                roberta_test_n_fold(num_folds, trainer, config_file, cvx, wrappedTokenizor, negative_sample_prob, meta, dataset_return_type)
            ## analysis: generate further model analysis data
            elif config_file['mode'] == 'analysis':
                roberta_model_analysis_val_data(trainer, config_file, cvx, wrappedTokenizor, negative_sample_prob, meta, dataset_return_type)
        else:  #eval mode
            random.seed(42)
            predOutput = trainer.evaluate(dev_dataset)
            print("metrics: ", predOutput)
    else:
        print("Error: invalid model name")
        exit()

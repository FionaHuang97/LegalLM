# superset_dir = '../../../data/bva/preprocessed-cached-v3/'
# subset_dir = '../../../data/bva/preprocessed-cached-debug/'
# n = 0
# diff = 0
# for fname in os.listdir(subset_dir):
#     superset_fpath = os.path.join(superset_dir, fname)
#     subset_fpath = os.path.join(subset_dir, fname)
#     with open(superset_fpath) as f:
#         superset_data = json.load(f)
#         superset_cit_indices = superset_data['citation_indices']
#         citation_texts = superset_data['citation_texts']
#     with open(subset_fpath) as f:
#         subset_data = json.load(f)
#         subset_cit_indices = subset_data['citation_indices']
#     if len(superset_cit_indices) != len(citation_texts):
#         print(fname)
#         print(citation_texts)
#         print(f'superset mismatch: {superset_cit_indices}')
#         assert False
#     if len(subset_cit_indices) != len(citation_texts):
#         print(fname)
#         print(citation_texts)
#         print(f'subset mismatch: {subset_cit_indices}')
#         assert False
#     for i in range(len(citation_texts)):
#         if superset_cit_indices[i] != subset_cit_indices[i]:
#             print(fname)
#             print(i)
#             print(citation_texts[i])
#             print('superset: '+str(superset_cit_indices[i]))
#             print('subset: '+str(subset_cit_indices[i]))
#             print()
#             diff += 1
#     n += 1
# print(f'n: {n}')
# print(f'diff: {diff}')

# ===

import dataset_build as db
import dataset_vocab as dv
import pickle
import copy
import functools
import argparse
import importlib as imp


cv_raw_fpath = '../../../data/bva/vocab/vocab_raw_v4.pkl'
cv_norm_fpath = '../../../data/bva/vocab/vocab_norm_v4.pkl'
cv_norm_min20_fpath = '../../../data/bva/vocab/vocab_norm_min20_v4.pkl'


def save_vocab(cv, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(cv, f)

def load_vocab(fpath):
    with open(fpath, 'rb') as f:
        cv = pickle.load(f)
    return cv


def build_vocab():
    print('building raw vocabulary')
    cv = db.make_raw_citation_vocab(db.preprocessed_dir, db.train_ids_fpath)
    print(cv.vocab_report())
    save_vocab(cv, cv_raw_fpath)
    print('normalizing vocabulary')
    cv_norm = db.normalize_vocabulary(cv, db.citation_dict_fpaths)
    print(cv_norm.vocab_report())
    save_vocab(cv_norm, cv_norm_fpath)
    print('thresholding vocabulary')
    cv_norm_min20 = copy.deepcopy(cv_norm)
    cv_norm_min20.reduce_sparse_to_unknown(20)
    save_vocab(cv_norm_min20, cv_norm_min20_fpath)
    print(cv_norm_min20.vocab_report())
    return cv, cv_norm, cv_norm_min20


def load_vocabs():
    print('loading raw vocabulary')
    cv = load_vocab(cv_raw_fpath)
    print('loading normalized vocabulary')
    cv_norm = load_vocab(cv_norm_fpath)
    print('loading normalized, thresholded vocabulary')
    cv_norm_min20 = load_vocab(cv_norm_min20_fpath)
    return cv, cv_norm, cv_norm_min20


def make_cache(vocab):
    print('caching preprocessed decisions')
    db.parallelize_cache_citation_indices('../../../data/bva/utils/updated_ids_all.txt',
            db.preprocessed_dir,
            '../../../data/bva/preprocessed-cached-v4/',
            vocab)

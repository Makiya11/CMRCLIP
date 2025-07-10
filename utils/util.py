import functools
import json
import os
import socket
import time
import torch
from collections import OrderedDict
from datetime import datetime
from itertools import repeat
from pathlib import Path

import humanize
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import roc_curve

def replace_nested_dict_item(obj, key, replace_value):
    for k, v in obj.items():
        if isinstance(v, dict):
            obj[k] = replace_nested_dict_item(v, key, replace_value)
    if key in obj:
        obj[key] = replace_value
    return obj


def state_dict_data_parallel_fix(load_state_dict, curr_state_dict):
    load_keys = list(load_state_dict.keys())
    curr_keys = list(curr_state_dict.keys())

    redo_dp = False
    undo_dp = False
    if not curr_keys[0].startswith('module.') and load_keys[0].startswith('module.'):
        undo_dp = True
    elif curr_keys[0].startswith('module.') and not load_keys[0].startswith('module.'):
        redo_dp = True

    if undo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
    elif redo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = 'module.' + k  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = load_state_dict
    return new_state_dict

def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array
    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def memory_summary():
    vmem = psutil.virtual_memory()
    msg = (
        f">>> Currently using {vmem.percent}% of system memory "
        f"{humanize.naturalsize(vmem.used)}/{humanize.naturalsize(vmem.available)}"
    )
    print(msg)

@functools.lru_cache(maxsize=64, typed=False)
def memcache(path):
    suffix = Path(path).suffix
    print(f"loading features >>>", end=" ")
    tic = time.time()
    if suffix == ".npy":
        res = np_loader(path)
    else:
        raise ValueError(f"unknown suffix: {suffix} for path {path}")
    print(f"[Total: {time.time() - tic:.1f}s] ({socket.gethostname() + ':' + str(path)})")
    return res

def np_loader(np_path, l2norm=False):
    with open(np_path, "rb") as f:
        data = np.load(f, encoding="latin1", allow_pickle=True)
    if isinstance(data, np.ndarray) and data.size == 1:
        data = data[()]  # handle numpy dict storage convnetion
    if l2norm:
        print("L2 normalizing features")
        if isinstance(data, dict):
            for key in data:
                feats_ = data[key]
                feats_ = feats_ / max(np.linalg.norm(feats_), 1E-6)
                data[key] = feats_
        elif data.ndim == 2:
            data_norm = np.linalg.norm(data, axis=1)
            data = data / np.maximum(data_norm.reshape(-1, 1), 1E-6)
        else:
            raise ValueError("unexpected data format {}".format(type(data)))
    return data

########################################################
import re
import nltk

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
STOPWORDS_TO_REMOVE = [ 'aren',
                        "aren't",
                        'ain',
                        'couldn',
                        "couldn't",
                        'didn',
                        "didn't",
                        'doesn',
                        "doesn't",
                        'don',
                        "don't",
                        'hadn',
                        "hadn't",
                        'hasn',
                        "hasn't",
                        'haven',
                        "haven't",
                        'isn',
                        "isn't",
                        'mightn',
                        "mightn't",
                        'mustn',
                        "mustn't",
                        'needn',
                        "needn't",
                        'no',
                        'nor',
                        'not',
                        'shan',
                        "shan't",
                        'shouldn',
                        "shouldn't",
                        'wasn',
                        "wasn't",
                        'weren',
                        "weren't",
                        'won',
                        "won't",
                        'wouldn',
                        "wouldn't",
                        ]
for word in STOPWORDS_TO_REMOVE:
    STOPWORDS.remove(word)

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
UNITS = {'cm', 'cc', 'beat', 'm2', 'l', 'gm', 's', 'min', 'mmol', 'hg', 'mm', 'ml', 'sq', 'm', 'sec', 'ms'}
QUANTS = {'lv edvi', 'lv ef', 'edv', 'edvi', 'esv', 'esvi', 'svi', 'lvef',  'lvmi', 'rvef', 'ef', 'sv', 'co', 'ci', 'ce'}
QUANTS_PHRASE = {'stroke volume', 'cardiac index', 'lv mass', 'cardiac output', 'forward volume', 
                'reverse volume', 'net forward volume', 'aortic regurgitant fraction', 'quantitative mitral regurgitant volume',
                'quantitative mitral regurgitant fraction'}


def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = text.replace('result:', '')
    text = text.replace('impression:', '')
    text = text.replace('overall', '')
    text = text.replace('\n', ' ') # remove new line
    text = text.replace('\t', ' ') # remove new line
    text = re.sub("[\(\[].*?[\)\]]", "", text) # remove texts () and []

    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = ' '.join(word for word in text.split() if word not in UNITS) # remove units from text
    text = ' '.join(word for word in text.split() if word not in QUANTS) # remove quantitative measures from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwords from text

    for phrase in QUANTS_PHRASE:
        text = text.replace(phrase, ' ')

    text = re.sub(r'[0-9]', '', text) # remove digits
    text = ' '.join(text.split()) # remove redundant spaces

    return text


def clean_data(df):
    # for findings
    MIN_WORDS_COUNT = 100
    MAX_WORDS_COUNT = 300

    df_cleaned = df.copy()
    df_cleaned['findings'] = df_cleaned['findings'].apply(clean_text)
    df_cleaned['findings_word_counts'] = df_cleaned['findings'].str.split().apply(len)

    # limit the length of the text
    df_cleaned = df_cleaned[(df_cleaned['findings_word_counts'] >= MIN_WORDS_COUNT) & 
                                (df_cleaned['findings_word_counts'] <= MAX_WORDS_COUNT)]

    # remove the multiple label
    df_cleaned = df_cleaned[df_cleaned['diagnosis'] != 'multiple']
    # remove the none label
    df_cleaned = df_cleaned[df_cleaned['diagnosis'] != 'none']

    df_cleaned.info()
    # df_cleaned.head(5)

    return df_cleaned


def clean_data_umls(df, labels_to_keep, single_label=False):
    # for impression
    MIN_WORDS_COUNT = 50
    MAX_WORDS_COUNT = 200

    df_cleaned = df.copy()
    df_cleaned.impressions = df_cleaned.impressions.astype(str)

    # remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates(subset='findings', keep="first")

    df_cleaned['findings'] = df_cleaned['findings'].apply(clean_text)

    # compute length of sequence
    df_cleaned['findings_word_counts'] = df_cleaned['findings'].str.split().apply(len)

    # limit the length of the text
    df_cleaned = df_cleaned[(df_cleaned['findings_word_counts'] >= MIN_WORDS_COUNT) & 
                                (df_cleaned['findings_word_counts'] <= MAX_WORDS_COUNT)]

    # only keeps findings and specified labels
    df_cleaned = df_cleaned[['impressions', 'findings', 'findings_word_counts'] + labels_to_keep]

    # compute number of labels for each sample
    df_cleaned['num_labels'] = df_cleaned[labels_to_keep].sum(axis=1)

    # only keep single label data
    if single_label:
        df_cleaned = df_cleaned[df_cleaned['num_labels'] == 1]

    # reset index
    df_cleaned = df_cleaned.reset_index(drop=True)

    df_cleaned.info()

    return df_cleaned

def clean_data_impr_sent(df, single_label=False):
    # for impression sentence
    MIN_WORDS_COUNT = 5
    MAX_WORDS_COUNT = 30

    df_cleaned = df.copy()
    df_cleaned.impr_sent = df_cleaned.impr_sent.astype(str)

    # remove duplicate rows
    # df_cleaned = df_cleaned.drop_duplicates(subset='impressions', keep="first")

    df_cleaned['impr_sent'] = df_cleaned['impr_sent'].apply(clean_text)

    # compute length of sequence
    df_cleaned['impr_sent_length'] = df_cleaned['impr_sent'].str.split().apply(len)

    # limit the length of the text
    df_cleaned = df_cleaned[(df_cleaned['impr_sent_length'] >= MIN_WORDS_COUNT) & 
                                (df_cleaned['impr_sent_length'] <= MAX_WORDS_COUNT)]

    # only keeps findings and specified labels
    # df_cleaned = df_cleaned[['impressions', 'findings', 'findings_word_counts'] + labels_to_keep]

    # compute number of labels for each sample
    # df_cleaned['num_view_labels'] = df_cleaned[['view_cine_lax', 'view_cine_3ch', 'view_cine_av',
                                            # 'view_cine_mitral', 'view_lge_lax', 'view_cine_sax', 'view_lge_sax']].sum(axis=1)

    # only keep single label data
    if single_label:
        df_cleaned = df_cleaned[df_cleaned['num_labels'] == 1]

    # reset index
    df_cleaned = df_cleaned.reset_index(drop=True)

    df_cleaned.info()

    return df_cleaned



def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    
    # Scale similarity from [-1, 1] to [0, 1]
    scaled_sim = (sim_mt + 1) / 2
    return scaled_sim

################################################################################################################

class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    """
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        sdata.append((d-y1) / (y2-y1) 
                     * (x2 - x1) + x1)
    return sdata

class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=6):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.9,0.9],polar=True,
                label = "axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, labels=variables, fontsize=25)
        
        # [txt.set_rotation(angle-90) for txt, angle 
        #      in zip(text, angles)]
        
        for txt, angle in zip(text, angles):
            # txt.set_rotation(angle-90) # TODO: doesn't work
            txt.set_position((-0.25,-0.25)) # move labels outward
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], 
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2)) 
                         for x in grid]
            # if ranges[i][0] > ranges[i][1]:
            #     grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            ax.set_rgrids(grid, labels=gridlabel,
                         angle=angles[i], fontsize=15)
            #ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]

    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        l = self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
        return l

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
        


def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold']) 
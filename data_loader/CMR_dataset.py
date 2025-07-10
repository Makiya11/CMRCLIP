import json
import os
import random
import ast

import numpy as np
import pandas as pd

from base.base_dataset import TextVideoDataset


class CMR(TextVideoDataset):
    def _load_metadata(self):
        print('Loading CMR metadata...', self.metadata_dir)
        
        df = pd.read_json(os.path.join(self.metadata_dir, 'annotation', 'CMR.json'))
        print('split: ', self.split)
        split_dir = os.path.join(self.metadata_dir, 'split')
        js_test_cap_idx_path = None
        challenge_splits = {"val", "public_server_val", "public_server_test"}
        
        try:
            train_df = pd.read_csv('data/train_list_CMR.txt', names=['videoid'])
            val_df = pd.read_csv('data/val_list_CMR.txt', names=['videoid'])          
            test_df = pd.read_csv('data/test_list_CMR.txt', names=['videoid'])
            self.split_sizes = {'train': len(train_df), 'val': len(val_df), 'test': len(test_df)}
        except:
            print('Testing Mode')
        
        if self.split == 'train':
            df = df[df['image_id'].isin(train_df['videoid'])]
        elif self.split == 'val':
            df = df[df['image_id'].isin(val_df['videoid'])]
        elif self.split == 'test':
            df = df[df['image_id'].isin(test_df['videoid'])]

        self.metadata = df.groupby(['image_id'])['caption'].apply(list)
        if self.subsample < 1:
            self.metadata = self.metadata.sample(frac=self.subsample)

        # use specific caption idx's in jsfusion
        if js_test_cap_idx_path is not None and self.split != 'train':
            caps = pd.Series(np.load(os.path.join(split_dir, js_test_cap_idx_path), allow_pickle=True))
            new_res = pd.DataFrame({'caps': self.metadata, 'cap_idx': caps})
            new_res['test_caps'] = new_res.apply(lambda x: [x['caps'][x['cap_idx']]], axis=1)
            self.metadata = new_res['test_caps']

        self.metadata = pd.DataFrame({'captions': self.metadata})

    def _get_video_path(self, sample):
        return os.path.join(self.data_dir, 'videos', 'all', sample.name + '.mp4'), sample.name + '.mp4'

    def _get_caption(self, sample):
        caption_sample = self.text_params.get('caption_sample', "rand")
        if self.split in ['train', 'val'] and caption_sample == "rand":
            caption = random.choice(sample['captions'])
        else:
            caption = sample['captions'][0]
        return caption
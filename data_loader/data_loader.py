from base import BaseDataLoaderExplicitSplit, BaseMultiDataLoader
from data_loader.transforms import init_transform_dict
from data_loader.CMR_dataset import CMR

def dataset_loader(dataset_name,
                   text_params,
                   video_params,
                   data_dir,
                   metadata_dir=None,
                   split='train',
                   tsfms=None,
                   cut=None,
                   subsample=1,
                   sliding_window_stride=-1,
                   reader='decord',
                   ):
    kwargs = dict(
        dataset_name=dataset_name,
        text_params=text_params,
        video_params=video_params,
        data_dir=data_dir,
        metadata_dir=metadata_dir,
        split=split,
        tsfms=tsfms,
        cut=cut,
        subsample=subsample,
        sliding_window_stride=sliding_window_stride,
        reader=reader,
    )
    if dataset_name == "CMR":
        dataset = CMR(**kwargs)    
    # elif dataset_name == "CMR_NPY":
    #     dataset = CMR_NPY(**kwargs)        
    else:
        raise NotImplementedError(f"Dataset: {dataset_name} not found.")
    return dataset


class TextVideoDataLoader(BaseDataLoaderExplicitSplit):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 tsfm_split=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='decord',
                 batch_size=1,
                 num_workers=1,
                 prefetch_factor=2,
                 shuffle=True,
                 val_batch_size=None):
        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict(**tsfm_params)

        if tsfm_split is None:
            tsfm_split = split
        tsfm = tsfm_dict[tsfm_split]
        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, metadata_dir, split, tsfm, cut,
                                 subsample, sliding_window_stride, reader)
        if val_batch_size is not None and split == 'val':
            batch_size = val_batch_size

        super().__init__(dataset, batch_size, shuffle, num_workers, prefetch_factor=prefetch_factor)
        self.dataset_name = dataset_name


class TextVideoMultiDataLoader(BaseMultiDataLoader):
    # TODO: figure out neat way to have N data_loaders
    # TODO: also add N weighted sampler
    def __init__(self, data_loader1, data_loader2):
        # get class from "type" in dict
        dls_cfg = [data_loader1, data_loader2]
        dls = []
        for dcfg in dls_cfg:
            dl = globals()[dcfg['type']](**dcfg['args'])
            dls.append(dl)
        super().__init__(dls)


if __name__ == "__main__":    
    kwargs = {
            "dataset_name": "CMR",
            "data_dir": "/data/aiiih/projects/nakashm2/multimodal/Video_text_retrieval/data/CMR_cine_lge_FL/annotation/CMR.json",
            "shuffle": True,
            "num_workers": 16,
            "batch_size": 16,
            "split": "train",
            "cut": "chall",
            "subsample": 1,
            "text_params": {
                "input": "text",
                "caption_replace_prob": 0.5
            },
            "video_params": {
                "extraction_fps": 25,
                "extraction_res": 256,
                "input_res": 224,
                "num_frames": 4,
                "shot_replace": True,
                "shot_replace_prob": 0.5
            },
            "tsfm_params": {"auto_aug": True}
        }
    
    dl = TextVideoDataLoader(**kwargs)
    dl.dataset.__getitem__(0)
    for x in range(1000):
        res = next(iter(dl))
        print(x)
        breakpoint()
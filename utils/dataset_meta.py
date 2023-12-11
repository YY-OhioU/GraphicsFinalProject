import torch


class DatasetMeta:
    def __init__(self):
        self.K = None
        self.img_wh = None
        self.poses = None

    def load_data(self, fp):
        info_dict = torch.load(fp)
        self.K = info_dict['k']
        self.img_wh = info_dict['wh']
        self.poses = info_dict['poses']

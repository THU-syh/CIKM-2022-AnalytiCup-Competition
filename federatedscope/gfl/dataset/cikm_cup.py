import logging

import torch
import os
from torch_geometric.data import InMemoryDataset

logger = logging.getLogger(__name__)

class CIKMCUPDataset(InMemoryDataset):
    name = 'CIKM22Competition'
    inmemory_data = {}

    def __init__(self, root):
        super(CIKMCUPDataset, self).__init__(root)

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name)

    @property
    def processed_file_names(self):
        return ['pre_transform.pt', 'pre_filter.pt']

    def __len__(self):
        return len([
            x for x in os.listdir(self.processed_dir)
            if not x.startswith('pre')
        ])

    def _load(self, idx, split):
        try:
            data = torch.load(
                os.path.join(self.processed_dir, str(idx), f'{split}.pt'))
        except:
            data = None
        return data

    def process(self):
        pass

    def __getitem__(self, idx):
        if idx in self.inmemory_data:
            return self.inmemory_data[idx]
        else:
            self.inmemory_data[idx] = {}
            for split in ['train', 'val', 'test']:
                split_data = self._load(idx, split)
                if split_data:
                    self.inmemory_data[idx][split] = split_data
            return self.inmemory_data[idx]


def load_cikmcup_data(config,client_cfg):
    from torch_geometric.loader import DataLoader

    # Build data
    logger.info(f'Loading CIKMCUP data from {os.path.abspath(os.path.join(config.data.root, "CIKM22Competition"))}.')
    dataset = CIKMCUPDataset(config.data.root)
    # config.merge_from_list(['federate.client_num', len(dataset)])

    if len(dataset) == 0:
        raise FileNotFoundError(f'Cannot load CIKMCUP data from {os.path.abspath(os.path.join(config.data.root, "CIKM22Competition"))}, please check if the directory is correct.')

    data_dict = {}
    # Build DataLoader dict
    for client_idx in config.federate.clients_id:
        logger.info(f'Loading CIKMCUP data for Client #{client_idx}.')
        dataloader_dict = {}
        tmp_dataset = []
        try:
            batch_size = client_cfg.get('client_{}'.format(client_idx)).data.batch_size
        except:
            batch_size = config.data.batch_size
        if 'train' in dataset[client_idx]:
            dataloader_dict['train'] = DataLoader(dataset[client_idx]['train'],
                                                  batch_size,
                                                  shuffle=config.data.shuffle)
            tmp_dataset += dataset[client_idx]['train']
        if 'val' in dataset[client_idx]:
            dataloader_dict['val'] = DataLoader(dataset[client_idx]['val'],
                                                batch_size,
                                                shuffle=False)
            tmp_dataset += dataset[client_idx]['val']
        if 'test' in dataset[client_idx]:
            dataloader_dict['test'] = DataLoader(dataset[client_idx]['test'],
                                                 batch_size,
                                                 shuffle=False)
            tmp_dataset += dataset[client_idx]['test']
        if tmp_dataset:
            dataloader_dict['num_label'] = 0

        data_dict[client_idx] = dataloader_dict

    return data_dict, config

import torch
import numpy as np
from torch.utils.data import Dataset

class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        """初始化带近邻采样能力的数据集包装器。"""
        super(NeighborsDataset, self).__init__()
        self.internal_dataset = dataset
        self.indices = indices  # 近邻索引矩阵，形状为 [len(dataset), k]。
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        assert(self.indices.shape[0] == len(self.internal_dataset))

    def __len__(self):
        """返回数据集样本数量。"""
        return len(self.internal_dataset)

    def __getitem__(self, index):
        """返回锚点样本及其随机近邻样本。"""
        output = {}
        anchor = list(self.internal_dataset.__getitem__(index))
        
        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.internal_dataset.__getitem__(neighbor_index)

        output['anchor'] = anchor
        output['neighbor'] = neighbor
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = anchor[3]
        output['index'] = index
        return output

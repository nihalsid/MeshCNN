import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
from models.layers.mesh import Mesh


class TextureData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase, "meshes")
        self.paths = self.make_dataset(self.dir, opt)
        self.size = len(self.paths)
        self.get_mean_std()
        # # modify for network later.
        opt.input_nc = self.ninput_channels
        opt.nclasses = 3

    def __getitem__(self, index):
        path = self.paths[index]
        mesh = Mesh(file=path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
        meta = {}
        meta['mesh'] = mesh
        meta['label'] = mesh.edge_target_colors.T
        # get edge features
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.opt.ninput_edges)
        # todo: dont mean the input color flag or the input colors
        meta['edge_features'] = edge_features
        if not type(self.mean) == int:
            n_feats = meta['edge_features'].shape[0]
            meta['edge_features'][:n_feats - 4, :] = (edge_features[:n_feats - 4, :] - self.mean[:n_feats - 4, :]) / self.std[:n_feats - 4, :]

        return meta

    def __len__(self):
        return self.size

    @staticmethod
    def make_dataset(path, opt):
        meshes = []
        assert os.path.isdir(path), '%s is not a valid directory' % path
        if opt.use_single_view:
            angle_list = [(180, 45)]
        else:
            angle_list = [(x, y) for x in range(225, 60, -45) for y in range(0, 360, 45)]
        for d in sorted(os.listdir(path)):
            for a in angle_list:
                meshes.append(os.path.join(path, d, f"model_normalized_input_{a[0]:03d}_{a[1]:03d}.obj"))
        return meshes

import torch.nn as nn
import torch.nn.functional as F

class QEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx=None,
                 max_norm=None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight=None, qconfig={}):
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
                         scale_grad_by_freq, sparse, _weight)
        self.set_qconfig(qconfig)
        self.idx = None
        self.param_added = False

    def set_qconfig(self, qconfig):
        self.weight_quantizer = qconfig['weight'](shape=self.weight.shape, axis=0)

    def forward(self, input):
        return F.embedding(input, self.weight_quantizer(self.weight),
                                                         self.padding_idx, self.max_norm, self.norm_type,
                                                         self.scale_grad_by_freq, self.sparse)

    @classmethod
    def from_float(cls, mod, qconfig=None, param_list={}):
        assert type(mod) == nn.Embedding
        embedding = cls(mod.num_embeddings, mod.embedding_dim, mod.padding_idx, mod.max_norm, mod.norm_type,
                  mod.scale_grad_by_freq, mod.sparse, mod.weight, qconfig)
        return embedding

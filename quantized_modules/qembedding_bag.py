import torch.nn as nn
import torch.nn.functional as F


class QEmbeddingBag(nn.EmbeddingBag):
    def __init__(self, num_embeddings, embedding_dim, max_norm=None, norm_type=2.0,
                 scale_grad_by_freq=False, mode='mean', sparse=False, _weight=None,
                 include_last_offset=False, qconfig={}):
        super().__init__(num_embeddings, embedding_dim, max_norm, norm_type,
                         scale_grad_by_freq, mode, sparse, _weight, include_last_offset)
        self.set_qconfig(qconfig)
        self.idx = None
        self.param_added = False

    def set_qconfig(self, qconfig):
        self.activation_quantizer = qconfig['activation']()
        self.weight_quantizer = qconfig['weight'](shape=self.weight.shape, axis=0)

    def forward(self, input, offsets=None, per_sample_weights=None):
        return self.activation_quantizer(F.embedding_bag(input, self.weight_quantizer(self.weight), offsets,
                                                         self.max_norm, self.norm_type,
                                                         self.scale_grad_by_freq, self.mode, self.sparse,
                                                         per_sample_weights, self.include_last_offset))

    @classmethod
    def from_float(cls, mod, qconfig=None, param_list={}):
        assert type(mod) == nn.EmbeddingBag
        # NOTE: Sparse was previously hardcoded to false here because of issues
        # with the pretrained model (which said it was sparse even when the
        # weights weren't). Probably need to fix this later. I think we fixed this.
        bag = cls(mod.num_embeddings, mod.embedding_dim, mod.max_norm, mod.norm_type,
                  mod.scale_grad_by_freq, mod.mode, mod.sparse, mod.weight,
                  mod.include_last_offset, qconfig)
        return bag

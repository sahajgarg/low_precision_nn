import sys

import torch

from models.dlrm.interaction_layer import InteractionLayer


class QInteractionLayer(InteractionLayer):
    def __init__(self, arch_interaction_op, arch_interaction_itself, qconfig={}):
        super().__init__(arch_interaction_op, arch_interaction_itself)
        self.set_qconfig(qconfig)
        self.idx = None
        self.param_added = False

    def set_qconfig(self, qconfig):
        self.activation_quantizer = qconfig['activation']()
        self.weight_quantizer = qconfig['weight']()

    def forward(self, x, ly):
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return self.activation_quantizer(R)

    @classmethod
    def from_float(cls, mod, qconfig={}, param_list={}):
        layer = cls(mod.arch_interaction_op, mod.arch_interaction_itself, qconfig)
        return layer

import torch


class FloatFunctional(torch.nn.Module):
    r"""State collector class for float operatitons.
    The instance of this class can be used instead of the ``torch.`` prefix for
    some operations. See example usage below.
    .. note::
        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).
    Examples::
        >>> f_add = FloatFunctional()
        >>> a = torch.tensor(3.0)
        >>> b = torch.tensor(4.0)
        >>> f_add.add(a, b)  # Equivalent to ``torch.add(a, b)``
    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """

    def __init__(self):
        super(FloatFunctional, self).__init__()
        self.activation_quantizer = torch.nn.Identity()
        self.idx = None
        self.param_added = False

    @classmethod
    def from_float(cls, mod, qconfig={}, param_list={}):
        functional = cls()
        functional.quant_gemm_only = qconfig["quant_gemm_only"]
        functional.quant_relu_only = qconfig["quant_relu_only"]
        if functional.quant_gemm_only:
            functional.activation_quantizer = torch.nn.Identity()
        else:
            functional.activation_quantizer = qconfig['activation']()
        return functional

    def forward(self, x):
        raise RuntimeError("FloatFunctional is not intended to use the " +
                           "'forward'. Please use the underlying operation")

    r"""Operation equivalent to ``torch.add(Tensor, Tensor)``"""

    def add(self, x, y):
        r = torch.add(x, y)
        if not self.quant_relu_only:
            r = self.activation_quantizer(r)
        return r

    r"""Operation equivalent to ``torch.add(Tensor, float)``"""

    def add_scalar(self, x, y):
        r = torch.add(x, y)
        if not self.quant_relu_only:
            r = self.activation_quantizer(r)
        return r

    r"""Operation equivalent to ``torch.mul(Tensor, Tensor)``"""

    def mul(self, x, y):
        r = torch.mul(x, y)
        if not self.quant_relu_only:
            r = self.activation_quantizer(r)
        return r

    r"""Operation equivalent to ``torch.mul(Tensor, float)``"""

    def mul_scalar(self, x, y):
        r = torch.mul(x, y)
        if not self.quant_relu_only:
            r = self.activation_quantizer(r)
        return r

    r"""Operation equivalent to ``torch.cat``"""

    def cat(self, x, dim=0):
        r = torch.cat(x, dim=dim)
        if not self.quant_relu_only:
            r = self.activation_quantizer(r)
        return r

    r"""Operation equivalent to ``relu(torch.add(x,y))``"""

    def add_relu(self, x, y):
        r = torch.add(x, y)
        r = torch.nn.functional.relu(r)
        r = self.activation_quantizer(r)
        return r

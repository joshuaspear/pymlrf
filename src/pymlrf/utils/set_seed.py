pkg_imports = []

__all__ = ["set_seed"]

try:
    import numpy
    pkg_imports.append("numpy")
except ImportError as e:
    pass

try:
    import random
    pkg_imports.append("random")
except ImportError as e:
    pass

try:
    import torch
    pkg_imports.append("torch")
    torch.backends.cudnn.deterministic = True
except ImportError as e:
    pass

try:
    import d3rlpy
    pkg_imports.append("d3rlpy")
except ImportError as e:
    pass

pkg_seed_lkp = {
    "numpy": [numpy.random.seed],
    "random": [random.seed],
    "torch": [
        torch.manual_seed,
        torch.cuda.manual_seed
        ],
    "d3rlpy": d3rlpy.seed
}

if "d3rlpy" in pkg_imports:
    pkg_imports = ["d3rlpy"]


def set_seed(n: int) -> None:
    
    for i in pkg_imports:
        for seed_func in pkg_seed_lkp[i]:
            seed_func(n)

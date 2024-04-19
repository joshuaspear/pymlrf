pkg_imports = []

__all__ = ["set_seed"]


pkg_seed_lkp = {}

try:
    import numpy
    pkg_imports.append("numpy")
    pkg_seed_lkp.update({"numpy": [numpy.random.seed]})
except ImportError as e:
    pass

try:
    import random
    pkg_imports.append("random")
    pkg_seed_lkp.update({"random": [random.seed]})
except ImportError as e:
    pass

try:
    import torch
    pkg_imports.append("torch")
    torch.backends.cudnn.deterministic = True
    pkg_seed_lkp.update({
        "torch": [
        torch.manual_seed,
        torch.cuda.manual_seed
        ]
    })
except ImportError as e:
    pass

try:
    import d3rlpy
    pkg_imports.append("d3rlpy")
    pkg_seed_lkp.update({"d3rlpy":[d3rlpy.seed]})
except ImportError as e:
    pass

if "d3rlpy" in pkg_imports:
    pkg_imports = ["d3rlpy"]

def set_seed(n: int) -> None:
    
    for i in pkg_imports:
        for seed_func in pkg_seed_lkp[i]:
            seed_func(n)

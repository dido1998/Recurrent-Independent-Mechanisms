from torch import nn, Tensor, torch
from typing import List, Union
from numpy import ndarray


class Device:
    """Utilities for handling devices
    """
    def __init__(self, use_cpu: bool = False, gpu_indices: List[int] = []) -> None:
        """
        :param gpu_limits: list of gpus you allow PyTorch to use
        """
        super().__init__()
        if use_cpu or not torch.cuda.is_available():
            self.__use_cpu()
        else:
            self.gpu_indices = gpu_indices if gpu_indices else self.__all_gpu()
            self.device = torch.device('cuda:{}'.format(self.gpu_indices[0]))

    @property
    def unwrapped(self) -> torch.device:
        return self.device

    def tensor(self, arr: Union[ndarray, List[ndarray], Tensor], dtype=torch.float32) -> Tensor:
        """Convert numpy array or Tensor into Tensor on main_device
        :param x: ndarray or Tensor you want to convert
        :return: Tensor
        """
        t = type(arr)
        if t is Tensor:
            return arr.to(device=self.device)  # type: ignore
        elif t is ndarray or t is list:
            return torch.tensor(arr, device=self.device, dtype=dtype)
        else:
            raise ValueError('arr must be ndarray or list or tensor')

    def zeros(self, size: Union[int, tuple]) -> Tensor:
        return torch.zeros(size, device=self.device)

    def ones(self, size: Union[int, tuple]) -> Tensor:
        return torch.ones(size, device=self.device)

    def data_parallel(self, module: nn.Module) -> nn.DataParallel:
        return nn.DataParallel(module, device_ids=self.gpu_indices)

    def is_multi_gpu(self) -> bool:
        return len(self.gpu_indices) > 1

    def __all_gpu(self) -> List[int]:
        return list(range(torch.cuda.device_count()))

    def __use_cpu(self) -> None:
        self.gpu_indices = []
        self.device = torch.device('cpu')

    def __repr__(self) -> str:
        return str(self.device)

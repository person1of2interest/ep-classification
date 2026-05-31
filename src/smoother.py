import abc
from typing import List
import yaml
from utils import CombinedMeta
from dataclasses import dataclass

import numpy as np
import pandas as pd

from joblib import Parallel, delayed


def moving_avg(x: List[float], window_size: int = 1) -> np.ndarray:
    """
    Считаем простое скользящее среднее

    Параметры:
        x - временной ряд, для которого считаем скользящее среднее
        window_size - размер окна усреднения
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    arr = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
    return arr


class Smoother(abc.ABC, yaml.YAMLObject, metaclass=CombinedMeta):
    def set_param(self, param_name: str, value: str | int):
        """
        Инициализируем параметр класса

        Параметры:
            param_name - параметр
            value - значение
        """
        if hasattr(self, param_name):
            setattr(self, param_name, value)
        else:
            raise AttributeError(f"{param_name} не является допустимым параметром")

    @abc.abstractmethod
    def smoothe(self, epochs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        pass    


@dataclass
class ChannelSmoother(Smoother, yaml.YAMLObject):
    yaml_tag = u'!ChannelSmoother'
    yaml_loader = yaml.SafeLoader

    smoothing_channels: List[str] | str = 'all'
    window_size: int = 1
    normalization: bool = False
    
    def __repr__(self):
        s = (
            f"{self.__class__.__name__}"
            f"(smoothing_channels={self.smoothing_channels}, "
            f"window_size={self.window_size}, "
            f"normalization={self.normalization})"
        )
        return s

    def __smoothe_epoch(self, epoch: pd.DataFrame) -> pd.DataFrame:
        """
        Сглаживаем отдельные каналы в эпохе с помощью
        скользящего среднего, перед эти опционально нормализуем
        сигналы по каналу как векторы
    
        Параметры:
            epoch - сглаживаемая эпоха
        """
        smoothed_epoch = epoch.copy()

        if self.smoothing_channels == 'all':
            self.smoothing_channels = epoch.columns.tolist()
    
        for channel in self.smoothing_channels:
            signal = epoch[channel]
            # нормализуем
            if self.normalization: 
                signal = signal / np.linalg.norm(signal)
            # вычисляем скользящее среднее
            new_signal = moving_avg(signal, self.window_size)
            # длина получившегося сигнала зависит от размера окна,
            # но она всегда меньше, чем у исходного
            m = len(signal) - len(new_signal)
            smoothed_epoch[channel] = np.concatenate(
                (new_signal, np.ones(m) * new_signal[-1])
            )
    
        return smoothed_epoch

    def smoothe(self, epochs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Сглаживаем каналы, определенные параметром 
        smoothing_channels, в эпохах
        
        Параметры:
            epochs - набор эпох для сглаживания
        """
        if self.smoothing_channels == []:
            return epochs
        else:
            smoothed_epochs = Parallel(n_jobs=4)(
                delayed(lambda x: self.__smoothe_epoch(x))(epoch) for epoch in epochs
            )
            return smoothed_epochs

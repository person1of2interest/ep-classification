import abc
from typing import List
import yaml
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from dtaidistance import dtw 

from joblib import Parallel, delayed


@dataclass
class DistanceCalculator(yaml.YAMLObject):
    """
    Класс для вычиления расстояния между эпохами
    """
    yaml_tag = u'!DistanceCalculator'
    yaml_loader = yaml.SafeLoader

    metric: str = 'correlation'

    def __repr__(self):
        return f"{self.__class__.__name__}(metric={self.metric})"
   
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

    def set_param_forced(self, param_name: str, value: List[str]):
        """
        Инициализируем параметр класса без проверки 
        на его допустимость

        Параметры:
            param_name - параметр
            value - значение
        """
        setattr(self, param_name, value)
    
    def calculate(self, epoch1: pd.DataFrame,
                  epoch2: pd.DataFrame) -> np.ndarray:
        """
        Вычисляем расстояние между эпохами;
        возвращаем вектор расстояний между
        сигналами по каждому каналу 
        (длина вектора = числу каналов)

        Параметры:
            epoch1 - первая эпоха
            epoch2 - вторая эпоха
        """
        if self.metric != 'dtw':
            dist_matrix = cdist(
                epoch1.transpose(),
                epoch2.transpose(),
                self.metric
            )
            dist_by_channel = np.diag(dist_matrix)

            if hasattr(self, 'learned_metrics'):
                learned_channels = self.channels
                learned_metrics = self.learned_metrics

                dist_by_channel = dist_by_channel.copy()
    
                for i in range(len(learned_channels)):
                    channel = learned_channels[i]
                    ind = epoch1.columns.get_loc(channel)
                    dist_by_channel[ind] = learned_metrics[i](
                        epoch1[channel].values,
                        epoch2[channel].values
                    )
                    # M_diag = learned_metrics[i]
                    # sig_diff = epoch1[channel].values - epoch2[channel].values
                    # dist_by_channel[ind] = np.sqrt(sig_diff.T @ M_diag @ sig_diff)

            if hasattr(self, 'reducer'):
                n = epoch1.shape[0] * epoch1.shape[1]
                embeddings = np.zeros((1, n))
                    
                for epoch in (epoch1, epoch2):
                    embeddings = np.vstack((
                        embeddings,
                        epoch.values.flatten()
                    ))

                embeddings = self.reducer.transform(embeddings[1:, :])
                dist = self.learned_metric(embeddings[0, :], embeddings[1, :])
                return np.array([dist] * epoch1.shape[1])

        else:
            dist_by_channel = Parallel(n_jobs=4)(
                delayed(dtw.distance_fast)(
                    epoch1[channel].values,
                    epoch2[channel].values
                )
                for channel in epoch1.columns
            )
            dist_by_channel = np.array(dist_by_channel)
        
        return dist_by_channel

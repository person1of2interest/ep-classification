import abc
import yaml
from utils import CombinedMeta

import numpy as np
import pandas as pd

from scipy.special import softmax

from trainer import Trainer


class CandidatesComparator(abc.ABC, metaclass=CombinedMeta):
    
    @abc.abstractmethod
    def compare(self, positive_distances: pd.DataFrame,
                negative_distances: pd.DataFrame,
                trainer: Trainer) -> int: #0/1
        pass


class AverageComparator(CandidatesComparator, yaml.YAMLObject):
    yaml_tag = u'!AverageComparator'
    yaml_loader = yaml.SafeLoader
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    def compare(self, positive_distances: pd.DataFrame,
                negative_distances: pd.DataFrame,
                trainer: Trainer) -> int: 
        """
        Сравниваем расстояния до позитивных и негативных 
        референтов, возвращаем лейбл ближайшего
        
        Параметры:
            positive_distances - расстояния до позитивных
            референтов
            negative_distances - расстояния до негативных
            референтов
            trainer - экземпляр класса Trainer
        """
        pos_dist = positive_distances.sum(axis=1)[0]
        neg_dist = negative_distances.sum(axis=1)[0]

        return 1 if pos_dist < neg_dist else 0


class MostCommonComparator(CandidatesComparator, yaml.YAMLObject):
    yaml_tag = u'!MostCommonComparator'
    yaml_loader = yaml.SafeLoader
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    def compare(self, positive_distances: pd.DataFrame,
                negative_distances: pd.DataFrame,
                trainer: Trainer) -> int: 
        """
        Возвращаем наиболее частый среди отобранных 
        референтов лейбл 
        
        Параметры:
            positive_distances - расстояния до позитивных
            референтов
            negative_distances - расстояния до негативных
            референтов
            trainer - экземпляр класса Trainer
        """
        n_pos = positive_distances.shape[0]
        n_neg = negative_distances.shape[0]
        
        return 1 if n_pos > n_neg else 0


class LogRegSoftmaxComparator(CandidatesComparator, yaml.YAMLObject):
    yaml_tag = u'!LogRegSoftmaxComparator'
    yaml_loader = yaml.SafeLoader
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    def compare(self, positive_distances: pd.DataFrame,
                negative_distances: pd.DataFrame, 
                trainer: Trainer) -> int: 
        """
        Объединяем расстояния до позитивных и негативных 
        референтов, переводим их в веса с помощью Softmax,
        формируем прогноз, с точки зрения каждого канала, как
        взвешенную сумму классов референтов, применяем обученную 
        логистическую регрессию на полученных эмбеддингах
        
        Параметры:
            positive_distances - расстояния до позитивных
            референтов
            negative_distances - расстояния до негативных
            референтов
            trainer - экземпляр класса Trainer
        """
        dist = (
            pd.concat(
                [positive_distances, negative_distances],
                ignore_index=True
            )
            .to_numpy()
        )

        n = len(positive_distances.columns)
        
        ref_classes = [[1]*len(positive_distances) + [0]*len(negative_distances)]
        coefs = np.repeat(ref_classes, n, axis=0).T
        preds = np.sum(softmax(-dist, axis=0) * coefs, axis=0)
        channel_preds = preds.reshape(1, -1)

        label = trainer.logreg.predict(channel_preds)[0]
        
        return label


class LogRegComparator(CandidatesComparator, yaml.YAMLObject):
    yaml_tag = u'!LogRegComparator'
    yaml_loader = yaml.SafeLoader
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    def compare(self, positive_distances: pd.DataFrame,
                negative_distances: pd.DataFrame,
                trainer: Trainer) -> int: 
        """
        Объединяем расстояния до позитивного и негативного 
        референтов, конкатенируем их в один вектор и применяем
        обученную логистическую регрессию на полученных 
        эмбеддингах
        
        Параметры:
            positive_distances - расстояния до позитивных
            референтов
            negative_distances - расстояния до негативных
            референтов
            trainer - экземпляр класса Trainer
        """
        dist = (
            pd.concat(
                [positive_distances, negative_distances],
                axis=1,
                ignore_index=True
            )
            .to_numpy()
        )
        channel_preds = dist
        
        # Z-СТАНДАРТИЗАЦИЯ
        # channel_preds = (
        #     channel_preds - trainer.mean_by_channel
        # ) / trainer.std_by_channel
        
        label = trainer.logreg.predict(channel_preds)[0]
        
        return label

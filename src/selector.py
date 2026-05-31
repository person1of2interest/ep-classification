import abc
from typing import List
import yaml
from utils import CombinedMeta
from dataclasses import dataclass

import numpy as np
import pandas as pd

from distance_calculator import DistanceCalculator


class CandidateSelector(abc.ABC, metaclass=CombinedMeta):

    @abc.abstractmethod
    def calculate(
        self, distance_calculator: DistanceCalculator,
        pos_refs: List[pd.DataFrame],
        neg_refs: List[pd.DataFrame],
        epoch: List[pd.DataFrame]) -> (pd.DataFrame, pd.DataFrame):
        pass


@dataclass
class KNNCandidateSelector(CandidateSelector, yaml.YAMLObject):
    yaml_tag = u'!KNNCandidateSelector'
    yaml_loader = yaml.SafeLoader

    k: int = 5

    def __repr__(self):
        return f"{self.__class__.__name__}(k={self.k})"

    def calculate(
        self, distance_calculator: DistanceCalculator,
        pos_refs: List[pd.DataFrame],
        neg_refs: List[pd.DataFrame],
        epoch: List[pd.DataFrame]) -> (pd.DataFrame, pd.DataFrame):
        """
        Вычисляем расстояния от эпохи до референтов;
        возвращаем расстояния до позитивных и негативных 
        среди топ-k ближайших
        
        Параметры:
            distance_calculator - экземпляр класса DistanceCalculator
            pos_refs - референты позитивных эпох
            neg_refs - референты негативных эпох
            epoch - эпоха, для которой осуществляем отбор референтов
        """ 
        dist_to_centroids = np.empty(epoch.shape[1])  
        centroids = pos_refs + neg_refs
        # рассчитываем расстояние до каждого из центроидов
        for i in range(len(centroids)):
            dist_to_centroids = np.vstack((
                dist_to_centroids,
                distance_calculator
                .calculate(
                    epoch,
                    centroids[i]
                )
            ))
        dist_to_centroids = dist_to_centroids[1:]
        dist_to_centroids = pd.DataFrame(
            dist_to_centroids,
            columns=epoch.columns
        )
        dist_to_centroids['dist'] = dist_to_centroids.sum(axis=1)
        dist_to_centroids['label'] = [1]*len(pos_refs) + [0]*len(neg_refs)
        # оставляем m наиболее близких центроидов
        indexes = (
            dist_to_centroids
            ['dist']
            .values
            .argsort()
            [:self.k]
        )
        dist_to_centroids = dist_to_centroids.iloc[indexes, :]

        dist_to_pos = (
            dist_to_centroids
            .query('label == 1')
            .drop(columns=['dist', 'label'])
            .reset_index(drop=True)
        )
        dist_to_neg = (
            dist_to_centroids
            .query('label == 0')
            .drop(columns=['dist', 'label'])
            .reset_index(drop=True)
        )
        
        return dist_to_pos, dist_to_neg


class AllCandidateSelector(CandidateSelector, yaml.YAMLObject):
    yaml_tag = u'!AllCandidateSelector'
    yaml_loader = yaml.SafeLoader

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def calculate(
        self, distance_calculator: DistanceCalculator,
        pos_refs: List[pd.DataFrame],
        neg_refs: List[pd.DataFrame],
        epoch: List[pd.DataFrame]) -> (pd.DataFrame, pd.DataFrame):
        """
        Вычисляем расстояния от эпохи до референтов;
        возвращаем расстояния до позитивных и негативных 
        (без отбора)
        
        Параметры:
            distance_calculator - экземпляр класса DistanceCalculator
            pos_refs - референты позитивных эпох
            neg_refs - референты негативных эпох
            epoch - эпоха, для которой осуществляем отбор референтов
        """ 
        n = len(epoch.columns)
        dist_to_pos = np.zeros((1, n))
        dist_to_neg = np.zeros((1, n))

        for pos_ref in pos_refs:
            dist_to_pos = np.vstack((
                dist_to_pos,
                distance_calculator
                .calculate(
                    pos_ref,
                    epoch
                )
            ))
            
        for neg_ref in neg_refs:
            dist_to_neg = np.vstack((
                dist_to_neg,
                distance_calculator
                .calculate(
                    neg_ref,
                    epoch
                )
            ))

        dist_to_pos = pd.DataFrame(dist_to_pos[1:], columns=epoch.columns)
        dist_to_neg = pd.DataFrame(dist_to_neg[1:], columns=epoch.columns)
        
        return dist_to_pos, dist_to_neg

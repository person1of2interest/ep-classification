import abc
from typing import List
import yaml
from functools import reduce

from utils import CombinedMeta
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans

from distance_calculator import DistanceCalculator

SEED = 43


class Averager(abc.ABC, metaclass=CombinedMeta):
    """
    Класс для усреднения эпох в широком смысле
    """
    @abc.abstractmethod
    def average(
        self, epochs: List[pd.DataFrame],
        distance_calculator: DistanceCalculator) -> List[pd.DataFrame]:
        pass


class NoAverager(Averager, yaml.YAMLObject):
    yaml_tag = u'!NoAverager'
    yaml_loader = yaml.SafeLoader
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    def average(
        self, epochs: List[pd.DataFrame],
        distance_calculator: DistanceCalculator):   
	"""
	Возвращаем переданные эпохи без изменений
	
	Параметры:
	     epochs - набор эпох	
	     distance_calculator - экземпляр класса 
	     DistanceCalculator
	"""   
        return epochs


@dataclass
class EpochAverager(Averager, yaml.YAMLObject):
    yaml_tag = u'!EpochAverager'
    yaml_loader = yaml.SafeLoader

    use_metric: bool = False
    max_iter: int = 100
    eps: float = 0.0001
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __sum_epochs(self, epoch1: pd.DataFrame, 
                     epoch2: pd.DataFrame) -> pd.DataFrame:
        """
        Суммируем две эпохи поканально
        и возвращаем их сумму
        
        Параметры:
             epoch1 - первая эпоха
             epoch2 - вторая эпоха
        """
        # эпохи начинаются в разное время,
        # поэтому выравниваем индексы
        res = (
            epoch1
            .reset_index(drop=True)
            .add(epoch2.reset_index(drop=True))
        )
        return res

    def __calculate_centroid(self, epochs: List[pd.DataFrame],
                             distance_calculator: DistanceCalculator,
                             initial_estimate: pd.DataFrame) -> pd.DataFrame:
        """
        Итеративно расчитываем эпоху, минимизирующую сумму квадратов 
        расстояний до всех эпох из epochs, при помощи алгоритма Визфельда
        (Weiszfeld algorithm)
        
	Параметры:
	     epochs - набор эпох	
	     distance_calculator - экземпляр класса DistanceCalculator
	     initial_estimate - начальное приближениедля алгоритма Визфельда (эпоха)
        """

        delta = 1e-8  # регуляризатор, можно настроить
        i = 0
        centroid = initial_estimate.copy()
        s = None

        while i < self.max_iter:
            centroid_new = centroid.copy() * 0
            sum_w = 0.0
            s_new = 0.0  # сумма квадратов расстояний (для критерия сходимости)

            for epoch in epochs:
                dist_by_channel = distance_calculator.calculate(centroid, epoch)
                dist_to_epoch = float(np.sum(dist_by_channel))  # скалярное расстояние

                # если точная совпадающая эпоха — вернуть её как центроид
                if dist_to_epoch < delta:
                    return epoch.copy()

                # для IRLS-подобного взвешивания берём w = 1 / d
                w_epoch = 1.0 / dist_to_epoch

                centroid_new = self.__sum_epochs(centroid_new, w_epoch * epoch)
                sum_w += w_epoch
                s_new += dist_to_epoch ** 2

            # защита от нулевой суммы весов
            if sum_w == 0:
                break

            centroid_new /= sum_w
            centroid = centroid_new.copy()

            if s is not None and abs(s_new - s) < self.eps:
                break

            s = s_new
            i += 1

        return centroid
    
    def average(
        self, epochs: List[pd.DataFrame],
        distance_calculator: DistanceCalculator) -> List[pd.DataFrame]:
	"""
	Усредняем переданные эпохи - возвращаем арифметическое среднее 
	или центроид, вычисленный с помощью алгоритма Визфельда, 
	использующего арифметическое среднее как начальное приближение
	
	Параметры:
	     epochs - набор эпох	
	     distance_calculator - экземпляр класса 
	     DistanceCalculator
	"""   
        avg_epoch = reduce(self.__sum_epochs, epochs) / len(epochs)

        if self.use_metric:
            avg_epoch = self.__calculate_centroid(epochs, distance_calculator, avg_epoch)
            
        return [avg_epoch]


class EpochClusterer(Averager, yaml.YAMLObject):
    yaml_tag = u'!EpochClusterer'
    yaml_loader = yaml.SafeLoader
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
        
    def __get_dist_matrix(
        self,
        epochs: List[pd.DataFrame],
        distance_calculator: DistanceCalculator
    ) -> np.ndarray:
        """
        Рассчитываем матрицу расстояний между эпохами
        
	Параметры:
	     epochs - набор эпох	
	     distance_calculator - экземпляр класса 
	     DistanceCalculator
        """
        s = len(epochs)
        dist_matrix = -np.empty((s, s)) # заводим матрицу
    
        # потом сгладим каналы
    
        # заполняем матрицу над главной диагональю
        for i in range(s):
            for j in range(i, s):
                dist_matrix[i][j] = np.sum(
                    distance_calculator
                    .calculate(
                        epochs[i],
                        epochs[j]
                    )
                )    
    
        # копируем часть матрицы над главной диагональю под неё
        dist_matrix = np.where(
            dist_matrix > 0,
            dist_matrix,
            dist_matrix.T
        )
    
        return dist_matrix

    def __cluster_epochs(
        self, epochs: List[pd.DataFrame], 
        distance_calculator: DistanceCalculator) -> List[pd.DataFrame]:
        """
        Кластеризуем эпохи при помощи AffinityPropagation
        с вычисленной относительно метрики из DistanceCalculator
        матрицей расстояний между эпохами
        
        Параметры:
	     epochs - набор эпох	
	     distance_calculator - экземпляр класса 
	     DistanceCalculator
        """
        aff_prop = AffinityPropagation(
            random_state=SEED,
            affinity='precomputed',
            damping=0.8
        )

        # считаем матрицу расстояний между эпохами
        dist_matrix = self.__get_dist_matrix(
            epochs,
            distance_calculator
        )

        # кластеризуем эпохи
        aff_prop.fit(-dist_matrix)
        self.labels = aff_prop.labels_
        self.centroids_indices = aff_prop.cluster_centers_indices_

        # собираем список центроидов и список их классов
        centroids = []
        for ind in self.centroids_indices:
            centroids.append(epochs[ind])

        return centroids

    def average(
        self, epochs: List[pd.DataFrame], 
        distance_calculator: DistanceCalculator) -> List[pd.DataFrame]:
	"""
	Усредняем переданные эпохи - ищем центроиды кластеров, 
	выделенных AffinityPropagation для этого набора эпох
	
	Параметры:
	     epochs - набор эпох	
	     distance_calculator - экземпляр класса DistanceCalculator
	"""   
        centroids = self.__cluster_epochs(epochs, distance_calculator)
        return centroids


@dataclass
class ChannelClusterer(Averager, yaml.YAMLObject):
    yaml_tag = u'!ChannelClusterer'
    yaml_loader = yaml.SafeLoader

    n_clusters: int = 2
    
    def __repr__(self):
        return f"{self.__class__.__name__}(n_clusters={self.n_clusters})"

    def __cluster_channels(
        self, epochs: List[pd.DataFrame], 
        distance_calculator: DistanceCalculator) -> List[pd.DataFrame]:
        """
        Кластеризуем сигналы по каждому из каналов при помощи 
        KMeans с фиксированным количеством кластеров
        
        Параметры:
	     epochs - набор эпох	
	     distance_calculator - экземпляр класса 
	     DistanceCalculator
        """
        channels = list(epochs[0].columns)

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=SEED,
            n_init='auto'
        )

        centroids = [epochs[0].reset_index(drop=True).copy()] * self.n_clusters

        for channel in channels:
            signals_for_channel = np.array(
                [epoch[channel].tolist() for epoch in epochs]
            )
            
            # кластеризуем сигналы по каналу
            kmeans.fit(signals_for_channel)
            channel_centroids = kmeans.cluster_centers_

            # заполняем данный канал в датафреймах
            for i in range(len(channel_centroids)):
                centroids[i][channel] = channel_centroids[i]

        return centroids

    def average(
        self, epochs: List[pd.DataFrame], 
        distance_calculator: DistanceCalculator) -> List[pd.DataFrame]:
	"""
	Усредняем переданные эпохи - ищем центроиды кластеров, 
	выделенных KMeans для каждого канала этого набора эпох
	
	Параметры:
	     epochs - набор эпох	
	     distance_calculator - экземпляр класса DistanceCalculator
	"""  
        centroids = self.__cluster_channels(epochs, distance_calculator)
        return centroids

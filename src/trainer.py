import abc
from typing import List
import yaml
from utils import CombinedMeta
from dataclasses import dataclass, field
import pickle

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax

from distance_calculator import DistanceCalculator

from metric_learn import LMNN, NCA, ITML_Supervised

SEED = 43
# PATH_TO_LEARNED_FUNCTIONS = '../experiments/results/learned_functions/'


class Trainer(abc.ABC, metaclass=CombinedMeta):
    def set_param(self, param_name: str, value: str):
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
    def train(self, train_epochs: List[pd.DataFrame],
    	      train_target: List[int],
              pos_refs: List[pd.DataFrame],
              neg_refs: List[pd.DataFrame], 
              distance_calculator: DistanceCalculator):
        pass


class NoTrainer(Trainer, yaml.YAMLObject):
    yaml_tag = u'!NoTrainer'
    yaml_loader = yaml.SafeLoader
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def train(self, train_epochs: List[pd.DataFrame],
    	      train_target: List[int],
              pos_refs: List[pd.DataFrame],
              neg_refs: List[pd.DataFrame], 
              distance_calculator: DistanceCalculator):
        """
        Не производим никакие действия (класс нужен 
        для согласованности описаний алгоритмов)
        """
        pass


@dataclass
class EmbedMetricTrainer(Trainer, yaml.YAMLObject):
    yaml_tag = u'!EmbedMetricTrainer'
    yaml_loader = yaml.SafeLoader

    algorithm: str = 'itml_supervised'
    reducer: str = 'pca'
    embedding_dim: int = 50 

    def __repr__(self):
        part1 = f"{self.__class__.__name__}(algorithm={self.algorithm}, "
        part2 = f"reducer={self.reducer}, embedding_dim={self.embedding_dim})"
        return part1 + part2

    def train(self, train_epochs: List[pd.DataFrame],
              train_target: List[int],
              pos_refs: List[pd.DataFrame],
              neg_refs: List[pd.DataFrame], 
              distance_calculator: DistanceCalculator):
        """
        Вытягиваем эпохи в один вектор, понижаем размерность PCA; 
        на полученных эмбеддингах обучаем метрику одним из 
        алгоритмов (self.algorithm)
        
        Параметры:
            train_epochs - эпохи обучающего набора
    	    train_target - лейблы эпох обучающего набора
            pos_refs - референты позитивных эпох
            neg_refs - референты негативных эпох
            distance_calculator - экземпляр класса DistanceCalculator
        """
        n = train_epochs[0].shape[0] * train_epochs[0].shape[1]

        embeddings = np.empty((0, n))
            
        for epoch in train_epochs:
            embeddings = np.vstack((
                embeddings,
                epoch.values.flatten()
            ))

        embedding_dim = min(self.embedding_dim, len(train_epochs))
        reducer = PCA(n_components=embedding_dim)

        embeddings = reducer.fit_transform(embeddings)
        
        if self.algorithm == 'lmnn':
            model = LMNN(n_neighbors=5, learn_rate=1e-6, random_state=SEED)
        elif self.algorithm == 'nca':
            model = NCA(n_components=None, max_iter=100, random_state=SEED)
        else:
            model = ITML_Supervised(n_constraints=None, random_state=SEED)
        
        model.fit(embeddings, train_target)
        metric = model.get_metric()
            
        distance_calculator.set_param_forced('reducer', reducer)    
        distance_calculator.set_param_forced('learned_metric', metric)


@dataclass
class MetricTrainer(Trainer, yaml.YAMLObject):
    yaml_tag = u'!MetricTrainer'
    yaml_loader = yaml.SafeLoader

    algorithm: str = 'itml_supervised'
    # каналы, для каждого из которых обучается метрика
    channels: List[str] = field(default_factory=lambda: ['C1_laplacian'])

    def __repr__(self):
    	part1 = f"{self.__class__.__name__}(algorithm={self.algorithm}, "
    	part2 = f"channels={self.channels})"
        return part1 + part2

    def train(self, train_epochs: List[pd.DataFrame],
              train_target: List[int],
              pos_refs: List[pd.DataFrame],
              neg_refs: List[pd.DataFrame], 
              distance_calculator: DistanceCalculator):
        """
        Обучаем свою метрику для каждого канала (self.channels) 
        одним из алгоритмов (self.algorithm)
        
        Параметры:
            train_epochs - эпохи обучающего набора
    	    train_target - лейблы эпох обучающего набора
            pos_refs - референты позитивных эпох
            neg_refs - референты негативных эпох
            distance_calculator - экземпляр класса DistanceCalculator
        """
        n = train_epochs[0].shape[0]

        learned_metrics = []

        for channel in self.channels:
            epoch_embeddings = np.empty((0, n))
            
            for epoch in train_epochs:
                epoch_embeddings = np.vstack((
                    epoch_embeddings,
                    epoch[channel].values
                ))
            
            if self.algorithm == 'lmnn':
                model = LMNN(n_neighbors=5, learn_rate=1e-6, random_state=SEED)
            elif self.algorithm == 'nca':
                model = NCA(n_components=None, max_iter=100, random_state=SEED)
            else:
                model = ITML_Supervised(n_constraints=None, random_state=SEED)
            
            model.fit(epoch_embeddings, train_target)
            metric = model.get_metric()
            learned_metrics.append(metric)

            # сохранение полученной функции
            # metric_save_path = f'{PATH_TO_LEARNED_FUNCTIONS}{channel}.pickle'
            
            # with open(metric_save_path, 'wb') as output_file:
            #     pickle.dump(metric, output_file)
            
        distance_calculator.set_param_forced('learned_metrics', learned_metrics)
        distance_calculator.set_param_forced('channels', self.channels)


@dataclass
class LogRegSoftmaxTrainer(Trainer, yaml.YAMLObject):
    yaml_tag = u'!LogRegSoftmaxTrainer'
    yaml_loader = yaml.SafeLoader

    logreg_penalty: str = 'l2'
    logreg_solver: str = 'liblinear'

    def __repr__(self):
        return f"{self.__class__.__name__}(logreg_penalty={self.logreg_penalty})"

    def train(self, train_epochs: List[pd.DataFrame],
              train_target: List[int],
              pos_refs: List[pd.DataFrame],
              neg_refs: List[pd.DataFrame], 
              distance_calculator: DistanceCalculator):
        """
        Вычисляем расстояния от эпох обучающего набора
        до референтов, переводим их в веса с помощью Softmax,
        формируем прогноз, с точки зрения каждого канала, как
        взвешенную сумму классов референтов, обучаем
        логистическую регрессию на полученных эмбеддингах
        
        Параметры:
            train_epochs - эпохи обучающего набора
    	    train_target - лейблы эпох обучающего набора
            pos_refs - референты позитивных эпох
            neg_refs - референты негативных эпох
            distance_calculator - экземпляр класса DistanceCalculator
        """
        n = len(train_epochs[0].columns)
        train_target_np = np.array(train_target)
        channel_preds = np.empty((0, n))
        ref_classes = [[1]*len(pos_refs) + [0]*len(neg_refs)]
        
        for i in range(len(train_epochs)):
            dist = np.empty((0, n))
            
            for ref in pos_refs + neg_refs:
                dist_to_ref = distance_calculator.calculate(
                    train_epochs[i],
                    ref,
                )
                dist = np.vstack((dist, dist_to_ref))
                
            coefs = np.repeat(ref_classes, n, axis=0).T
            preds = np.sum(softmax(-dist, axis=0) * coefs, axis=0)
            channel_preds = np.vstack((channel_preds, preds))
        
        logreg = LogisticRegression(random_state=SEED, 
                                    solver=self.logreg_solver, 
                                    penalty=self.logreg_penalty)
        logreg.fit(channel_preds, train_target_np)
        self.logreg = logreg


@dataclass
class LogRegTrainer(Trainer, yaml.YAMLObject):
    yaml_tag = u'!LogRegTrainer'
    yaml_loader = yaml.SafeLoader

    logreg_penalty: str = 'l2'
    logreg_solver: str = 'liblinear'

    def __repr__(self):
        return f"{self.__class__.__name__}(logreg_penalty={self.logreg_penalty})"

    def train(self, train_epochs: List[pd.DataFrame],
              train_target: List[int],
              pos_refs: List[pd.DataFrame],
              neg_refs: List[pd.DataFrame], 
              distance_calculator: DistanceCalculator):
        """
        Вычисляем расстояния от эпох обучающего набора
        до референтов (усредненной позитивной и усредненной 
        негативной эпох), конкатенируем их в один вектор и обучаем
        логистическую регрессию на полученных эмбеддингах
        
        Параметры:
            train_epochs - эпохи обучающего набора
    	    train_target - лейблы эпох обучающего набора
            pos_refs - референты позитивных эпох
            neg_refs - референты негативных эпох
            distance_calculator - экземпляр класса DistanceCalculator
        """
        n = len(train_epochs[0].columns)
        train_target_np = np.array(train_target)
        # название channel_preds оставлено 
        # для согласованности с LogRegSoftmaxTrainer
        channel_preds = np.empty((0, 2*n))
        
        for i in range(len(train_epochs)):
            dist_to_pos = distance_calculator.calculate(
                train_epochs[i],
                pos_refs[0], # усредненная позитивная эпоха
            )
            dist_to_neg = distance_calculator.calculate(
                train_epochs[i],
                neg_refs[0], # усредненная негативная эпоха
            )
            dist = np.hstack((dist_to_pos, dist_to_neg))
            channel_preds = np.vstack((channel_preds, dist))

        # Z-СТАНДАРТИЗАЦИЯ -- НАЧАЛО
        # mean_by_channel = np.mean(channel_preds, axis=0)
        # std_by_channel = np.std(channel_preds, axis=0)
        # channel_preds = (channel_preds - mean_by_channel) / std_by_channel

        # self.mean_by_channel = mean_by_channel
        # self.std_by_channel = std_by_channel
        # Z-СТАНДАРТИЗАЦИЯ -- КОНЕЦ
        
        logreg = LogisticRegression(random_state=SEED, 
                                    solver=self.logreg_solver, 
                                    penalty=self.logreg_penalty)
        logreg.fit(channel_preds, train_target_np)
        self.logreg = logreg

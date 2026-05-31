from typing import List, Dict, Tuple
from itertools import compress, product
import yaml
from utils import dynamic_constructor
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge

import sys
sys.path.append('../src')

from distance_calculator import DistanceCalculator
from smoother import Smoother, ChannelSmoother
from averager import Averager, NoAverager, EpochAverager, EpochClusterer, ChannelClusterer
from trainer import Trainer, NoTrainer, LogRegSoftmaxTrainer, LogRegTrainer, MetricTrainer, EmbedMetricTrainer
from selector import CandidateSelector, AllCandidateSelector, KNNCandidateSelector
from comparator import CandidatesComparator, AverageComparator, LogRegSoftmaxComparator, MostCommonComparator, LogRegComparator

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

from concurrent.futures import ProcessPoolExecutor, as_completed


SEED = 43
PATH_TO_RESULTS = '../results/centroid/'
PATH_TO_CONFIG = '../configs/config.yaml'

# маппинг названий классов на классы
classes = [
    DistanceCalculator,
    ChannelSmoother,
    NoAverager, EpochClusterer, ChannelClusterer, EpochAverager,
    NoTrainer, LogRegSoftmaxTrainer, LogRegTrainer, MetricTrainer, EmbedMetricTrainer,
    AllCandidateSelector, KNNCandidateSelector,
    AverageComparator, LogRegSoftmaxComparator, MostCommonComparator, LogRegComparator
]


def specificity_func(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn+fp)


def sensitivity_func(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp+fn)


def balanced_accuracy_func(y_true, y_pred):
    spec = specificity_func(y_true, y_pred)
    sens = sensitivity_func(y_true, y_pred)
    return (spec + sens)/2
    

def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in product(*kwargs.values()):
        yield dict(zip(keys, instance))


# добавляем конструктор для каждого класса
for cls in classes:
    yaml.add_constructor(f'!{cls.__name__}', dynamic_constructor)


class Experiment:
    def __init__(
        self,
        parameters: Tuple[
            DistanceCalculator,
            Smoother,
            Averager,
            Trainer,
            CandidateSelector,
            CandidatesComparator
        ]):
        self.parameters = parameters

    def __repr__(self):
        return tuple(repr(param) for param in self.parameters)

    def set_params(self, params_grid: Dict[str, str | int]):
        """
        Инициализируем параметры классов в соответствии 
        с заданной сеткой
        
        Параметры:
            params_grid - сетка параметров для эксперимента
        """
        for key in params_grid:
            class_name, param = key.split('__')
            val = params_grid[key]
            for component in self.parameters:
                if component.__class__.__name__ == class_name:
                    component.set_param(param, val)
                    break

    def run(self, name: str,
            participant_dict: Dict[str, List[pd.DataFrame]]) -> pd.DataFrame:
        """
        Запускаем эксперимент для одного испытуемого
        
        Параметры:
            name - ключ испытуемого в общем словаре	
            participant_dict - словарь позитивных и негативных 
            эпох испытуемого
        """
        # объединяем все эпохи в один список
        all_epochs = (
            participant_dict['positive_epochs']
            + participant_dict['negative_epochs']
        )
        pos_epochs_num = len(participant_dict['positive_epochs'])
        neg_epochs_num = len(participant_dict['negative_epochs'])
        target = [True] * pos_epochs_num + [False] * neg_epochs_num

        distance_calculator = self.parameters[0]
        smoother = self.parameters[1]
        averager = self.parameters[2]
        trainer = self.parameters[3]
        selector = self.parameters[4]
        comparator = self.parameters[5]

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED
        )

        results_by_iter = []
        
        for i, (train_index, test_index) in enumerate(cv.split(all_epochs, target)):    
            train_epochs, train_target = [], []
            test_epochs, test_target = [], []
            
            for ind in range(len(target)):
                if ind in train_index:
                    train_epochs.append(all_epochs[ind])
                    train_target.append(target[ind])
                else:
                    test_epochs.append(all_epochs[ind])
                    test_target.append(target[ind])

            train_epochs = smoother.smoothe(train_epochs)
            test_epochs = smoother.smoothe(test_epochs)

            train_target_np = np.array(train_target)
            pos_epochs = list(compress(train_epochs, train_target_np))
            neg_epochs = list(compress(train_epochs, ~train_target_np))
            
            pos_refs = averager.average(pos_epochs, distance_calculator)
            neg_refs = averager.average(neg_epochs, distance_calculator)
            trainer.train(
                train_epochs,
                train_target,
                pos_refs,
                neg_refs,
                distance_calculator
            )

            test_target_pred = []

            for epoch in tqdm(test_epochs):
                positive_distances, negative_distances = selector.calculate(
                    distance_calculator,
                    pos_refs,
                    neg_refs,
                    epoch
                )
                pred = comparator.compare(positive_distances, negative_distances, trainer)
                test_target_pred.append(pred)

            specificity = specificity_func(test_target, test_target_pred)
            sensitivity = sensitivity_func(test_target, test_target_pred)
            accuracy = balanced_accuracy_func(test_target, test_target_pred)
            
            results_by_iter.append([
                specificity,
                sensitivity,
                accuracy,
            ])
        
        mean_results = np.mean(results_by_iter, axis=0)
        specificity = mean_results[0]
        sensitivity = mean_results[1]
        accuracy = mean_results[2]

        results_sd = np.std(results_by_iter, axis=0)
        specificity_sd = results_sd[0]
        sensitivity_sd = results_sd[1]
        accuracy_sd = results_sd[2]
            
        participant_res = {
            'Participant': [name],
            'Positive epochs': [pos_epochs_num],
            'Negative epochs': [neg_epochs_num],
            'Specificity (TNR)': [round(specificity, 4)],
            'Sensitivity (Recall, TPR)': [round(sensitivity, 4)],
            'Accuracy': [round(accuracy, 4)],
            'Dist Metric': [distance_calculator.metric],
            'Window Size': [smoother.window_size],
            'Config': [self.__repr__()],
            'Specificity SD': [round(specificity_sd, 4)],
            'Sensitivity SD': [round(sensitivity_sd, 4)],
            'Accuracy SD': [round(accuracy_sd, 4)],
        } 
        res_df = pd.DataFrame.from_dict(participant_res)

        return res_df

    def run_for_all(
        self,
        signals: Dict[str, Dict[str, List[pd.DataFrame]]]) -> pd.DataFrame:
        """
        Запускаем эксперимент для всех испытуемых
        
        Параметры:
            signals - словарь, ключи которого - имена испытуемых,
            а значения - словари позитивных и негативных эпох 
        """
        results = []
        pairs = list(signals.items())

        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.run, *pair): pair[0] for pair in pairs}

            for future in as_completed(futures):
                name = futures[future]  # забираем исходный кортеж параметров
                result = future.result()  # забираем результат завершенной задачи
                print(f'   Эксперимент {name} завершен')
                results.append(result)
                            
            res_df = pd.concat(results, ignore_index=True)

        return res_df

        # Calculate metrics
        # Dump into XLS table
        # Participant, Pos No, Neg. No, Specificity, Sensititvity, BA, parameters

def save_best_results(exp_results: pd.DataFrame, exp_name: str):
    """
    Сохраняем лучшие результаты по каждому испытуемому в файл
    
    Параметры:
    	exp_results - датафрейм с результатами для всех 
    	наборов параметров данного эксперимента
    	exp_name - название эксперимента
    """
    res = (
        exp_results
        .sort_values('Accuracy', ascending=False)
        .groupby('Participant', as_index=False)
        .first()
    )
    print(
        f'Среднее значение Balanced Accuracy для '
        f'эксперимента {exp_name}: {res["Accuracy"].mean()}'
    )
    res.to_csv(f'{PATH_TO_RESULTS}{exp_name}.csv')


def load_config(config_file_path: str = PATH_TO_CONFIG) -> Dict:
    """
    Загружаем yaml-конфиг эксперимента
    
    Параметры:
        config_file_path - путь до yaml-файла с конфигом
    """
    try:
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Ошибка: файл с конфигом '{config_file_path}' не найден")
        return None
    except yaml.YAMLError as e:
        print(f"Ошибка парсинга yaml-файла: {e}")
        return None


def main(signals: Dict[str, Dict[str, List[pd.DataFrame]]]) -> Dict[str, pd.DataFrame]:
    results_dict = {}

    config = load_config()

    if config:
        param_grids = list(product_dict(**config['param_grid']))
        
        for i in config['experiments_grid']:
            exp_results = pd.DataFrame()
            exp_config = config['experiments_grid'][i]
            exp = Experiment(exp_config)
            
            for grid in param_grids:
                exp.set_params(grid)
                print(f'Эксперимент {i} с конфигом {repr(exp_config)} начат')
                res_df = exp.run_for_all(signals)
                res_df['Experiment'] = i
                print(f'Эксперимент {i} с конфигом {repr(exp_config)} завершен')
                exp_results = pd.concat([exp_results, res_df], ignore_index=True)
            
            results_dict[i] = exp_results
            save_best_results(exp_results, i)

    return results_dict

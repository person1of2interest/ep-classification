from typing import List, Dict, Tuple
from itertools import compress, product
import yaml
from utils import dynamic_constructor
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge

from distance_calculator import DistanceCalculator
from smoother import Smoother, ChannelSmoother
from averager import Averager, NoAverager, EpochAverager, EpochClusterer, ChannelClusterer
from trainer import Trainer, NoTrainer, LogRegSoftmaxTrainer, LogRegTrainer, MetricTrainer, EmbedMetricTrainer
from selector import CandidateSelector, AllCandidateSelector, KNNCandidateSelector
from comparator import CandidatesComparator, AverageComparator, LogRegSoftmaxComparator, MostCommonComparator, LogRegComparator

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

SEED = 43
PATH_TO_RESULTS = '../experiments/results/final/'
PATH_TO_CONFIG = '../experiments/configs/config.yaml'

exp_counter = 0 # глобальная переменная для сохранения коэф-в лог.регресии

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


# ДЛЯ FEATURE IMPORTANCES
def compute_importances(feature_data: np.ndarray, logreg_coefs: np.ndarray, 
                        ridge_alpha: float = 1.0) -> np.ndarray:
    """
    Вычисляем важности всех признаков, используя коэффициенты
    логистической регрессии, с учетом масштабов признаков
    и их скоррелированности друг с другом

    Параметры:
        feature_data - массив значений признаков (строки - отдельные сэмплы)
        logreg_coefs - коэффициенты модели логистической регрессии
        ridge_alpha - параметр alpha для Ridge-регрессии
    """
    # importance_i^(t) = |β_i^(t)| * IQR_i^(t) * (1 − R2_i^(t))
    # вычисляем IQR для каждого признака   
    q75 = np.quantile(feature_data, 0.75, axis=0)    
    q25 = np.quantile(feature_data, 0.25, axis=0)    
    iqr = np.where(q75 - q25 == 0, 1e-8, q75 - q25) # избегаем деления на 0

    # base scaled effect (absolute)    
    s = np.abs(logreg_coefs) * iqr

    # вычисляем R^2_i, предсказывая каждый признак по всем остальным 
    R2 = np.zeros(feature_data.shape[1])    
    for i in range(feature_data.shape[1]):        
        y = feature_data[:, i]       
        X_others = np.delete(feature_data, i, axis=1)
        
        # обучаем Ridge-регрессию       
        model = Ridge(alpha=ridge_alpha, fit_intercept=True)        
        model.fit(X_others, y)        
        
        y_pred = model.predict(X_others)  
        
        # считаем R^2 (обрезаем до [0,1])        
        r2 = r2_score(y, y_pred)        
        R2[i] = np.clip(r2, 0.0, 1.0)

    imp_raw = s * (1.0 - R2)

    # нормализуем так, чтобы сумма важностей всех признаков равнялась 1   
    if imp_raw.sum() > 0:        
        res = imp_raw / imp_raw.sum()    
    else:        
        res = np.zeros_like(imp_raw)

    return res


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
        # ДЛЯ FEATURE IMPORTANCES
        feature_importances_by_iter = np.empty((0, 2*len(all_epochs[0].columns)+1))
        
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

            # ДЛЯ FEATURE IMPORTANCES НАЧАЛО
            train_feature_data = np.empty((0, 2*len(all_epochs[0].columns)))

            for epoch in tqdm(train_epochs):
                positive_distances, negative_distances = selector.calculate(
                    distance_calculator,
                    pos_refs,
                    neg_refs,
                    epoch
                )
                train_feature_data = np.vstack((
                    train_feature_data, 
                    np.hstack((positive_distances, negative_distances))
                ))

            feature_importances = compute_importances(
                train_feature_data, 
                trainer.logreg.coef_.ravel()
            )
            feature_importances_by_iter = np.vstack((
                feature_importances_by_iter, 
                np.hstack((feature_importances, [accuracy]))
            ))
            # ДЛЯ FEATURE IMPORTANCES КОНЕЦ
        
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

        # ДЛЯ FEATURE IMPORTANCES
        global exp_counter
        col_names = (
            [column + '_POS' for column in epoch.columns] 
            + [column + '_NEG' for column in epoch.columns] 
            + ['Accuracy']
        )
        feature_importances_dict = {
            col_names[j]: feature_importances_by_iter[:, j] for j in range(len(col_names))
        }
        (
            pd.DataFrame.from_dict(feature_importances_dict)
            .to_csv(
                f'{PATH_TO_RESULTS}{distance_calculator.metric}'
                f'_{smoother.window_size}_{trainer.logreg_penalty}'
                f'_{name}.csv', 
                index=False
            )
        )

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

        global exp_counter
        exp_counter += 1

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
import abc
import yaml
from typing import Any

# мета-класс для abc.ABC и yaml.YAMLObject
class CombinedMeta(abc.ABCMeta, type(yaml.YAMLObject)):
    pass


def dynamic_constructor(yaml_loader: yaml.SafeLoader, 
			node: yaml.nodes.MappingNode) -> Any:
    """
    Конструктор класса из yaml-rjyabuf
    
    Параметры:
    	yaml_loader - экземпляр загрузчика PyYAML
    	node - узел типа mapping, содержащий тег и 
    	пары ключ-значение
    """
    # получаем имя класса по тэгу
    tag = node.tag.lstrip('!')
    class_type = class_mapping.get(tag)

    if class_type is None:
        raise ValueError(f"Неизвестный тэг класса: {tag}")

    # конструируем маппинг и инициализируем инстанс класса
    value = yaml_loader.construct_mapping(node, deep=False)
    return class_type(**value)

from __future__ import annotations
from copy import deepcopy
import dill

from pathlib import Path
from alibi_detect.version import __version__, __config_spec__
import logging
from typing import Dict, Any
from typing import Union
from alibi_detect.saving.registry import registry
import toml
import os
import torch

logger = logging.getLogger(__name__)


def write_config(cfg: dict, filepath: Union[str, os.PathLike]):
    filepath = Path(filepath)
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)

    logger.info('Writing config to {}'.format(filepath.joinpath('config.toml')))
    with open(filepath.joinpath('config.toml'), 'w') as f:
        toml.dump(cfg, f, encoder=toml.TomlNumpyEncoder())  # type: ignore[misc]


class ConfigMixin:
    """Mixed in to anything you want to save.

    define CONFIG_PARAMS, LARGE_PARAMS and BASE_OBJ on the class you've mixed this into in order to
    obtain the desired behavour. Users should be able to use this mix in on there own objects in order
    to ensure they are serialisable.
    """
    CONFIG_PARAMS: tuple = ()   # set of args passed to init that will be in config
    LARGE_PARAMS: tuple = ()    # set of args passed to init that are big and are added to config when it's getted
    BASE_OBJ: bool = False      # Base objects are things like detectors and Ensembles that should have there own 
                                # self contained config and be referenced from other configs.
    # FROM_PATH: tuple = ()       # Set of values that are are resolved from paths in the config
    MODULES = ()

    def _set_config(self, inputs):
        pass
        name = self.__class__.__name__
        config: Dict[str, Any] = {
            'name': name,
        }

        if self.BASE_OBJ:
            config['meta'] = {
                'version': __version__,
                'config_spec': __config_spec__,
            }

        for key in self.CONFIG_PARAMS:
            if key not in self.LARGE_PARAMS:
                config[key] = inputs[key]
        self.config = config

    def get_config(self) -> dict:
        if self.config is not None:
            cfg = self.config

            for key in self.LARGE_PARAMS:
                cfg[key] = getattr(self, key)

        else:
            raise NotImplementedError('Getting a config (or saving via a config file) is not yet implemented for this'
                                      'detector')

        return cfg

    @classmethod
    def from_config(cls, config: dict):
        """
        Note: For custom or complicated behavour this should be overidden
        """
        return cls(**config)

    def serialize_value(self, key, val, path):
        path = f'{path}/{key}'

        if hasattr(self, f'_{key}_serializer'):
            key_serialiser = getattr(self, f'_{key}_serializer')
            return key_serialiser(key, val, path)

        elif hasattr(self, f'_{type(val).__name__}_serializer'):
            type_serialiser = getattr(self, f'_{type(val).__name__}_serializer')
            return type_serialiser(key, val, path)

        elif hasattr(val, 'BASE_OBJ') and not getattr(val, 'BASE_OBJ'):
            return val.serialize(path)

        elif hasattr(val, 'BASE_OBJ') and getattr(val, 'BASE_OBJ'):
            return val.save(path)

        else:
            return val

    def _list_serializer(self, key, val, path):
        return [self.serialize_value(ind, item, path) for ind, item in enumerate(val)]

    def _dict_serializer(self, key, val, path):
        return {k: self.serialize_value(k, v, path) for k, v in val.items()}

    def _function_serializer(self, key, val, path):
        path = Path(path)
        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)
        path = path.with_suffix('.dill')
        with open(path, 'wb') as f:
            dill.dump(val, f)
        return str(path)

    def serialize(self, path):
        cfg = self.get_config()
        n_cfg = {}
        for key, val in cfg.items():
            n_cfg[key] = self.serialize_value(key, val, path)
        return n_cfg

    def save(self, path):
        cfg = self.serialize(path)
        write_config(cfg, path)
        return path

    @classmethod
    def deserialize_value(cls, key, val):
        if hasattr(cls, f'_{key}_deserializer'):
            key_deserialiser = getattr(cls, f'_{key}_deserializer')
            return key_deserialiser(key, val)

        if isinstance(val, str) and val.startswith('@'):
            res_path = val[1:]
            target = registry.get_all().get(res_path, None)
            return target

        if isinstance(val, str) and '.dill' in val:
            return dill.load(open(f'{val}', 'rb'))

        elif isinstance(val, dict) and val.get('name', None):
            object_name = val.pop('name')
            obj = registry.get_all()[object_name]
            return obj.deserialize(val)

        else:
            return val

    @classmethod
    def deserialize(cls, cfg):
        for key, val in cfg.items():
            cfg[key] = cls.deserialize_value(key, val)
        return cls(**cfg)

    # def save_state(self, path):
    #     import torch
    #     for module in self.MODULES:
    #         torch.save(module, path)


class ModelWrapper(ConfigMixin):
    BASE_OBJ = False

    def __init__(self, model, input_shape=None):
        self.model = model
        self.input_shape = input_shape

    def __getattr__(self, key: str):
        """Expose the wrapped model's methods and attributes."""
        if hasattr(self.model, key):
            return getattr(self.model, key)
        raise AttributeError(...)

    def serialize(self, path):
        path = Path(path)
        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)
        path = path.with_suffix('.pt')
        import torch
        torch.save(self.model, path)
        return str(path)

    def copy(self):
        return deepcopy(self.model)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# class CustomModel(ConfigMixin):
#     BASE_OBJ = False

#     def __init__(self, model):
#         self.model = model

#     def __getattr__(self, key: str):
#         """Expose the wrapped model's methods and attributes."""
#         if hasattr(self.model, key):
#             return getattr(self.model, key)
#         raise AttributeError(...)

#     def serialize(self, path):
#         raise NotImplementedError()

#     @classmethod
#     def deserialize(cls, cfg):
#         raise NotImplementedError()

#   def __call__(self, *args, **kwargs):
#     return self.model(*args, **kwargs)


class StatefulMixin:
    STATE = ()

    def save_state(self, path):
        path = str(path) + '/state'
        make_dir(path)
        for attr_name in self.STATE:
            item = getattr(self, attr_name)
            item_path = f'{path}/{attr_name}'
            self.save_item_state(attr_name, item, item_path)

    def save_item_state(self, name, item, path):
        if hasattr(self, f'_{name}_save_state'):
            getattr(self, f'_{name}_save_state')(item, path)
        if hasattr(self, f'_{type(item).__name__}_save_state'):
            getattr(self, f'_{type(item).__name__}_save_state')(item, path)
        elif isinstance(item, torch.nn.Module):
            # print([i.__name__ for i in type(item).__mro__])
            # torch.save(item.state_dict(), f'{path}.pt')
            self.save_torch_module(item, path)
        elif isinstance(item, StatefulMixin):
            item.save_state(path)

    def save_torch_module(self, module, path):
        state_dict = module.state_dict()
        if hasattr(module, 'STATE'):
            for attr_name in module.STATE:
                attr = getattr(module, attr_name)
                state_dict[attr_name] = attr
        torch.save(state_dict, f'{path}.pt')

    def _list_save_state(self, items, path):
        make_dir(path)
        for ind, item in enumerate(items):
            if isinstance(item, (StatefulMixin, ConfigMixin)):
                item_path = f'{path}/{type(item).__name__}'
            else:
                item_path = f'{path}/{ind}'
            self.save_item_state(ind, item, item_path)

    def load_state(self, path):
        path = str(path) + '/state'
        for attr_name in self.STATE:
            item = getattr(self, attr_name)
            item_path = f'{path}/{attr_name}'
            self.load_item_state(attr_name, item, item_path)

    def load_item_state(self, name, item, path):
        if hasattr(self, f'_{name}_load_state'):
            getattr(self, f'_{name}_load_state')(item, path)
        if hasattr(self, f'_{type(item).__name__}_load_state'):
            getattr(self, f'_{type(item).__name__}_load_state')(item, path)
        elif isinstance(item, torch.nn.Module):
            self.load_torch_module(item, path)
        elif isinstance(item, StatefulMixin):
            item.load_state(path)

    def load_torch_module(self, item, path):
        state_dict = torch.load(f'{path}.pt')
        if hasattr(item, 'STATE'):
            for attr_name in item.STATE:
                setattr(item, attr_name, state_dict.pop(attr_name))
        item.load_state_dict(state_dict)

    def _list_load_state(self, items, path):
        for ind, item in enumerate(items):
            if isinstance(item, (StatefulMixin, ConfigMixin)):
                item_path = f'{path}/{type(item).__name__}'
            else:
                item_path = f'{path}/{ind}'
            self.load_item_state(ind, item, item_path)


def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

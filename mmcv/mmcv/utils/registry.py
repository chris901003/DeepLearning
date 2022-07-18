# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import warnings
from functools import partial
from typing import Any, Dict, Optional

from .misc import deprecated_api_warning, is_seq_of


def build_from_cfg(cfg: Dict,
                   registry: 'Registry',
                   default_args: Optional[Dict] = None) -> Any:
    """Build a module from config dict when it is a class configuration, or
    call a function from config dict when it is a function configuration.

    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = build_from_cfg(dict(type='Resnet'), MODELS)
        >>> # Returns an instantiated object
        >>> @MODELS.register_module()
        >>> def resnet50():
        >>>     pass
        >>> resnet = build_from_cfg(dict(type='resnet50'), MODELS)
        >>> # Return a result of the calling function

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """

    """
    :param cfg: config文件當中的model部分 
    :param registry: 註冊器，裏面有module_dict有很多的backbone
    :param default_args: train_cfg以及test_cfg
    :return: 
    """
    # 已看過

    # 一下都是做一些檢查，正常的config文件是不會有問題的
    # 如果cfg不是dict格式在這裡就會報錯
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    # 在cfg當中一定會有一個type，來表示要用哪個class或是function
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')
    # 如果registry不是Registry格式會報錯
    if not isinstance(registry, Registry):
        raise TypeError('registry must be an mmcv.Registry object, '
                        f'but got {type(registry)}')
    # 如果default_args不是dict格式或是default_args是None就會報錯
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}')

    # args = cfg的深度拷貝，這裡的cfg是config裏面的model部分
    args = cfg.copy()

    if default_args is not None:
        # 如果args裏面沒有default_args的key就會添加上去，同時value也會放上去
        # 如果已經有了就不會有任何操作
        for name, value in default_args.items():
            args.setdefault(name, value)

    # 將args當中key為type的value取出，同時將type從args當中移除
    # obj_type = 目前的type
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        # 如果obj_type的型態為str就會進入這裡

        # 這裡就是在已經實現好的class當中找到我們配置文件指定的class並拿出來做使用
        # obj_cls = class型態，根據type會找到我們需要的實例對象
        # 如果去找到該檔案位置的class可以發現該class會有裝飾器@xxx.register_module()，xxx依據不同的註冊表會有不同
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            # 如果obj_cls=None表示沒有該實例對象，這裡就會直接報錯
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
        # 透過inspect可以檢查obj_type是什麼類別
        # 如果是class或是function類別就直接給到obj_cls，通常是不會有這種情況
        obj_cls = obj_type
    else:
        # 其他類別就直接報錯
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')
    try:
        # 嘗試將args裡的東西放到obj_cls裏面，也就是對obj_cls進行實例化，args就是初始化中的參數
        return obj_cls(**args)
    except Exception as e:
        # 依照正常的config不會跑到這裡來，這裡就是有問題
        # Normal TypeError does not print class name.
        raise type(e)(f'{obj_cls.__name__}: {e}')


class Registry:
    """A registry to map strings to classes or functions.

    Registered object could be built from registry. Meanwhile, registered
    functions could be called from registry.

    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODELS.build(dict(type='ResNet'))
        >>> @MODELS.register_module()
        >>> def resnet50():
        >>>     pass
        >>> resnet = MODELS.build(dict(type='resnet50'))

    Please refer to
    https://mmcv.readthedocs.io/en/latest/understand_mmcv/registry.html for
    advanced usage.

    Args:
        name (str): Registry name.
        build_func(func, optional): Build function to construct instance from
            Registry, func:`build_from_cfg` is used if neither ``parent`` or
            ``build_func`` is specified. If ``parent`` is specified and
            ``build_func`` is not given,  ``build_func`` will be inherited
            from ``parent``. Default: None.
        parent (Registry, optional): Parent registry. The class registered in
            children registry could be built from parent. Default: None.
        scope (str, optional): The scope of registry. It is the key to search
            for children registry. If not specified, scope will be the name of
            the package where class is defined, e.g. mmdet, mmcls, mmseg.
            Default: None.
    """

    def __init__(self, name, build_func=None, parent=None, scope=None):
        """
        :param name: 就是註冊器的名稱，這裡好像要取什麼都可以
        :param build_func: 註冊器接收到build函數時會透過哪個函數進行找到類對象並且進行實例化
        :param parent: 繼承來自其他註冊器
        :param scope:
        """
        # 已看過

        # 將註冊器名稱記錄下來
        self._name = name
        # 主要的str對應到class或是function就會記錄在_module_dict當中
        self._module_dict = dict()
        # 繼承於這個register的子register
        self._children = dict()
        # 作用域，這個部分我還沒有很清楚，之後再來看看是做什麼用的
        self._scope = self.infer_scope() if scope is None else scope

        # build_func的優先順序
        # self.build_func will be set with the following priority:
        # 1. build_func
        # 2. parent.build_func
        # 3. build_from_cfg
        if build_func is None:
            if parent is not None:
                self.build_func = parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func
        if parent is not None:
            # parent也必須要是一個註冊器
            assert isinstance(parent, Registry)
            # 在parent中添加上children
            parent._add_children(self)
            self.parent = parent
        else:
            self.parent = None

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(name={self._name}, ' \
                     f'items={self._module_dict})'
        return format_str

    @staticmethod
    def infer_scope():
        """Infer the scope of registry.

        The name of the package where registry is defined will be returned.

        Example:
            >>> # in mmdet/models/backbone/resnet.py
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            The scope of ``ResNet`` will be ``mmdet``.

        Returns:
            str: The inferred scope name.
        """
        # 已看過
        # 可使用範圍

        # We access the caller using inspect.currentframe() instead of
        # inspect.stack() for performance reasons. See details in PR #1844
        frame = inspect.currentframe()
        # get the frame where `infer_scope()` is called
        infer_scope_caller = frame.f_back.f_back
        filename = inspect.getmodule(infer_scope_caller).__name__
        split_filename = filename.split('.')
        return split_filename[0]

    @staticmethod
    def split_scope_key(key):
        """Split scope and key.

        The first scope will be split from key.

        Examples:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'

        Return:
            tuple[str | None, str]: The former element is the first scope of
            the key, which can be ``None``. The latter is the remaining key.
        """
        # 已看過
        # 將作用域以及class的名稱分開來
        split_index = key.find('.')
        if split_index != -1:
            # 當有著名作用域時中間會用.隔開，這裡就會分別回傳
            return key[:split_index], key[split_index + 1:]
        else:
            # 沒有.表示作用域為None，輸入的就是key
            return None, key

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def children(self):
        return self._children

    def get(self, key):
        """Get the registry record.

        Args:
            # 傳入的資料
            key (str): The class name in string format.

        Returns:
            class: The corresponding class.
        """
        # 已看過

        # 透過split_scope_key可以將key分成scope與real_key兩個部分，如果原本key當中只有key部分scope就會是None
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope:
            # 如果scope是None或是與當前的scope相同就會走這裡
            # get from self，從self當中獲取該key對應上去的對象
            if real_key in self._module_dict:
                # 獲取到實例對象後就直接回傳
                return self._module_dict[real_key]
        else:
            # get from self._children，從children找實例對象
            if scope in self._children:
                return self._children[scope].get(real_key)
            else:
                # goto root，直接到root找實例對象
                parent = self.parent
                while parent.parent is not None:
                    parent = parent.parent
                return parent.get(key)

    def build(self, *args, **kwargs):
        """
        :param args: tuple(ConfigDict)，裡面放的就會是model的config
        :param kwargs: dict，裡面會有train_cfg以及test_cfg，在新版本中這兩個東西都會是None，都已經寫在model裏面了
        :return:
        """
        return self.build_func(*args, **kwargs, registry=self)

    def _add_children(self, registry):
        """Add children for a registry.

        The ``registry`` will be added as children based on its scope.
        The parent registry could build objects from children registry.

        Example:
            >>> models = Registry('models')
            >>> mmdet_models = Registry('models', parent=models)
            >>> @mmdet_models.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet = models.build(dict(type='mmdet.ResNet'))
        """

        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert registry.scope not in self.children, \
            f'scope {registry.scope} exists in {self.name} registry'
        self.children[registry.scope] = registry

    # deprecated_api_warning是用來檢查一些變數的合法性，沒有對資料進行任何操作
    @deprecated_api_warning(name_dict=dict(module_class='module'))
    def _register_module(self, module, module_name=None, force=False):
        # 透過裝飾器添加新的class時會進入到這裡進行保存
        if not inspect.isclass(module) and not inspect.isfunction(module):
            # 傳入的module必須是一個class類別或是function類別，其他的就會直接報錯
            raise TypeError('module must be a class or a function, '
                            f'but got {type(module)}')

        if module_name is None:
            # 如果沒有傳入指定名稱，這裡就會默認使用傳入的class的名稱
            module_name = module.__name__
        if isinstance(module_name, str):
            # 可以多個名稱對應上同一個class或是function，所以這裡我們將module_name用list包裝
            module_name = [module_name]
        # 遍歷所有的名稱
        for name in module_name:
            if not force and name in self._module_dict:
                # 如果有重複的key出現就會直接報錯，除非我們使用有將force開啟也就是直接覆蓋之前已經有的value
                raise KeyError(f'{name} is already registered '
                               f'in {self.name}')
            # 將key與value保存下來，這樣之後就可以透過key找到對應的class或是function
            self._module_dict[name] = module

    def deprecated_register_module(self, cls=None, force=False):
        warnings.warn(
            'The old API of register_module(module, force=False) '
            'is deprecated and will be removed, please use the new API '
            'register_module(name=None, force=False, module=None) instead.',
            DeprecationWarning)
        if cls is None:
            return partial(self.deprecated_register_module, force=force)
        self._register_module(cls, force=force)
        return cls

    def register_module(self, name=None, force=False, module=None):
        """Register a module.

        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(ResNet)

        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class or function to be registered.
        """
        # 已看過，如果是透過呼叫函數的方式就會是到這裡
        if not isinstance(force, bool):
            # 檢查force的型態
            raise TypeError(f'force must be a boolean, but got {type(force)}')
        # NOTE: This is a walkaround to be compatible with the old api,
        # while it may introduce unexpected bugs.
        if isinstance(name, type):
            # 這個是已經要被淘汰的方式，就是在呼叫register_module沒有明確寫出module參數
            return self.deprecated_register_module(name, force=force)

        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                'name must be either of None, an instance of str or a sequence'
                f'  of str, but got {type(name)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            # 這裡最後還是會call使用裝飾器構建的函數
            self._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            # 透過裝飾器方式將module添加到註冊器會先進入到這裡
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register

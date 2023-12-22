# coding=utf-8
import os
import sys
import psutil
import time
import torch
import pandas as pd
import functools
import pickle
from collections import defaultdict

from .walk import walk_modules

class Profile(object):
    """PyTorch模型的逐层分析器，可以获取模型各层初始化、执行时间和输出数据大小"""

    def __init__(self, model, model_name,enabled=True, use_cuda=False, depth=-1):

        self._model = model
        self.model_name = model_name
        self.enabled = enabled
        self.use_cuda = use_cuda
        self.depth = depth
        

        self.entered = False
        self.exited = False
        self.traces = ()

        self.input = {}

    def __enter__(self):
        if not self.enabled:
            return self
        if self.entered:
            raise RuntimeError("pytorchtool profiler is not reentrant")
        self.entered = True
        self._forwards = {}
        self.input = {}


        self.traces = tuple(walk_modules(self._model, depth=self.depth))

        tuple(map(self._hook_trace, self.traces))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return

        tuple(map(self._remove_hook_trace, self.traces))
        del self._forwards  # remove unnecessary forwards
        self.exited = True

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def _hook_trace(self, trace):
        (name, module) = trace
        _forward = module.forward
        self._forwards[name] = _forward

        @functools.wraps(_forward)
        def wrap_forward(*args, **kwargs):
            print("running----------: ", name)
            self.input[name] = args

            if self.use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                output = _forward(*args, **kwargs)
                end.record()
    
                torch.cuda.synchronize()
            else:
                output = _forward(*args, **kwargs)

            return output

        module.forward = wrap_forward
        return trace

    def _remove_hook_trace(self, trace):
        [name, module] = trace
        module.forward = self._forwards[name]

    def saveInput(self, filePath):
        with open(filePath, "wb") as f:
            pickle.dump(self.input, f)
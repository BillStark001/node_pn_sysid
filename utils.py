from typing import Optional, Union, TypeVar, Generic, List
from numpy.typing import NDArray

import logging

import numpy as np


import pickle
import os
from functools import wraps

import uuid
import base64


def gen_uuid_b64():
  uuid_obj = uuid.uuid4()
  uuid_bytes = uuid_obj.bytes
  b64_encoded = base64.urlsafe_b64encode(
      uuid_bytes).rstrip(b'=').decode('ascii')
  return b64_encoded


def get_logger(name: str, path: str):

  logger = logging.getLogger(name)

  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.DEBUG)

  file_handler = logging.FileHandler(path)
  file_handler.setLevel(logging.INFO)

  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

  console_handler.setFormatter(formatter)
  file_handler.setFormatter(formatter)

  logger.addHandler(console_handler)
  logger.addHandler(file_handler)

  logger.setLevel(logging.DEBUG)

  return logger


class DictWrapper:
  
  _data: dict

  def __init__(self, data):
    if not isinstance(data, dict):
      raise ValueError("Input must be a dictionary.")
    super().__setattr__('_data', data)

  def __getitem__(self, key):
    value = self._data[key]
    if isinstance(value, dict):
      return DictWrapper(value)
    return value
  
  def __setitem__(self, key, value):
    if isinstance(value, DictWrapper):
      value = value._data
    self._data[key] = value

  def __getattr__(self, key):
    try:
      value = self._data[key]
    except KeyError:
      raise AttributeError(f"'DictWrapper' object has no attribute '{key}'")
    if isinstance(value, dict):
      return DictWrapper(value)
    return value
  
  def __setattr__(self, key, value):
    if key == '_data':
      super().__setattr__(key, value)
      return
    if isinstance(value, DictWrapper):
      value = value._data
    self._data[key] = value
    

  def __repr__(self):
    return f"DictWrapper({self._data})"

  def __hash__(self) -> int:
    return hash(self._data)

  def __eq__(self, other):
    return isinstance(other, DictWrapper) and self._data == other._data


USE_CACHE = object()


def cache(cache_file='func_cache.pkl'):
  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      if args and args[0] is USE_CACHE:
        # try load
        if os.path.exists(cache_file):
          try:
            # first try using cache
            with open(cache_file, 'rb') as f:
              args, kwargs = pickle.load(f)
            return func(*args, **kwargs)
          except (pickle.PickleError, EOFError, FileNotFoundError) as e:
            # do nothing if failed
            return None
        else:
          # cache file inexistent
          return None
      else:  # execute normally and cache args
        try:
          with open(cache_file, 'wb') as f:
            pickle.dump((args, kwargs), f)
        except (pickle.PickleError, IOError) as e:
          pass  # do nothing
        result = func(*args, **kwargs)
        return result
    return wrapper
  return decorator

  
T = TypeVar("T")

class ContextManager(Generic[T]):
  def __init__(self, default: T = None):
    self._default = default
    self._context_stack: List[T] = []

  def push(self, context: T):
    self._context_stack.append(context)

  def pop(self):
    if self._context_stack:
      return self._context_stack.pop()
    raise IndexError("pop from empty context stack")

  @property
  def current(self) -> T:
    if self._context_stack:
      return self._context_stack[-1]
    return self._default

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.pop()

  def provide(self, context: T):
    """Context manager method to use a context."""
    self.push(context)
    return self
  
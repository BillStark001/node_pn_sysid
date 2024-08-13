from typing import cast
import miss_hit_core

import torch
import pickle

from miss_hit_core.m_ast import *
from miss_hit_core.m_lexer import MATLAB_Lexer, MATLAB_Latest_Language
from miss_hit_core.config import Config
from miss_hit_core.errors import Message_Handler, Message
from miss_hit_core.m_parser import MATLAB_Parser

from solver_wrapper import ScenarioParameters
from syntax_tree.miss_hit_helper import get_function_by_name, parse_matlab_code
from syntax_tree.src_cfg import generate_cfg
from syntax_tree.src_exec import CodeBlockExecutor
from syntax_tree.src_exec_cfg import exec_func
from syntax_tree.src_rel import RelationRecorder, analyze_relation
from utils import DictWrapper


path = './test_mat_func.m'

with open(path, "r", encoding="utf-8") as f:
  content = f.read()
 
cu = parse_matlab_code(content, path)
 
assert isinstance(cu, Function_File)

func_main = get_function_by_name(cu)

def create_struct(*args):
  d = {}
  for i in range(0, len(args), 2):
    d[args[i]] = args[i + 1]
  return DictWrapper(d)

global_funcs = {
  'size': lambda a: tuple(a.size()),
  'error': print,
  'eig': lambda a: torch.linalg.eig(a).eigenvalues,
  'sum': lambda a: torch.sum(a).reshape((1, 1)),
  # TODO
  'struct': create_struct,
  'isequal': lambda a, b: torch.all(a == b),
  'det': torch.det,
  'disp': print,
  'NaN': torch.nan,
  'numel': torch.numel,
}

rel = analyze_relation(func_main)
cfg = generate_cfg(func_main.n_body)

A = torch.tensor([
  [1, 2, 3],
  [2, 5, 6],
  [3, 6, 9],
], dtype=float)

ret = exec_func(func_main, [A], global_funcs)

print(rel)
print(cfg)

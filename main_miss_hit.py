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
from syntax_tree.exec import CodeBlockExecutor, exec_func
from syntax_tree.rel_analysis import RelationRecorder, analyze_relation
from utils import DictWrapper


path = './network_2bus2gen_ode.m'

with open(path, "r", encoding="utf-8") as f:
  content = f.read()
 
 
class ModifiedMessageHandler(Message_Handler):
  def __init__(self, config):
    super().__init__(config)
  def register_message(self, msg):
    assert isinstance(msg, Message)
    self.process_message(msg)
    if msg.fatal:
      raise Exception(msg.location, msg.message)
    
mh = ModifiedMessageHandler('debug')
  
lexer = MATLAB_Lexer(
  MATLAB_Latest_Language(),
  mh,
  content,
  path, 
  None
)

cfg = Config()
cfg.style_rules = {}

parser = MATLAB_Parser(
  mh,
  lexer,
  cfg,
)

cu = parser.parse_file()
assert isinstance(cu, Function_File)

func_main = cast(Function_Definition, cu.l_functions[0])

rel = analyze_relation(func_main)

with open('./run/solver_copy.pkl', 'rb') as f:
  solver_data = pickle.load(f)[0][0]
  assert isinstance(solver_data, ScenarioParameters)

params_dict = { name: torch.from_numpy(value['Data']) for name, value in solver_data.params.items() }
inputs = dict(y=torch.tensor([[1], [2], [3], [4]], dtype=torch.float64, requires_grad=True))

with torch.enable_grad():
  dydt = exec_func(
    func_main, 
    [
      DictWrapper(inputs),
      DictWrapper(params_dict),
      DictWrapper(dict(
        sin = torch.sin,
        cos = torch.cos,
      ))
    ]
  )

print(dydt)
print(rel)
print(cu)

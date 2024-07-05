from typing import Optional, List, Union


from miss_hit_core.m_ast import *
from miss_hit_core.m_ast import Node


def can_accept_as_cfg(node: Node):
  return isinstance(node, (
    Simple_Assignment_Statement,
    Compound_Assignment_Statement,
    Naked_Expression_Statement,
  ))


class CFG_Node:
  
  def __init__(
    self, 
    name: Optional[str] = None, 
    statements: Optional[List[Union[
      Simple_Assignment_Statement, 
      Naked_Expression_Statement
    ]]] = None
  ):
    self.name = name
    self.statements = list(statements if statements is not None else [])
    assert all(can_accept_as_cfg(x) for x in self.statements)
    
  
  


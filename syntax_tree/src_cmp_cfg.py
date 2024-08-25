from typing import Any, List, Dict, cast, Generator, Any, Tuple

import ast

import torch
import numpy as np
import dataclasses

from miss_hit_core.m_ast import *

from syntax_tree.src_cfg import CFG, CFGNode, CFGType, generate_cfg
from syntax_tree.src_cmp import CodeExecutor, Evaluated
from utils import ContextManager

STOP_ITR = -1

def get_gen(gen):
  try:
    return next(gen), True
  except StopIteration:
    return None, False
  
@dataclasses.dataclass
class ForLoopContext:
  generator: Generator[Any, None, None]
  identifier: str
  expression: ast.expr
  
  has_next: bool = False
  
  def gen_stmt(self):
    return ast.For(
      ast.Name(self.identifier),
      self.expression,
      [],
      [],
      lineno=0,
    )
  
  def next(self):
    return get_gen(self.generator)
  
default_for = ForLoopContext(generator=(x for x in []), identifier='', expression=None)
NO_OPR_TYPES = {
  CFGType.GLOBAL_ENTRY,
  CFGType.GLOBAL_EXIT,
  CFGType.IF_ENTRY,
  CFGType.IF_EXIT,
  CFGType.IF_ACTION_ENTRY,
  CFGType.FOR_CONTINUE,
  
  CFGType.WHILE_ENTRY,
  CFGType.WHILE_CONTINUE,
  CFGType.WHILE_EXIT,
}
# TODO add all types


class CodeControlExecutor(CodeExecutor):
  
  def __init__(self):
    super().__init__()
    self.for_loop = ContextManager(default_for)
    self.cur_stmt_pool = ContextManager([])
    self.pushed_node_set = set()
    
  def eval(self, node: Expression | str, strict_matrix=True) -> Evaluated:
    if isinstance(node, str):
      if node == 'FOR_HAS_NEXT':
        return (self.for_loop.current.has_next, (1, 1), None)
      assert False, 'TODO'
    return super().eval(node, strict_matrix=strict_matrix)

  def exec_node(
    self,
    node: CFGNode,
  ) -> List[ast.stmt]:
    t = node.type
    pool = self.cur_stmt_pool.current
    
    if t == CFGType.SEQUENCE:
      stmts = self.exec(node.stmt_list)
      if node.uid not in self.pushed_node_set:
        pool.extend(stmts)
        self.pushed_node_set.add(node.uid)
      
    elif t == CFGType.FOR_INIT:
      stmt = cast(General_For_Statement, node.stmt_list)
      (iterator, _, itr_expr) = self.eval(stmt.n_expr)
      if isinstance(iterator, torch.Tensor):
        iterator = iterator.view(-1)
      
      if isinstance(iterator, slice):
        iterator = range(iterator.start, iterator.stop, iterator.step or 1)
      context = ForLoopContext(
        generator = (e for e in iterator),
        identifier = stmt.n_ident.t_ident.value,
        expression = itr_expr,
      )
      self.for_loop.push(context)
      for_stmt = context.gen_stmt()
      
      if node.uid not in self.pushed_node_set:
        pool.append(for_stmt)
        self.pushed_node_set.add(node.uid)
        
      self.cur_stmt_pool.push(for_stmt.body)
      
    elif t == CFGType.FOR_ENTRY:
      _for = self.for_loop.current
      element, cont = _for.next()
      _for.has_next = cont
      if cont:
        self.load_vars(**{_for.identifier: element})
    
    elif t == CFGType.FOR_EXIT:
      self.for_loop.pop()
      self.cur_stmt_pool.pop()
      
    elif t in NO_OPR_TYPES:
      pass
    
    else:
      assert False, 'TODO'
    
    self.pushed_node_set.add(node.uid)

  def get_next_node(
    self,
    node_id: int,
    cfg: CFG,
  ) -> int:
    opts = cfg.next_node(node_id)
    for id, cond in opts: # evaluate all edges by precedence
      if cond is None: # unconditional jump
        return id
      # else cond is not None, evaluate it
      (val, _, expr) = self.eval(cond)
      cond_eval = False
      if isinstance(val, torch.Tensor):
        cond_eval = torch.all(val)
      elif isinstance(val, bool):
        cond_eval = val
      else:
        cond_eval = not not val
      if cond_eval:
        return id
    return STOP_ITR


def exec_func(
  src: Function_Definition, 
  params: List | None = None, 
  scope: Dict | None = None, 
):
  ex = CodeControlExecutor()
  
  # load global scope
  if scope is not None:
    ex.load_vars(**scope)
    
  # load parameters
  params_dict = {}
  for i, n in enumerate(src.n_sig.l_inputs):
    nv: str = n.t_ident.value
    params_dict[nv] = params[i]
  ex.load_vars(**params_dict)
  
  cfg = generate_cfg(src.n_body)
  cur_node_id = cfg.entry_id
  
  # CFG
  stmt_pool = []
  ex.pushed_node_set = set()
  with ex.cur_stmt_pool.provide(stmt_pool):
    while cur_node_id != cfg.exit_id:
      cur_node = cfg.node(cur_node_id)
      ex.exec_node(cur_node)
      next_node = ex.get_next_node(cur_node_id, cfg)
      if next_node == STOP_ITR:
        assert False, 'Should not happen'
      cur_node_id = next_node
  
  # gather and return outputs
  ret_value = None
  if len(src.n_sig.l_outputs) != 0:
    ret = []
    for n in src.n_sig.l_outputs:
      nv: str = n.t_ident.value
    ret.append(ex.vars[nv])
    ret_value = ret[0] if len(src.n_sig.l_outputs) == 1 else ret
  
  return ret_value, stmt_pool

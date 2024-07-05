from typing import Any, List

import torch

from miss_hit_core.m_ast import *
from miss_hit_core.m_ast import Node

from syntax_tree.ast import FunctionASTVisitor
from syntax_tree.eval_opr import eval_binary_opr, eval_unary_opr

def is_simple_node(node: Node):
  return isinstance(node, (
    Simple_Assignment_Statement,
    Compound_Assignment_Statement,
    Naked_Expression_Statement,
    Sequence_Of_Statements,
    Expression,
    Row_List,
    Row,
  ))

def eval_literal(node: Literal):
  if isinstance(node, Number_Literal):
    v = node.t_value.value
    return int(v) if v.isdigit() else float(v)
  if isinstance(node, Char_Array_Literal):
    return str(node.t_string.value)
  if isinstance(node, String_Literal):
    return str(node.t_string.value)
  assert False


class CodeBlockExecutor(FunctionASTVisitor):
  
  def __init__(self):
    super().__init__()
    self.vars = {}
    
  def exec(self, node: Sequence_Of_Statements):
    return node.visit(None, self, 'Root')
    
  def visit(self, node: Node, n_parent: Node | None, relation: str):
    assert is_simple_node(node)
    
    return super().visit(node, n_parent, relation)
  
  def subs_ref(self, path: List):
    pass
  
  def subs_assign(self, path: List, obj: List):
    pass
  
  def eval_name(self, node: Name, results: List, relations: List[str]):
    path = None
    if isinstance(node, Identifier):
      path = [node.t_ident.value]
    elif isinstance(node, (Selection, Dynamic_Selection)):
      rp, rf = results
      path = rp + rf
    elif isinstance(node, (Reference, Cell_Reference)):
      rp, *rf = results
      path = rp
      for rfc in rf:
        path += rfc
    else:
      raise 'TODO'

    return path
    
    
  def eval_expression(self, node: Expression, params: List, relations: List[str]):
    if isinstance(node, Literal):
      return eval_literal(node)
    
    if isinstance(node, Unary_Operation):
      return eval_unary_opr(node.t_op.value, params[0])
    if isinstance(node, Binary_Operation):
      return eval_binary_opr(node.t_op.value, params[0], params[1])
    
    if isinstance(node, Reshape):
      return slice(None)
    if isinstance(node, Range_Expression):
      first = params[0]
      stride = params[1] if node.n_stride else 1
      last = params[-1]
      return slice(first, last + stride, stride)
    
    if isinstance(node, Matrix_Expression):
      # params should be row list
      return torch.concat(params, dim=0)
    if isinstance(node, Cell_Expression):
      # treat it as a cascaded list
      return params
    
    raise 'TODO'
    
    
  def handle_rows(self, node: Row | Row_List, params: List):
    if isinstance(node, Row):
      ret = []
      for i, n in enumerate(node.l_items):
        ret_ = self.subs_ref(params[i]) if isinstance(n, Name) else n
        ret.append(ret_)
      return ret
    # else it is a row list
    return params
        
    
  def on_visited(self, node: Node, relation: str, results: List, relations: List[str]) -> Any:
    if isinstance(node, (Name, Function_Call)):
      return self.eval_name(node, results, relations)
    if isinstance(node, (Row, Row_List)):
      return self.handle_rows(node, results)
    if isinstance(node, Expression):
      return self.eval_expression(node, results, relations)
    
    if isinstance(node, (Naked_Expression_Statement, Sequence_Of_Statements)):
      # the expression is already evaluated
      return None
    
    if isinstance(node, (Simple_Assignment_Statement, Compound_Assignment_Statement)):
      r_lhs, r_rhs = results
      r_expr = r_rhs
      if isinstance(node.n_rhs, Name):
        r_expr = self.subs_ref(r_rhs)
      if isinstance(node.n_rhs, Literal):
        r_expr = torch.tensor([[r_rhs]]) # so that all vars are matrices
        
      if isinstance(node, Simple_Assignment_Statement):
        self.subs_assign(r_lhs, r_expr)
      else:
        for i, n in enumerate(r_lhs):
          self.subs_assign(n, r_expr[i]) # TODO is this correct?
      return None
    
    raise 'What the hell?'
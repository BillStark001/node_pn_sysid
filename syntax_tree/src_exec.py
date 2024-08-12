from typing import Any, List, Dict

import torch
import numpy as np

from miss_hit_core.m_ast import *
from miss_hit_core.m_ast import Node

from syntax_tree.ast import FunctionASTVisitor
from syntax_tree.opr_eval import eval_binary_opr, eval_subs_ref_arr, eval_unary_opr


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

  def load_vars(self, **vars: Any):
    for k, v in vars.items():
      self.vars[k] = v

  def visit(self, node: Node, n_parent: Node | None, relation: str):
    assert is_simple_node(node)

    return super().visit(node, n_parent, relation)

  def subs_ref(self, path: List, root_obj=None):
    if root_obj is None:
      if isinstance(path[0], str):
        ret = self.vars[path[0]]
        path_lst = path[1:]
      else:
        assert len(path) == 1
        return path[0]
    else:
      ret = root_obj
      path_lst = path
    for p in path_lst:
      if isinstance(p, str):
        ret = getattr(ret, p)
      elif isinstance(p, int):
        ret = eval_subs_ref_arr(ret, [p])
      elif isinstance(p, tuple):
        opr, path = p
        path_obj = self.subs_ref(path)
        if opr == 'ds':
          ret = getattr(ret, path_obj)
        elif callable(ret) and opr == 'r':
          ret = ret(path_obj)
        else:  # r, cr
          ret = eval_subs_ref_arr(ret, [path_obj])
      else:
        assert False, 'TODO'
    return ret

  def subs_assign(self, path: List, obj: List):
    assert isinstance(path[0], str)
    if len(path) < 2:
      self.vars[path[0]] = obj
      return
    # at least 2
    ret = self.vars[path[0]]
    final_path = path[-1]
    for p in path[1:-1]:
      if isinstance(p, str):
        ret = getattr(ret, p)
      if isinstance(p, tuple):
        opr, path = p
        path_obj = self.subs_ref(path, ret)
        if opr == 'ds':
          ret = getattr(ret, path_obj)
        else:  # r, cr
          ret = ret[path_obj]
    if isinstance(final_path, str):
      setattr(ret, final_path, obj)
    else:  # tuple
      opr, path = final_path
      path_obj = self.subs_ref(path, ret)
      if opr == 'ds':
        setattr(ret, path_obj, obj)
      else:  # r, cr
        eval_subs_ref_arr(ret, path_obj, obj)

  def eval_name(self, node: Name, relation: str, results: List, relations: List[str]):
    path = None
    if isinstance(node, Identifier):
      path = [node.t_ident.value]
    elif isinstance(node, (Selection, Dynamic_Selection)):
      rp, rf = results
      path = rp + rf if isinstance(node, Selection) else rp + [('ds', rf)]
    elif isinstance(node, (Reference, Cell_Reference)):
      rp, *rf = results
      path = rp + [('r' if isinstance(node, Reference) else 'cr', rf)]
    else:
      assert False, 'TODO'

    if not isinstance(node.n_parent, Name) \
        and not (isinstance(node.n_parent, (Simple_Assignment_Statement, Compound_Assignment_Statement))
                 and relation == 'LHS'):
      return self.subs_ref(path)

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

    if isinstance(node, (Matrix_Expression, Cell_Expression)):
      return params[0]

    assert False, 'TODO'

  def handle_rows(self, node: Row | Row_List, params: List):
    if isinstance(node, Row):
      if len(params) == 0:
        return None
      if node.n_parent is not None and isinstance(node.n_parent.n_parent, Matrix_Expression):
        return torch.cat(params, dim=-1) if len(params) > 1 else params[0]
      return params
    # else it is a row list
    params_f = [x for x in params if x is not None]
    if isinstance(node.n_parent, Matrix_Expression):
      return torch.cat(params_f, dim=-2)
    return params_f

  def on_visited(self, node: Node, relation: str, results: List, relations: List[str]) -> Any:
    if isinstance(node, (Name, Function_Call)):
      return self.eval_name(node, relation, results, relations)
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
      if isinstance(node.n_rhs, Literal):
        r_expr = np.array([[r_rhs]])  # so that all vars are matrices

      if isinstance(node, Simple_Assignment_Statement):
        self.subs_assign(r_lhs, r_expr)
      else:
        for i, n in enumerate(r_lhs):
          self.subs_assign(n, r_expr[i])  # TODO is this correct?
      return None

    assert False, 'TODO'


def exec_func(
  src: Function_Definition, 
  params: List | None = None, 
  scope: Dict | None = None, 
):
  ex = CodeBlockExecutor()
  if scope is not None:
    ex.load_vars(**scope)
  params_dict = {}
  for i, n in enumerate(src.n_sig.l_inputs):
    nv: str = n.t_ident.value
    params_dict[nv] = params[i]
  ex.load_vars(**params_dict)
  # TODO CFG
  ex.exec(src.n_body)
  if len(src.n_sig.l_outputs) == 0:
    return
  ret = []
  for n in src.n_sig.l_outputs:
    nv: str = n.t_ident.value
  ret.append(ex.vars[nv])
  return ret[0] if len(src.n_sig.l_outputs) == 1 else ret

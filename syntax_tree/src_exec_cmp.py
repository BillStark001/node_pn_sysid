from typing import Any, List, Dict, TypeAlias, Tuple

import ast
import json

import torch
import numpy as np

from miss_hit_core.m_ast import *
from miss_hit_core.m_ast import Node

from syntax_tree.ast import FunctionASTVisitor
from syntax_tree.opr_eval import BinaryOprRecorder, UnaryOprRecorder, eval_binary_opr, eval_subsref_arr, eval_unary_opr
from syntax_tree.opr_shape import binary_shape, unary_shape
from utils.ast_template import CodeTemplate


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


Evaluated: TypeAlias = Tuple[
    Any,  # value
    Tuple[int, ...],  # shape
    ast.AST  # python ast
]

tpl_ref_obj = CodeTemplate('a.b', no_expr=True)
tpl_ref_arr = CodeTemplate('a[b]', no_expr=True)
tpl_asgn_name = CodeTemplate('b = c', mode='exec')
tpl_asgn_obj = CodeTemplate('a.b = c', mode='exec')
tpl_asgn_arr = CodeTemplate('a[b] = c', mode='exec')
tpl_call = CodeTemplate('a(b)', no_expr=True)

tpl_create_tensor = CodeTemplate(
    'torch.tensor(elem, dtype=dtype)', dtype=ast.Name('float'),
    no_expr=True,
)

# func call


def func_call(object: Evaluated, target: Evaluated) -> Evaluated:
  ov, os, oe = object
  tv, ts, te = target
  return (
      ov(*tv),
      os,
      ast.Call(oe, te.elts if isinstance(te, ast.Tuple) else [te], [])
  )

# subsref


def subsref_obj(object: Evaluated, target: str | Evaluated) -> Evaluated:
  ov, os, oe = object
  if isinstance(target, str):  # static reference
    return (getattr(ov, target), os, tpl_ref_obj.create(a=oe, b=ast.Name(target))[0])
  # else, it is a dynamic reference
  tv, ts, te = target
  return (getattr(ov, tv), os, tpl_ref_arr.create(a=oe, b=te)[0])


def subsref_list(object: Evaluated, target: Evaluated) -> Evaluated:
  ov, os, oe = object
  tv, ts, te = target  # te: Tuple
  return (
      ov[*tv], os,
      tpl_ref_arr.create(
          a=oe, b=ast.Tuple(list(te)) if isinstance(te, (tuple, list)) else te)[0]
  )


def subsref_arr(object: Evaluated, target: Evaluated) -> Evaluated:
  ov, os, oe = object
  tv, ts, te = target  # te: Tuple
  return (
      eval_subsref_arr(ov, tv),
      os,
      ast.Call(ast.Name('eval_subsref_arr'), [oe, te], []),
  )

# subsasgn


def subsasgn_obj(object: Evaluated | None, target: str | Evaluated, value: Evaluated) -> Evaluated:
  is_static = isinstance(target, str)
  ov, os, oe = object if object is not None else (None, None, None)
  vv, vs, ve = value
  tv, ts, te = (target, (1, 1), ast.Name(target)) if is_static else target
  setattr(ov, tv, vv)
  return (ov, os, ((tpl_asgn_name if oe is None else tpl_asgn_obj)\
    if is_static else tpl_asgn_arr)
      .create(a=oe, b=te, c=ve)[0])


def subsasgn_list(object: Evaluated, target: Evaluated, value: Evaluated) -> Evaluated:
  ov, os, oe = object
  tv, ts, te = target  # te: Tuple
  vv, vs, ve = value
  ov[*tv] = vv
  return (
      ov, os,
      tpl_asgn_arr.create(
          a=oe, b=ast.Tuple(list(te)) if isinstance(te, (tuple, list)) else te, c=ve)[0]
  )


def subsasgn_arr(object: Evaluated, target: Evaluated, value: Evaluated) -> Evaluated:
  ov, os, oe = object
  tv, ts, te = target  # te: Tuple
  vv, vs, ve = value
  rv = eval_subsref_arr(ov, tv, vv)
  return (
      rv,
      os,
      ast.Call(ast.Name('eval_subsref_arr'), [oe, te, ve], []),
  )

# arithmetic


def expr_unary(expr: str, value: Evaluated) -> Evaluated:
  vv, vs, ve = value
  rv, r_ast = UnaryOprRecorder.prune_t(expr, vv)
  r_ret = r_ast.body[0]
  assert isinstance(r_ret, ast.Return), 'WTF'
  re = CodeTemplate(r_ret.value).create(elem=ve)[0]
  return (rv, unary_shape(vs), re)


def expr_binary(expr: str, v1: Evaluated, v2: Evaluated) -> Evaluated:
  vv, vs, ve = v1
  wv, ws, we = v2
  rv, r_ast = BinaryOprRecorder.prune_t(expr, vv, wv)
  r_ret = r_ast.body[0]
  assert isinstance(r_ret, ast.Return), 'WTF'
  re = CodeTemplate(r_ret.value).create(elem1=ve, elem2=we)[0]
  return (rv, binary_shape(vs, ws), re)


def expr_literal(node: Literal, strict_matrix=True) -> Evaluated:
  ret = None
  expr = None
  if isinstance(node, Number_Literal):
    v = node.t_value.value
    ret = int(v) if v.isdigit() else float(v)
    if strict_matrix:
      expr = tpl_create_tensor.create(elem=ast.List([
        ast.List([ast.Constant(ret)])]))[0]
      ret = torch.tensor([[ret]], dtype=float)
  elif isinstance(node, Char_Array_Literal):
    ret = str(node.t_string.value)
  elif isinstance(node, String_Literal):
    ret = str(node.t_string.value)
  else:
    assert False, 'TODO'
  expr = expr if expr is not None else ast.Constant(ret)
  return (ret, (1, 1), expr)


def _shape(val: Any):
  if isinstance(val, torch.Tensor):
    return tuple(val.size())
  return (1, 1)

# matrices


class CodeExecutor:

  def __init__(self):
    self.vars = {}

  def load_vars(self, **vars: Any):
    for k, v in vars.items():
      self.vars[k] = v

  def exec(self, node: Sequence_Of_Statements):
    
    expr_list = []
    
    for stmt in node.l_statements:
      if isinstance(stmt, Naked_Expression_Statement):
        _, _, expr = self.eval_expression(stmt.n_expr)
        expr_list.append(expr)
      elif isinstance(stmt, (Simple_Assignment_Statement, Compound_Assignment_Statement)):
        exprs = self.eval_subsasgn(stmt)
        expr_list.extend(exprs)
        
      # TODO what about other types?
    
    for i in range(len(expr_list)):
      if isinstance(expr_list[i], ast.expr):
        expr_list[i] = ast.Expr(expr_list[i])
    
    return expr_list

  def eval_subsasgn(self, node: Simple_Assignment_Statement | Compound_Assignment_Statement):
    is_simple = isinstance(node, Simple_Assignment_Statement)
    rhs, rhs_shape, rhs_expr = self.eval_expression(node.n_rhs)
    lhs_lst: List[Name] = [node.n_lhs] if is_simple else node.l_lhs
    rhs_lst = [rhs] if is_simple else list(rhs)
    
    # commit assignments
    lhs_exprs = []
    trailing_exprs = []
    
    for i in range(len(lhs_lst)):
      lhs_i = lhs_lst[i]
      rhs_i = rhs_lst[i]
      
      if isinstance(lhs_i, Identifier):
        lhs_exprs.append(ast.Name(lhs_i.t_ident.value))
        self.vars[lhs_i.t_ident.value] = rhs_i # lhs_i
        
      elif isinstance(lhs_i, (Selection, Dynamic_Selection)):
        ev, es, ee = self.eval_expression(lhs_i.n_prefix)
        tv, ts, te = (lhs_i.n_field.t_ident.value, (1, 1), ast.Name(lhs_i.n_field.t_ident.value))\
          if isinstance(lhs_i, Selection) else self.eval_expression(lhs_i.n_field)
        setattr(ev, tv, rhs_i)
        lhs_exprs.append(ast.Attribute(ee, te)) # ee.target
        
      elif isinstance(lhs_i, (Cell_Reference, Reference)):
        is_cell = isinstance(lhs_i, Cell_Reference)
        args = [self.eval_expression(x) for x in lhs_i.l_args]
        args_object = [x[0] for x in args]
        args_tuple = ast.Tuple([x[2] for x in args]) if len(args) > 1 else args[0][2]

        lhs_value, _, lhs_expr = self.eval_expression(lhs_i.n_ident)
        
        if is_cell:
          lhs_value[*args_object] = rhs_i
          lhs_exprs.append(ast.Subscript(lhs_expr, args_tuple))
        else:
          # create a temporary variable
          # evaluate the subsref at last
          eval_subsref_arr(lhs_value, args_object, rhs_i)
          k = ast.Name(f'__temporary_subsasgn_{i}')
          lhs_exprs.append(k)
          trailing_exprs.append( \
            ast.Call(ast.Name('eval_subsref_arr'), [lhs_expr, args_tuple, k], []))
        
      else:
        assert False, 'WTF'
        
    # create expression
    if is_simple:
      ret = ast.Assign([lhs_exprs[0]], rhs_expr, lineno=-1)
    else:
      ret = ast.Assign([ast.Tuple(lhs_exprs)], rhs_expr, lineno=-1)
    return [ret] + trailing_exprs
    
     
    # TODO

  def eval_expression(self, node: Expression, strict_matrix=True) -> Evaluated:

    # literal and oprs

    if isinstance(node, Literal):
      return expr_literal(node, strict_matrix=strict_matrix)

    if isinstance(node, Unary_Operation):
      elem = self.eval_expression(node.n_expr)
      return expr_unary(node.t_op.value, elem)

    if isinstance(node, Binary_Operation):
      elem1 = self.eval_expression(node.n_lhs)
      elem2 = self.eval_expression(node.n_rhs)
      return expr_binary(node.t_op.value, elem1, elem2)

    if isinstance(node, Reshape):
      return (slice(None), (0, 0), ast.Slice())
    
    if isinstance(node, Range_Expression):
      fv, fs, fe = self.eval_expression(node.n_first)
      sv, ss, se = self.eval_expression(node.n_stride) if node.n_stride\
          else (None, None, None)
      lv, ls, le = self.eval_expression(node.n_last)
      return (
          slice(fv, lv + 1, sv),
          (1, 1),
          ast.Slice(fe, se, le)
      )

    # identifier

    if isinstance(node, Identifier):
      # references should never reach here
      value = self.vars[node.t_ident.value]
      shape = _shape(value)
      expr = ast.Name(node.t_ident.value)
      return (value, shape, expr)

    if isinstance(node, Selection):
      object = self.eval_expression(node.n_prefix)
      target = node.n_field.t_ident.value
      return subsref_obj(object, target)

    if isinstance(node, Dynamic_Selection):
      object = self.eval_expression(node.n_prefix)
      target = self.eval_expression(node.n_field)
      return subsref_obj(object, target)

    if isinstance(node, (Cell_Reference, Reference)):
      is_cell = isinstance(node, Cell_Reference)
      args = [self.eval_expression(x) for x in node.l_args]
      args_object = [x[0] for x in args]
      args_tuple = ast.Tuple([x[2] for x in args]) if len(args) > 1 else args[0][2]

      lhs = self.eval_expression(node.n_ident)
      is_call = callable(lhs[0])

      return (subsref_list if is_cell else (
          func_call if is_call else subsref_arr
      ))(
          # this must be a name, i.e. identifier, reference, ...
          lhs,
        (args_object, (1, 1), args_tuple)
      )

    if isinstance(node, (Matrix_Expression, Cell_Expression)):
      return self.eval_cols(node)

    assert False, 'TODO'

  def eval_rows(self, node: Row, is_cell: bool) -> Evaluated:
    if not node.l_items:
      return None  # dummy row
    items = [
        self.eval_expression(x, strict_matrix=True)
        for x in node.l_items
    ]
    if len(items) == 1:
      return items[0][0], items[0][1], ast.List([items[0][2]])
    # else there are multiple items, concatenation needed
    # string arrays are currently unsupported
    vals = [x[0] for x in items]
    if not is_cell:
      vals: torch.Tensor = torch.cat(vals, dim=-1)
    dims = tuple(vals.size())
    expr = ast.List([x[2] for x in items])
    return vals, dims, expr

  def eval_cols(self, node: Matrix_Expression | Cell_Expression) -> Evaluated:
    is_cell = isinstance(node, Cell_Expression)
    rows = [self.eval_rows(x, is_cell) for x in node.n_content.l_items]
    if len(rows) == 0:
      if is_cell:
        return [], (0, 0), ast.List([])
      return torch.tensor([], float), (0, 0), ast.parse('torch.tensor([], float)').body[0]
    if len(rows) == 1:
      cv, cs, ce = rows[0]
      if is_cell:
        return cv, cs, ast.List([ce])
      return cv, cs, tpl_create_tensor.create(elem=ast.List([ce]))[0]
    cv = [x[0] for x in rows]
    cs = rows[0][1]  # TODO
    ce = ast.List([x[2] for x in rows])
    if is_cell:
      return cv, cs, ce
    return torch.cat(cv, dim=-2), cs, tpl_create_tensor.create(elem=ce)[0]


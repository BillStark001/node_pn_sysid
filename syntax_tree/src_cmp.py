from typing import Any, List, Dict, TypeAlias, Tuple

import ast
import json

import functools

import torch
import numpy as np

from miss_hit_core.m_ast import *
from miss_hit_core.m_ast import Node

from syntax_tree.ast import FunctionASTVisitor
from syntax_tree.opr_eval import BinaryOprRecorder, UnaryOprRecorder, concat_cells_col, concat_cells_row, eval_binary_opr, eval_subsref_arr, eval_unary_opr
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
tpl_create_tensor_no_dtype = CodeTemplate(
    'torch.tensor(elem)',
    no_expr=True,
)
tpl_cat_tensor = CodeTemplate(
  'torch.cat(tensors, dim=dim)', no_expr=True,
)
tpl_slice_tensor_row = CodeTemplate(
  't[..., i, :]', no_expr=True,
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


# TODO this is not correct
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


# arithmetic


def expr_unary(expr: str, value: Evaluated) -> Evaluated:
  vv, vs, ve = value
  rv, r_ast = UnaryOprRecorder.prune_t(expr, vv)
  r_ret = r_ast.body[0]
  assert isinstance(r_ret, ast.Return), 'WTF'
  re = CodeTemplate(r_ret.value).create(elem=ve)[0]
  return (rv, unary_shape(expr, vs), re)


def expr_binary(expr: str, v1: Evaluated, v2: Evaluated) -> Evaluated:
  vv, vs, ve = v1
  wv, ws, we = v2
  rv, r_ast = BinaryOprRecorder.prune_t(expr, vv, wv)
  r_ret = r_ast.body[0]
  assert isinstance(r_ret, ast.Return), 'WTF'
  re = CodeTemplate(r_ret.value).create(elem1=ve, elem2=we)[0]
  return (rv, binary_shape(expr, vs, ws), re)


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

  def exec(self, node: Sequence_Of_Statements) -> List[ast.stmt]:
    
    expr_list = []
    
    for stmt in node.l_statements:
      if isinstance(stmt, Naked_Expression_Statement):
        _, _, expr = self.eval(stmt.n_expr)
        expr_list.append(expr)
      elif isinstance(stmt, (Simple_Assignment_Statement, Compound_Assignment_Statement)):
        exprs = self.subsasgn(stmt)
        expr_list.extend(exprs)
        
      # TODO what about other types?
    
    for i in range(len(expr_list)):
      if isinstance(expr_list[i], ast.expr):
        expr_list[i] = ast.Expr(expr_list[i])
    
    return expr_list
  

  def subsasgn(self, node: Simple_Assignment_Statement | Compound_Assignment_Statement) -> List[ast.stmt]:
    is_simple = isinstance(node, Simple_Assignment_Statement)
    rhs, rhs_shape, rhs_expr = self.eval(node.n_rhs)
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
        
      elif isinstance(lhs_i, Selection):
        ev, es, ee = self.eval(lhs_i.n_prefix)
        target = lhs_i.n_field.t_ident.value
        setattr(ev, target, rhs_i)
        lhs_exprs.append(ast.Attribute(ee, target)) # ee.target
        
      elif isinstance(lhs_i, Dynamic_Selection):
        ev, es, ee = self.eval(lhs_i.n_prefix)
        tv, ts, te = self.eval(lhs_i.n_field)
        setattr(ev, tv, rhs_i)
        lhs_exprs.append(ast.Subscript(ee, te)) # ee.target
        
      elif isinstance(lhs_i, (Cell_Reference, Reference)):
        is_cell = isinstance(lhs_i, Cell_Reference)
        args = [self.eval(x) for x in lhs_i.l_args]
        args_object = [x[0] for x in args]
        args_tuple = ast.Tuple([x[2] for x in args]) if len(args) > 1 else args[0][2]

        lhs_value, _, lhs_expr = self.eval(lhs_i.n_ident)
        
        if is_cell:
          # TODO this is not correct
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

  def eval(self, node: Expression, strict_matrix=True) -> Evaluated:

    # literal and oprs

    if isinstance(node, Literal):
      return expr_literal(node, strict_matrix=strict_matrix)

    if isinstance(node, Unary_Operation):
      elem = self.eval(node.n_expr)
      return expr_unary(node.t_op.value, elem)

    if isinstance(node, Binary_Operation):
      elem1 = self.eval(node.n_lhs)
      elem2 = self.eval(node.n_rhs)
      return expr_binary(node.t_op.value, elem1, elem2)

    if isinstance(node, Reshape):
      return (slice(None), (0, 0), ast.Slice())
    
    if isinstance(node, Range_Expression):
      fv, fs, fe = self.eval(node.n_first, strict_matrix=False)
      sv, ss, se = self.eval(node.n_stride, strict_matrix=False) if node.n_stride\
          else (None, None, None)
      lv, ls, le = self.eval(node.n_last, strict_matrix=False)
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
      object = self.eval(node.n_prefix)
      target = node.n_field.t_ident.value
      return subsref_obj(object, target)

    if isinstance(node, Dynamic_Selection):
      object = self.eval(node.n_prefix)
      target = self.eval(node.n_field)
      return subsref_obj(object, target)

    if isinstance(node, (Cell_Reference, Reference)):
      is_cell = isinstance(node, Cell_Reference)
      args = [self.eval(x) for x in node.l_args]
      args_object = [x[0] for x in args]
      args_tuple = ast.Tuple([x[2] for x in args]) if len(args) > 1 else args[0][2]

      lhs = self.eval(node.n_ident)
      is_call = callable(lhs[0])
      # TODO strict matrix adaption

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

  # TODO refine eval_rows and eval_cols

  def eval_rows(self, node: Row, is_cell_syntax: bool) -> Evaluated:
    if not node.l_items:
      return None  # dummy row
    items = [
        self.eval(x, strict_matrix=False)
        for x in node.l_items
    ]
    # sanity check
    is_mat_row = all(isinstance(x, (int, float, bool, torch.Tensor)) for x, _, __ in items)
    # this means it is not cell syntax {A}, but cell concatenation [A, B]
    is_cell_row = all(isinstance(x, list) for x, _, __ in items)
    if not is_mat_row and not is_cell_row:
      raise Exception('Unsupported row concatenation for multiple types of arrays')
    is_mono_row = not is_cell_row and all(isinstance(x, (int, float, bool)) \
      or (isinstance(x, torch.Tensor) and x.size(-2) == 1) for x, _, __ in items)
    
    if is_cell_syntax: # it returns a cell array row
      vals = [x[0] for x in items]
      dims = tuple(vals.size())
      expr = ast.List([x[2] for x in items])
    
    elif is_cell_row: # it returns a cell array form
      # TODO support cell arrays
      vals = concat_cells_row([x[0] for x in items])
      dims = (len(vals), len(vals[0]))
      expr = ast.Call(ast.Name('concat_cells_row'), [x[2] for x in items], [], lineno=0)
    
    elif is_mono_row:
      if len(items) == 1:
        i_v, i_s, i_e = items[0]
        if isinstance(i_v, torch.Tensor):
          is_mono_row = False
          vals, dims, expr = i_v, i_s, i_e
        else:
          vals = [i_v]
          dims = (1, 1)
          expr = ast.List([i_e]) # TODO how to solve batched data?
      else:
        vals = []
        dims_0 = 0
        expr_tmp = []
        for i_v, i_s, i_e in items:
          if isinstance(i_v, torch.Tensor):
            vals.extend(i_v[..., 0, :])
            dims_0 += i_v.shape(-2)
            expr_tmp.append(ast.Starred(tpl_slice_tensor_row.create(t=i_e, i=ast.Constant(0))[0]))
          else:
            vals.append(i_v)
            dims_0 += 1
            expr_tmp.append(i_e)
        dims = (1, dims_0)
        expr = ast.List(expr_tmp)        
    
    else: # all items must be tensors, otherwise it does not make sense
      if len(items) == 1:
        vals, dims, expr = items[0] # just return the 2D tensor
      # else there are multiple items, concatenation needed
      # string arrays are unsupported
      else:
        vals: torch.Tensor = torch.cat([x[0] for x in items], dim=-1)
        dims = tuple(vals.size())
        expr = tpl_cat_tensor.create(tensors=[x[2] for x in items], dim=-1)[0]
      
    return (vals, dims, expr), (is_cell_row, is_mono_row)

  def eval_cols(self, node: Matrix_Expression | Cell_Expression) -> Evaluated:
    is_cell = isinstance(node, Cell_Expression)
    rows = [self.eval_rows(x, is_cell) for x in node.n_content.l_items]
    if len(rows) == 0:
      if is_cell:
        return [], (0, 0), ast.List([])
      return torch.tensor([], float), (0, 0), ast.parse('torch.tensor([], float)').body[0]
    if len(rows) == 1:
      (cv, cs, ce), (c_cr, c_mr) = rows[0]
      if is_cell:
        return cv, cs, ast.List([ce])
      elif c_cr:
        return cv, cs, ce
      elif c_mr:
        return torch.tensor([cv]), cs, tpl_create_tensor_no_dtype.create(
          elem = ast.List([ce])
        )[0]
      # else it is already a tensor
      return cv, cs, ce
    # else
    rows_elems = [x[0] for x in rows]
    any_cell_rows = any(x[1][0] for x in rows)
    any_mat_rows = any(not x[1][0] for x in rows)
    all_mono_rows = all(x[1][1] for x in rows)
    if any_cell_rows and any_mat_rows:
      raise Exception('Unsupported column concatenation')
    # otherwise it must be all cell rows or all mat rows
    
    if any_cell_rows:
      cv = concat_cells_col([x[0] for x in rows_elems])
      cs = (len(cv), len(cv[0]))
      ce = ast.Call(ast.Name('concat_cells_col'), [x[2] for x in rows_elems], [], lineno=0)
      return cv, cs, ce
    # else, it must be all_mat_rows
    elif all_mono_rows:
      cv = torch.tensor([x[0] for x in rows_elems], dtype=float)
      cs = tuple(cv.size())
      ce = tpl_create_tensor.create(elem=ast.List(
        [x[2] for x in rows_elems]))[0]
      return cv, cs, ce
    
    # else just cat
    cv = [x[0] for x in rows_elems]
    ce_tmp = [x[2] for x in rows_elems]
    for i, (_, mono_row) in enumerate(x[1] for x in rows):
      if mono_row:
        cv[i] = [cv[i]]
        ce_tmp[i] = ast.List([ce_tmp[i]])
    ce = ast.List([x[2] for x in rows])
    cv = torch.cat(cv, dim=-2)
    ce = tpl_cat_tensor.create(tensors=ce, dim=-2)[0]
    return cv, tuple(cv.size()), ce


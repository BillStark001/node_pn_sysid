from typing import List, Any
import torch
import numpy as np


import functools
from utils.prune import FunctionTracer


def eval_unary_opr(opr: str, elem: torch.Tensor) -> torch.Tensor:

  if opr in ('ctranspose', "'"):
    return torch.conj(elem).transpose(-2, -1)
  elif opr in ('transpose', ".'"):
    return torch.transpose(elem, -2, -1)
  elif opr in ('uplus', '+'):
    return elem
  elif opr in ('uminus', '-'):
    return -elem
  elif opr in ('~', '!', 'not'):
    return torch.logical_not(elem)
  assert False, 'TODO'

UnaryOprRecorder = FunctionTracer(eval_unary_opr, torch=torch)

def eval_binary_opr(opr: str, elem1: torch.Tensor, elem2: torch.Tensor) -> torch.Tensor:

  if opr in ('+', 'plus'):
    return elem1 + elem2
  elif opr in ('-', 'minus'):
    return elem1 - elem2

  elif opr in ('.*', 'times'):
    return elem1 * elem2
  elif opr in ('*', 'mtimes'):
    return elem1 @ elem2
  elif opr in ('./', 'rdivide'):
    return elem1 / elem2
  elif opr in ('.\\', 'ldivide'):
    return elem2 / elem1
  elif opr in ('/', 'mrdivide'):
    return elem1 @ torch.inverse(elem2)
  elif opr in ('\\', 'mldivide'):
    return torch.inverse(elem2) @ elem1

  elif opr in ('.^', 'power'):
    return elem1 ** elem2
  elif opr in ('^', 'mpower'):
    return torch.linalg.matrix_power(elem1, elem2)

  elif opr in ('<', 'lt'):
    return elem1 < elem2
  elif opr in ('>', 'gt'):
    return elem1 > elem2
  elif opr in ('<=', 'le'):
    return elem1 <= elem2
  elif opr in ('>=', 'ge'):
    return elem1 >= elem2
  elif opr in ('~=', 'ne'):
    return elem1 != elem2
  elif opr in ('==', 'eq'):
    return elem2 == elem1

  # logical

  elif opr in ('&', '&&', 'and'):
    return torch.logical_and(elem1, elem2)
  elif opr in ('|', '||', 'or'):
    return torch.logical_or(elem1, elem2)

  assert False, 'TODO'
  
BinaryOprRecorder = FunctionTracer(eval_binary_opr, torch=torch)

# horizontal
def concat_cells_row(items: List[List[List[Any]]]) -> List[List[Any]]:
  if not items:
    return [[]]
  if len(items) == 1:
    return items[0]
  row_number = len(items[0])
  new_cell= [
    functools.reduce(lambda x, y: x + y, (item[i] for item in items), [])
    for i in row_number # i: row number
  ]
  return new_cell

# vertical
def concat_cells_col(items: List[List[List[Any]]]) -> List[List[Any]]:
  if not items:
    return [[]]
  if len(items) == 1:
    return items[0]
  return functools.reduce(lambda x, y: x + y, items, [])

def parse_subsref_arr_slice_oo(sub):
  # TODO end operator

  if sub == ':':
    return slice(None)
  if isinstance(sub, np.ndarray):
    sub_int = sub.astype(int) - 1
    if len(sub.shape) == 0:
      # retain dimension
      sub_int = slice(sub_int, sub_int + 1, 1)
    elif len(sub.shape) > 1:
      # sub_int(:)
      sub_int = np.swapaxes(sub_int, -1, -2)
      sub_int = sub_int.flatten()
    return sub_int

  assert False, 'TODO'


def parse_subsref_arr_slice_sa(sub):

  if isinstance(sub, slice):
    # A = a:b:c
    # TODO what about end operator?
    # TODO fancier parser
    return slice(sub.start - 1, sub.stop - 1, sub.step)
  if isinstance(sub, int):
    return slice(sub - 1, sub, 1)

  if isinstance(sub, torch.Tensor):
    # variable defined
    sub_int = sub.int() - 1
    if len(sub.shape) == 0:
      sub_int = sub_int.view((1))
    elif len(sub.shape) > 1:
      sub_int = sub_int.transpose(-1, -2).contiguous().view((-1))
    return sub_int

  assert False, 'TODO'


def commit_subsref_or_subsasgn_arr(
    node: torch.Tensor,
    subs_parsed: List[slice | np.ndarray | torch.Tensor],
    assign_target: torch.Tensor | None = None
):
  if not subs_parsed:
    assert assign_target is None, 'WTF'
    return node

  if len(subs_parsed) == 1:
    sub = subs_parsed[0]
    if isinstance(sub, slice):
      sub = torch.arange(sub.start, sub.stop, sub.step, dtype=int)
    if isinstance(sub, np.ndarray):
      sub = torch.tensor(sub, dtype=int)
    # parse row and col
    row_cnt = node.size()[-2]
    sub_col = sub // row_cnt
    sub_row = sub - sub_col

  elif len(subs_parsed) == 2:
    sub0, sub1 = subs_parsed
    xx, yy = torch.meshgrid(sub0, sub1, indexing='ij')
    ret = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    sub_col = ret[:, 0]
    sub_row = ret[:, 1]

  else:
    assert False, 'TODO'

  if assign_target is not None:
    node[..., sub_row, sub_col] = assign_target.transpose(-2, -1).flatten()
    return node
  else:
    return node[..., sub_row, sub_col] \
        .reshape((sub_row.size(0), sub_col.size(0))).transpose(-2, -1)


def eval_subsref_arr(
    node: torch.Tensor,
    subs: List[torch.Tensor | slice],
    assign_target: torch.Tensor | None = None
):
  subs_parsed = [
      parse_subsref_arr_slice_oo(sub)
      if sub == ':' or isinstance(sub, np.ndarray)
      else parse_subsref_arr_slice_sa(sub)
      for sub in subs
  ]
  
  return commit_subsref_or_subsasgn_arr(node, subs_parsed, assign_target)

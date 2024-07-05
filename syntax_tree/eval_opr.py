import torch


def eval_unary_opr(opr: str, elem: torch.Tensor) -> torch.Tensor:

  if opr in ('ctranspose', '\''):
    return torch.conj(elem).transpose(elem, -2, -1)
  elif opr in ('transpose', ".'"):
    return torch.transpose(elem, -2, -1)
  elif opr in ('uplus', '+'):
    return elem
  elif opr in ('uminus', '-'):
    return -elem
  elif opr in ('~', '!', 'not'):
    return torch.logical_not(elem)
  raise 'TODO'
  
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
    
  raise 'TODO'
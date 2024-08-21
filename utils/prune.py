from typing import Dict, Any, Tuple, Callable, TypeVar, ParamSpec, Generic

import ast
import inspect
import copy

from utils.ast_template import CodeTemplate
from utils.gen_uuid import gen_uuid_b64


class BranchEvent:
  IF = 0
  ELIF = 1
  ELSE = 2

  MATCH_CASE = 10
  MATCH_DEFAULT = 11

  FOR = 20

  WHILE = 30


tpl_create_tracer = CodeTemplate('tracer = {}\nret = None', mode='exec')
tpl_add_tracer = CodeTemplate('tracer[lineno] = (event, data)', mode='exec')
tpl_ret_assign = CodeTemplate('ret = return_body', mode='exec')
tpl_real_return = CodeTemplate('return ret, tracer', mode='exec')
none_node = ast.parse('None').body[0].value


def strip_leading_spaces(source: str):
  space_count = len(source) - len(source.lstrip(' \t'))
  if space_count > 0:
    source_splitted = [(
        x[space_count:] if len(x) > space_count else ''
    ) for x in source.split('\n')]
    source = '\n'.join(source_splitted)
  return source


class BranchTracedCompiler(ast.NodeTransformer):
  def __init__(self):
    self.uuid = gen_uuid_b64()[:16:2]
    self.tracer_name = '_tracer_' + self.uuid
    self.ret_name = '_ret_' + self.uuid
    self.def_dict = dict(tracer=self.tracer_name, ret=self.ret_name)

  def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
    new_node: ast.FunctionDef = self.generic_visit(copy.deepcopy(node))
    new_node.body = [
        *tpl_create_tracer.create(**self.def_dict).body,
        *new_node.body,
        *tpl_real_return.create(**self.def_dict).body,
    ]
    return new_node

  def visit_If(self, node):
    lineno = node.lineno
    line_dict = dict(lineno=lineno, data=none_node)
    new_node: ast.If = self.generic_visit(copy.copy(node))
    new_node.body = [
        *tpl_add_tracer.create(**self.def_dict, **line_dict,
                               event=BranchEvent.IF).body,
        *new_node.body,
    ]
    if hasattr(new_node, 'orelse') and not isinstance(new_node.orelse[0], ast.If):
      # this means it is an `else:` branch
      new_node.orelse = [
          *tpl_add_tracer.create(**self.def_dict, **line_dict,
                                 event=BranchEvent.ELSE).body,
          *new_node.orelse,
      ]

    return new_node

  def visit_Return(self, node: ast.Return) -> ast.Return:
    ret_assign = tpl_ret_assign.create(
        **self.def_dict, return_body=copy.copy(node.value)).body
    return ret_assign


class BranchRemover(ast.NodeTransformer):
  def __init__(self, executed_lines: Dict[int, Tuple[int, Any]]):
    self.uuid = gen_uuid_b64()
    self.executed_lines = executed_lines

  def visit_If(self, node):
    def _v(v): return [self.generic_visit(x) for x in v]
    if node.lineno in self.executed_lines:
      (event, _) = self.executed_lines[node.lineno]
      if event == BranchEvent.ELSE:
        return _v(node.orelse)
      else:
        return _v(node.body)
    elif hasattr(node, 'orelse') and node.orelse:
      return _v(node.orelse)
    return None


P = ParamSpec('P')
R = TypeVar('R')


class FunctionTracer(Generic[P, R]):

  def __init__(self, func: Callable[P, R]) -> None:
    self.func = func
    self.source = strip_leading_spaces(inspect.getsource(func))

    # get the function ast
    tree = ast.parse(self.source)
    self.func_def: ast.FunctionDef = tree.body[0]

    # traced function
    cmp = BranchTracedCompiler()
    traced_func = cmp.visit(self.func_def)
    self.traced_func_str = ast.unparse(traced_func)

    local_ns = {}
    exec(self.traced_func_str, globals(), local_ns)
    self.traced_func = local_ns[self.func_def.name]

  def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
    return self.func(*args, **kwargs)

  def prune(self, *args: P.args, **kwargs: P.kwargs) -> str:
    _, trace = self.traced_func(*args, **kwargs)
    repl = BranchRemover(trace)
    new_tree = repl.visit(self.func_def)
    new_source = ast.unparse(new_tree)

    return new_source

  def prune_f(self, *args: P.args, **kwargs: P.kwargs) -> Callable[P, R]:
    src = self.prune(*args, **kwargs)

    local_ns = {}
    exec(src, globals(), local_ns)
    f = local_ns[self.func_def.name]

    return f


if __name__ == "__main__":

  def example_function(x: int):
    if x > 0:
      return "Positive"
    elif x < 0:
      return "Negative"
    else:
      return "Zero"

  t = FunctionTracer(example_function)

  optimized_source = t.prune(3)
  f = t.prune_f(3)

  print(optimized_source)
  print(f(-3))

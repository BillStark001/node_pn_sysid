from typing import List, Dict, Tuple, Optional

import ast
import copy
import uuid
import base64


def gen_uuid_b64():
  uuid_obj = uuid.uuid4()
  uuid_bytes = uuid_obj.bytes
  b64_encoded = base64.urlsafe_b64encode(
      uuid_bytes).rstrip(b'=').decode('ascii')
  return b64_encoded


class VariableVisitor(ast.NodeVisitor):

  def __init__(self):
    self.var_dict: Dict[str, List[ast.Name]] = {}

  def visit_Name(self, node: ast.Name):
    name = node.id
    if not name in self.var_dict:
      self.var_dict[name] = []
    self.var_dict[name].append(node)


def create_var_dict(node: ast.AST):
  v = VariableVisitor()
  v.visit(node)
  return v.var_dict


class CodeTemplate:

  def __init__(
      self,
      template: str,
      mode: str = 'eval',
      **default_replace: Optional[str]
  ):
    self.tree = ast.parse(template, mode=mode)
    self.var_dict = create_var_dict(self.tree)
    self.default_replace = default_replace

  def create(self, **replace_args: Optional[str]):

    # check optional arguments
    for src, dst in self.default_replace.items():
      if not src in replace_args:
        replace_args[src] = dst
    for src, dst in replace_args.items():
      if dst is None:
        replace_args[src] = 'var_' + gen_uuid_b64()

    # replace in-place

    replaced: List[Tuple[ast.Name, str]] = []

    for src, dst in replace_args.items():
      if not src in self.var_dict:
        continue
      for node in self.var_dict[src]:
        node.id = dst
        replaced.append((node, src))

    # copy tree
    tree_copy = copy.deepcopy(self.tree)

    for node, src in replaced:
      node.id = src

    return tree_copy

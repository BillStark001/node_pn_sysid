from miss_hit_core.m_ast import *
from dataclasses import dataclass, field
from typing import List, Optional, Dict, cast


@dataclass
class CFGNode:
  uid: int
  label: str
  stmt_list: Optional[Sequence_Of_Statements] = None


@dataclass
class CFGEdge:
  from_node: int
  to_node: int
  label: str
  cond: Optional[Expression] = None


@dataclass
class CFG:
  nodes: Dict[int, CFGNode] = field(default_factory=dict)
  edges: List[CFGEdge] = field(default_factory=list)
  current_id: int = 0

  def add_node(self, label: str, stmt_list: Optional[Sequence_Of_Statements] = None) -> int:
    node_id = self.current_id
    self.nodes[node_id] = CFGNode(node_id, label, stmt_list)
    self.current_id += 1
    return node_id

  def add_edge(self, from_node: int, to_node: int, label: str, cond: Optional[Expression] = None):
    self.edges.append(CFGEdge(from_node, to_node, label, cond))


def generate_cfg(statements: Sequence_Of_Statements) -> CFG:
  cfg = CFG()

  # Add entry and exit nodes
  entry_id = cfg.add_node("Entry")
  exit_id = cfg.add_node("Exit")

  # Process statements
  last_id = process_statements(cfg, statements, entry_id, exit_id)

  # Connect last statement to exit node
  cfg.add_edge(last_id, exit_id, "")

  return cfg


def process_statements(cfg: CFG, statements: Sequence_Of_Statements, start_id: int, exit_id: int) -> int:
  current_id = start_id

  for stmt in statements.l_statements:
    if isinstance(stmt, Simple_Statement):
      current_id = process_simple_statement(cfg, stmt, current_id)
    elif isinstance(stmt, Compound_Statement):
      current_id = process_compound_statement(cfg, stmt, current_id, exit_id)

  return current_id


def process_simple_statement(cfg: CFG, stmt: Simple_Statement, current_id: int) -> int:
  if isinstance(stmt, (Simple_Assignment_Statement, Compound_Assignment_Statement, Naked_Expression_Statement)):
    return cfg.add_node(stmt.__class__.__name__, stmt)
  elif isinstance(stmt, Return_Statement):
    node_id = cfg.add_node("Return", stmt)
    cfg.add_edge(node_id, 1, "Return")
    return node_id
  elif isinstance(stmt, Break_Statement):
    node_id = cfg.add_node("Break", stmt)
    # The edge to the loop exit will be added in the loop processing function
    return node_id
  elif isinstance(stmt, Continue_Statement):
    node_id = cfg.add_node("Continue", stmt)
    # The edge to the loop start will be added in the loop processing function
    return node_id
  elif isinstance(stmt, (Global_Statement, Persistent_Statement, Import_Statement)):
    return cfg.add_node(stmt.__class__.__name__, stmt)
  else:
    raise ValueError(f"Unknown Simple_Statement type: {type(stmt)}")


def process_compound_statement(cfg: CFG, stmt: Compound_Statement, current_id: int, exit_id: int) -> int:
  if isinstance(stmt, For_Loop_Statement):
    return process_for_loop(cfg, stmt, current_id, exit_id)
  elif isinstance(stmt, While_Statement):
    return process_while_loop(cfg, stmt, current_id, exit_id)
  elif isinstance(stmt, If_Statement):
    return process_if_statement(cfg, stmt, current_id, exit_id)
  elif isinstance(stmt, Switch_Statement):
    return process_switch_statement(cfg, stmt, current_id, exit_id)
  elif isinstance(stmt, Try_Statement):
    return process_try_statement(cfg, stmt, current_id, exit_id)
  elif isinstance(stmt, SPMD_Statement):
    return process_spmd_statement(cfg, stmt, current_id)
  else:
    raise ValueError(f"Unknown Compound_Statement type: {type(stmt)}")


def process_for_loop(cfg: CFG, stmt: For_Loop_Statement, current_id: int, exit_id: int) -> int:
  loop_start = cfg.add_node("For Loop Start", stmt)
  cfg.add_edge(current_id, loop_start, "")

  loop_body_end = process_statements(cfg, stmt.n_body, loop_start, exit_id)

  cfg.add_edge(loop_body_end, loop_start, "Next Iteration")
  loop_exit = cfg.add_node("For Loop Exit")
  cfg.add_edge(loop_start, loop_exit, "Loop Complete")

  # Handle break statements
  for node_id, node in cfg.nodes.items():
    if node.label == "Break" and node.stmt_list and node.stmt_list in stmt.n_body.l_statements:
      cfg.add_edge(node_id, loop_exit, "Break")

  return loop_exit


def process_while_loop(cfg: CFG, stmt: While_Statement, current_id: int, exit_id: int) -> int:
  loop_start = cfg.add_node("While Loop", stmt)
  cfg.add_edge(current_id, loop_start, "")

  loop_body_end = process_statements(cfg, stmt.n_body, loop_start, exit_id)

  cfg.add_edge(loop_body_end, loop_start, "Next Iteration")
  loop_exit = cfg.add_node("While Loop Exit")
  cfg.add_edge(loop_start, loop_exit, "Condition False", stmt.n_guard)

  # Handle break statements
  for node_id, node in cfg.nodes.items():
    if node.label == "Break" and node.stmt_list and node.stmt_list in stmt.n_body.l_statements:
      cfg.add_edge(node_id, loop_exit, "Break")

  return loop_exit


def process_if_statement(cfg: CFG, stmt: If_Statement, current_id: int, exit_id: int) -> int:
  if_node = cfg.add_node("If", stmt)
  cfg.add_edge(current_id, if_node, "")

  end_if = cfg.add_node("End If")

  for action in cast(List[Action], stmt.l_actions):
    action_start = cfg.add_node(f"{action.kind()} Action", action)
    cfg.add_edge(if_node, action_start, action.kind(), action.n_expr)

    action_end = process_statements(cfg, action.n_body, action_start, exit_id)
    cfg.add_edge(action_end, end_if, "")

  return end_if


def process_switch_statement(cfg: CFG, stmt: Switch_Statement, current_id: int, exit_id: int) -> int:
  switch_node = cfg.add_node("Switch", stmt)
  cfg.add_edge(current_id, switch_node, "")

  end_switch = cfg.add_node("End Switch")

  for action in cast(List[Action], stmt.l_actions):
    action_start = cfg.add_node(f"{action.kind()} Action", action)
    cfg.add_edge(switch_node, action_start, action.kind(), action.n_expr)

    action_end = process_statements(cfg, action.n_body, action_start, exit_id)
    cfg.add_edge(action_end, end_switch, "")

  return end_switch


def process_try_statement(cfg: CFG, stmt: Try_Statement, current_id: int, exit_id: int) -> int:
  try_node = cfg.add_node("Try", stmt)
  cfg.add_edge(current_id, try_node, "")

  try_body_end = process_statements(cfg, stmt.n_body, try_node, exit_id)

  catch_node = cfg.add_node("Catch", stmt)
  cfg.add_edge(try_node, catch_node, "Exception")

  catch_body_end = process_statements(cfg, stmt.n_handler, catch_node, exit_id)

  end_try = cfg.add_node("End Try")
  cfg.add_edge(try_body_end, end_try, "No Exception")
  cfg.add_edge(catch_body_end, end_try, "")

  return end_try


def process_spmd_statement(cfg: CFG, stmt: SPMD_Statement, current_id: int) -> int:
  spmd_node = cfg.add_node("SPMD", stmt)
  cfg.add_edge(current_id, spmd_node, "")

  spmd_body_end = process_statements(cfg, stmt.n_body, spmd_node, None)

  end_spmd = cfg.add_node("End SPMD")
  cfg.add_edge(spmd_body_end, end_spmd, "")

  return end_spmd


def generate_cfg(statements: Sequence_Of_Statements) -> CFG:
  cfg = CFG()

  entry_id = cfg.add_node("Entry")
  exit_id = cfg.add_node("Exit")

  last_id = process_statements(cfg, statements, entry_id, exit_id)

  cfg.add_edge(last_id, exit_id, "")

  return cfg

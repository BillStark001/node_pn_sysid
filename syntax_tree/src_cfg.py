from miss_hit_core.m_ast import *
from dataclasses import dataclass, field
from typing import List, Optional, Dict, cast, Any, Callable


class CFGType:
  
  SEQUENCE = 0
  
  GLOBAL_ENTRY = 1
  GLOBAL_EXIT = 2
  
  IF_ENTRY = 3
  IF_EXIT = 4
  IF_ACTION_ENTRY = 5
  
  SWITCH_ENTRY = 6
  SWITCH_EXIT = 7
  SWITCH_ACTION_ENTRY = 8
  
  FOR_ENTRY = 11
  FOR_EXIT = 12
  FOR_INIT = 13
  FOR_CONTINUE = 14
  
  WHILE_ENTRY = 15
  WHILE_EXIT = 16
  
  CONTINUE = 21
  BREAK = 22
  RETURN = 23
  
  SPMD_ENTRY = 28
  SPMD_EXIT = 29
  
  TRY_ENTRY = 41
  TRY_EXIT = 42
  TRY_CATCH = 43
  
  
@dataclass
class CFGNode:
  uid: int
  type: int = CFGType.SEQUENCE
  label: str = ''
  stmt_list: Optional[Sequence_Of_Statements] = None


@dataclass
class CFGEdge:
  from_node: int
  to_node: int
  label: str
  cond: Optional[Expression] = None
  precedence: Optional[int] = None


@dataclass
class CFG:
  nodes: Dict[int, CFGNode] = field(default_factory=dict)
  edges: List[CFGEdge] = field(default_factory=list)
  current_id: int = 0

  def add_node(
    self, 
    type: int,
    label: str = '', 
    stmt_list: Optional[Sequence_Of_Statements] = None
  ) -> int:
    node_id = self.current_id
    self.nodes[node_id] = CFGNode(node_id, type, label, stmt_list)
    self.current_id += 1
    return node_id

  def add_edge(
    self, 
    from_node: int, to_node: int, label: str, 
    cond: Optional[Expression] = None,
    precedence: Optional[int] = None,
  ):
    self.edges.append(CFGEdge(from_node, to_node, label, cond, precedence))


def generate_cfg(statements: Sequence_Of_Statements) -> CFG:
  cfg = CFG()

  # Add entry and exit nodes
  entry_id = cfg.add_node(CFGType.GLOBAL_ENTRY, "Entry")
  exit_id = cfg.add_node(CFGType.GLOBAL_EXIT, "Exit")

  # Process statements
  last_id = process_statements(cfg, statements, entry_id, exit_id)

  # Connect last statement to exit node
  cfg.add_edge(last_id, exit_id, "End")

  return cfg

def is_sequential_statement(stmt):
  return isinstance(stmt, (
    Simple_Assignment_Statement, 
    Compound_Assignment_Statement, 
    Naked_Expression_Statement,
    # TODO should these be considered sequential?
    Global_Statement, 
    Persistent_Statement, 
    Import_Statement
  ))
  
def find_first_matching_index(
  lst: List[Any],
  condition: Callable[[Any], bool], 
  start = 0,
  fallback = -1
):
  return next((i for i in range(start, len(lst)) if condition(lst[i])), fallback)

def process_statements(cfg: CFG, statements: Sequence_Of_Statements, start_id: int, exit_id: int) -> int:
  current_id = start_id

  i = 0
  l = len(statements.l_statements)
  while i < l:
    stmt = statements.l_statements[i] # current statement
    prev_id = current_id
    
    # pack successive simple statements
    if is_sequential_statement(stmt):
      sequential_statement_end_index = find_first_matching_index(
        statements.l_statements,
        lambda x: not is_sequential_statement(x),
        i + 1,
        l
      )
      seq_stmts = statements.l_statements[i: sequential_statement_end_index]
      seq_obj = Sequence_Of_Statements(seq_stmts)
      
      current_id = cfg.add_node(
        CFGType.SEQUENCE,
        seq_obj.__class__.__name__, seq_obj
      )
      i = sequential_statement_end_index
      
    # simple statements with control flow
    elif isinstance(stmt, Simple_Statement):
      current_id = process_simple_control_statement(cfg, stmt, current_id)
      i += 1
    elif isinstance(stmt, Compound_Statement):
      current_id = process_compound_statement(cfg, stmt, current_id, exit_id)
      i += 1
    
    cfg.add_edge(prev_id, current_id, 'Flow')  
    

  return current_id


def process_simple_control_statement(cfg: CFG, stmt: Simple_Statement, current_id: int) -> int:
  
  if isinstance(stmt, (Return_Statement, Break_Statement, Continue_Statement)):
    node_name = stmt.__class__.__name__[:-10]
    t = CFGType.RETURN
    if isinstance(stmt, Break_Statement):
      t = CFGType.BREAK
    elif isinstance(stmt, Continue_Statement):
      t = CFGType.CONTINUE
      
    node_id = cfg.add_node(t, node_name, stmt)
    cfg.add_edge(current_id, node_id, '->' + node_name)
    if isinstance(stmt, Return_Statement):
      # otherwise, the edge to the loop exit / start
      # will be added in the loop processing function
      cfg.add_edge(node_id, 1, "Return")
    return node_id

  else:
    raise ValueError(f"Unknown simple control statement type: {type(stmt)}")


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
  loop_init = cfg.add_node(
    CFGType.FOR_INIT, 
    f"For Loop Init #{stmt.uid}", stmt
  )
  loop_start = cfg.add_node(
    CFGType.FOR_ENTRY,
  )
  cfg.add_edge(current_id, loop_init, "For Loop Init")
  cfg.add_edge(loop_init, loop_start, "For Loop Start")

  # body of the for loop
  loop_body_end = process_statements(cfg, stmt.n_body, loop_start, exit_id)
  
  loop_continue = cfg.add_node(CFGType.FOR_CONTINUE)
  cfg.add_edge(loop_body_end, loop_continue, "Next Iteration", None, 0)
  cfg.add_edge(loop_continue, loop_start, "Next Iteration", None, 0)
  
  loop_exit = cfg.add_node(CFGType.FOR_EXIT, "For Loop Exit")
  cfg.add_edge(loop_start, loop_exit, "Loop Complete", None, 1)

  # Handle break statements
  for node_id, node in cfg.nodes.items():
    if node.label == "Break" and node.stmt_list and node.stmt_list in stmt.n_body.l_statements:
      cfg.add_edge(node_id, loop_exit, "Break")

  return loop_exit


def process_while_loop(cfg: CFG, stmt: While_Statement, current_id: int, exit_id: int) -> int:
  loop_start = cfg.add_node(CFGType.WHILE_ENTRY, f"While Loop #{stmt.uid}", stmt)
  cfg.add_edge(current_id, loop_start, "")

  loop_body_end = process_statements(cfg, stmt.n_body, loop_start, exit_id)

  cfg.add_edge(loop_body_end, loop_start, "Next Iteration", stmt.n_guard, 0)
  loop_exit = cfg.add_node(CFGType.WHILE_EXIT, "While Loop Exit")
  cfg.add_edge(loop_start, loop_exit, "Condition False", None, 1)

  # Handle break statements
  for node_id, node in cfg.nodes.items():
    if node.label == "Break" and node.stmt_list and node.stmt_list in stmt.n_body.l_statements:
      cfg.add_edge(node_id, loop_exit, "Break")

  return loop_exit


def process_if_statement(cfg: CFG, stmt: If_Statement, current_id: int, exit_id: int) -> int:
  if_node = cfg.add_node(CFGType.IF_ENTRY, f"If #{stmt.uid}", stmt)
  # add an edge from the current one to `if` entry
  cfg.add_edge(current_id, if_node, "")

  end_if = cfg.add_node(CFGType.IF_EXIT, "End If")

  for i, action in enumerate(cast(List[Action], stmt.l_actions)):
    action_start = cfg.add_node(
      CFGType.IF_ACTION_ENTRY, 
      f"{action.kind()} Action #{i}", action
    )
    cfg.add_edge(
      if_node, action_start, action.kind(), 
      action.n_expr, i
    )

    action_end = process_statements(cfg, action.n_body, action_start, exit_id)
    cfg.add_edge(action_end, end_if, "")

  return end_if


def process_switch_statement(cfg: CFG, stmt: Switch_Statement, current_id: int, exit_id: int) -> int:
  switch_node = cfg.add_node(CFGType.SWITCH_ENTRY, f"Switch #{stmt.uid}", stmt)
  cfg.add_edge(current_id, switch_node, "")

  end_switch = cfg.add_node(CFGType.SWITCH_EXIT, "End Switch")

  for i, action in enumerate(cast(List[Action], stmt.l_actions)):
    action_start = cfg.add_node(CFGType.SWITCH_ACTION_ENTRY, f"{action.kind()} Action", action)
    cfg.add_edge(
      switch_node, action_start, action.kind(), action.n_expr
    )

    action_end = process_statements(cfg, action.n_body, action_start, exit_id)
    cfg.add_edge(action_end, end_switch, "")

  return end_switch


def process_try_statement(cfg: CFG, stmt: Try_Statement, current_id: int, exit_id: int) -> int:
  try_node = cfg.add_node(CFGType.TRY_ENTRY, "Try", stmt)
  cfg.add_edge(current_id, try_node, "")

  try_body_end = process_statements(cfg, stmt.n_body, try_node, exit_id)

  catch_node = cfg.add_node(CFGType.TRY_CATCH, "Catch", stmt)
  cfg.add_edge(try_node, catch_node, "Exception")

  catch_body_end = process_statements(cfg, stmt.n_handler, catch_node, exit_id)

  end_try = cfg.add_node(CFGType.TRY_EXIT, "End Try")
  cfg.add_edge(try_body_end, end_try, "No Exception")
  cfg.add_edge(catch_body_end, end_try, "")

  return end_try


def process_spmd_statement(cfg: CFG, stmt: SPMD_Statement, current_id: int) -> int:
  spmd_node = cfg.add_node(CFGType.SPMD_ENTRY, "SPMD", stmt)
  cfg.add_edge(current_id, spmd_node, "")

  spmd_body_end = process_statements(cfg, stmt.n_body, spmd_node, None)

  end_spmd = cfg.add_node(CFGType.SPMD_EXIT, "End SPMD")
  cfg.add_edge(spmd_body_end, end_spmd, "")

  return end_spmd



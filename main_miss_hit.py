import miss_hit_core

from miss_hit_core.m_lexer import MATLAB_Lexer, MATLAB_Latest_Language
from miss_hit_core.config import Config
from miss_hit_core.errors import Message_Handler, Message
from miss_hit_core.m_parser import MATLAB_Parser


path = './network_2bus2gen_ode.m'

with open(path, "r", encoding="utf-8") as f:
  content = f.read()
 
all_node_types = [
  "Node",
  "Expression",
  "Name",
  "Literal",
  "Definition",
  "Pragma",
  "Statement",
  "Simple_Statement",
  "Compound_Statement",
  "Compilation_Unit",
  "Script_File",
  "Function_File",
  "Class_File",
  "Class_Definition",
  "Function_Definition",
  "Copyright_Info",
  "Docstring",
  "Function_Signature",
  "Sequence_Of_Statements",
  "Name_Value_Pair",
  "Special_Block",
  "Entity_Constraints",
  "Argument_Validation_Delegation",
  "Class_Enumeration",
  "Action",
  "Row",
  "Row_List",
  "Reference",
  "Cell_Reference",
  "Identifier",
  "Selection",
  "Dynamic_Selection",
  "Superclass_Reference",
  "For_Loop_Statement",
  "General_For_Statement",
  "Parallel_For_Statement",
  "While_Statement",
  "If_Statement",
  "Switch_Statement",
  "Try_Statement",
  "SPMD_Statement",
  "Simple_Assignment_Statement",
  "Compound_Assignment_Statement",
  "Naked_Expression_Statement",
  "Return_Statement",
  "Break_Statement",
  "Continue_Statement",
  "Global_Statement",
  "Persistent_Statement",
  "Import_Statement",
  "Tag_Pragma",
  "No_Tracing_Pragma",
  "Justification_Pragma",
  "Metric_Justification_Pragma",
  "Number_Literal",
  "Char_Array_Literal",
  "String_Literal",
  "Reshape",
  "Range_Expression",
  "Matrix_Expression",
  "Cell_Expression",
  "Function_Call",
  "Unary_Operation",
  "Binary_Operation",
  "Binary_Logical_Operation",
  "Lambda_Function",
  "Function_Pointer",
  "Metaclass",
] 

class ModifiedMessageHandler(Message_Handler):
  def __init__(self, config):
    super().__init__(config)
  def register_message(self, msg):
    assert isinstance(msg, Message)
    self.process_message(msg)
    if msg.fatal:
      raise Exception(msg.location, msg.message)
    
mh = ModifiedMessageHandler('debug')
  
lexer = MATLAB_Lexer(
  MATLAB_Latest_Language(),
  mh,
  content,
  path, 
  None
)

cfg = Config()
cfg.style_rules = {}

parser = MATLAB_Parser(
  mh,
  lexer,
  cfg,
)

cu = parser.parse_file()
print(cu)
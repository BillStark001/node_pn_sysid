import torch.nn as nn

from syntax_tree.graph_eval import eval_m

class MatlabGraphTraceModule(nn.Module):
  
  def __init__(self, uuid_list, graph):
    super().__init__()
    self.uuid_list = uuid_list
    self.graph = graph
    
  def forward(self, *args):
    replace_dict = {
      self.uuid_list[i]: args[i] for i in range(len(self.uuid_list))
    }
    return eval_m(self.graph, replace_dict)
  

class MatlabGraphModule(nn.Module):

  def __init__(
    self, 
    graph, params, param_uuids, input_uuids, 
    traced_model=None
  ):
    super().__init__()
    self.graph: dict = graph
    self.params: dict = params
    self.param_uuids: dict = param_uuids
    
    self.inputs = []
    self.uuid_tensor_map = {}
    for k, uuid in input_uuids.items():
      self.inputs.append(k)
      self.uuid_tensor_map[uuid] = None
    for k, uuid in param_uuids.items():
      self.uuid_tensor_map[uuid] = params[k]
    
    # input_uuids and then param uuids
    self.uuid_order = list(self.uuid_tensor_map.keys())
    
    self.traced_model = traced_model
    if traced_model is None:
      self.traced_model = MatlabGraphTraceModule(self.uuid_order, graph)

  def get_traced_module_input(self, *inputs):
    w_list = [self.uuid_tensor_map.get(uuid, None) \
      for uuid in self.uuid_order]
    for i in range(len(inputs)):
      w_list[i] = inputs[i]
    return w_list

  def forward(self, *inputs):
    inputs_ = self.get_traced_module_input(*inputs)
    return self.traced_model(*inputs_)

  def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True):
    for name, param in self.params.items():
      yield name, param

  def parameters(self, recurse: bool = True):
    for name, param in self.params.items():
      yield param


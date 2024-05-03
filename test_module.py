class TestClass:
  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs
    
  def exec(self, *args, **kwargs):
    print(args)
    print(kwargs)
    
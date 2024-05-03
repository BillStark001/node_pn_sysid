pythonPath = getenv("CONDA_PYTHON_EXE");
if pythonPath == ""
  pythonPath = getenv("PYTHON_EXE");
end

pyenv("Version", pythonPath);

testClassInstance = PyClassWrapper('test_module', 'TestClass');

testClassInstance.init(M_1=100, D_1=10);

a(1, 2, test2=3);

function a(varargin)
  disp(varargin);
end
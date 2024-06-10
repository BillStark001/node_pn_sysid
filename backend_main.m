env_vars = utils.env(File='.env');

% init python

pythonPath = '/opt/anaconda3/bin/python';

pyenv(Version = pythonPath);
pyenv(ExecutionMode = "OutOfProcess");

torch = py.importlib.import_module('torch');

G = TorchBackend(torch);

r = [1, 2, 3; 4, 5, 6; 7, 8, 9];
A = G.array(r);
B = G.array(r);
C = A + B;

dddd = C.Data;
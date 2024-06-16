% compile syntax tree

G = syntax_tree.Backend();

x = G.array(randn(1, 1), 'x');
inputs = struct(x = x);
w = toy_nn_init(G);

cache = containers.Map();


y = toy_nn_forward(inputs, w, G);

w2 = strip_weight(w, cache);
i2 = strip_weight(inputs, cache);
y_py = syntax_tree.st2py(y, cache);


% init python

pythonPath = '/opt/anaconda3/bin/python';

pyenv(Version = pythonPath);
pyenv(ExecutionMode = "OutOfProcess");

py_port = py.importlib.import_module('backend');

% pass syntax tree

py_port.main_matlab(w2, i2, y_py);


% functions

function w2r = strip_weight(w, cache)
  w2 = struct();
  fields = fieldnames(w);
  for i = 1:numel(fields)
      field = fields{i};
      w2.(field) = syntax_tree.st2py(w.(field), cache);
  end
  w2r = syntax_tree.st2py(w2);
end

function z = fullconnected(x, W, b)
    z = W * x + b;
end

function y = toy_nn_forward(inputs, w, G)
    % 1-4-4-2
    l1 = G.tanh(fullconnected(inputs.x, w.W1, w.b1));
    l2 = G.tanh(fullconnected(l1, w.W2, w.b2));
    y = G.tanh(fullconnected(l2, w.W3, w.b3));
end

function w = toy_nn_init(G)
    w = struct();
    w.W1 = G.array(randn(4, 1), 'W1');
    w.b1 = G.array(randn(4, 1), 'b2');
    w.W2 = G.array(randn(4, 4), 'W2');
    w.b2 = G.array(randn(4, 1), 'b2');
    w.W3 = G.array(randn(2, 4), 'W3');
    w.b3 = G.array(randn(2, 1), 'b3');
end



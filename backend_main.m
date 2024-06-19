% compile syntax tree

G = syntax_tree.Backend();

y = G.array(randn(4, 1), "y");
inputs = struct(y = y);
w = toy_nn_init(G);

cache = containers.Map();


dydt = network_2bus2gen_ode(inputs, w, G);

w2 = strip_weight(w, cache);
i2 = strip_weight(inputs, cache);
y_py = syntax_tree.st2py(dydt, cache);

xx = y(:, 1:end-1);

% init python

%{
pythonPath = "/opt/anaconda3/bin/python";

pyenv(Version = pythonPath);
pyenv(ExecutionMode = "OutOfProcess");

py_port = py.importlib.import_module("backend");

% pass syntax tree

py_port.main_matlab(w2, i2, y_py);
%}

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

function w = toy_nn_init(G)
    w = struct();
    all_params = [...
        "M1", "D1", "V1", "Pmech1", ...
        "M2", "D2", "V2", "Pmech2", ...
        "G", "B", "G11", "G22", ...
        "omega0",
        ];
    for i = 1:numel(all_params)
        p = all_params(i);
        w.(p) = G.array(randn(), p);
    end
end



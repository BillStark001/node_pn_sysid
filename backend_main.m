env_vars = utils.env(File='.env');

% init python

%{

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

w = toy_nn_init(G);
w2 = strip_weight(w);

%}

w3 = toy_nn_init_syms();
inputs = struct();
inputs.x = sym(x);
y = toy_nn_forward(inputs, w3);
g = buildGraph(y);


a = 1;
b = 2;
getname(4, 3);

% py_port = py.importlib.import_module('backend');
% py_port.main_matlab(w2, @(x, ~) toy_nn_forward(x, w2));

function w2 = strip_weight(w)
    w2 = struct();
    fields = fieldnames(w);
    for i = 1:numel(fields)
        field = fields{i};
        w2.(field) = w.(field).Data;
    end
end


function z = fullconnected(x, W, b)
    z = W * x + b;
end

% function y = tanh(x)
%     e_px = e ^ x;
%     e_mx = e ^ (-x);
%     y = (e_px - e_mx) / (e_px + e_mx);
% end

function y = toy_nn_forward(inputs, w)
    % 1-4-4-2
    l1 = tanh(fullconnected(inputs.x, w.W1, w.b1));
    l2 = tanh(fullconnected(l1, w.W2, w.b2));
    y = tanh(fullconnected(l2, w.W3, w.b3));
end

function w = toy_nn_init_syms()
    w = struct();
    keys = {'W1', 'b1', 'W2', 'b2', 'W3', 'b3'};
    for i = 1:numel(keys)
        s = sym(keys{i});
        w.(keys{i}) = s;
    end
end

function getname(a,b)
    s = inputname(1);
    disp(['First calling variable is ''' s '''.'])
end

function w = toy_nn_init(G)
    w = struct();
    w.W1 = G.array(randn(4, 1));
    w.b1 = G.array(randn(4, 1));
    w.W2 = G.array(randn(4, 4));
    w.b2 = G.array(randn(4, 1));
    w.W3 = G.array(randn(2, 4));
    w.b3 = G.array(randn(2, 1));
end

function graph = buildGraph(expr)
    disp(expr);
    op = symFunType(expr);
    args = children(expr);
    if isempty(args) || strcmp(op, 'variable')
        % 基本变量或常量节点
        graph = struct('type', 'leaf', 'value', expr);
    else
        % 操作节点
        graph = struct('type', 'operation', 'operation', op, 'arguments', []);
        for i = 1:numel(args)
            graph.children = buildGraph(args{i});
        end
    end
end
env_vars = utils.env(File='.env');

% init python

pythonPath = 'python';

if isfield(env_vars, 'CONDA_PYTHON_EXE')
    pythonPath = env_vars.CONDA_PYTHON_EXE;
elseif isfield(env_vars, 'PYTHON_PATH')
    pythonPath = env_vars.PYTHON_PATH;
elseif isfield(env_vars, 'PYTHON_EXE')
    pythonPath = env_vars.PYTHON_EXE;
end

pyenv(Version = pythonPath);
pyenv(ExecutionMode = "OutOfProcess");

solver = py.importlib.import_module('solver');

% init guilda

addpath(env_vars.GUILDA_PRJ_PATH);

% create power network

net = network();

% power flow
% simulation
% TODO


% identify

s_params = solver.ScenarioParameters( ...
    M_1 = 100, ...
    D_1 = 10, ...
    V_field_1 = 2.04379374735951, ...
    P_mech_1 = -0.513568531598284, ...
    M_2 = 12, ...
    D_2 = 10, ...
    V_field_2 = 1.98829642487374, ...
    P_mech_2 = 0.486559381709619, ...
    G_12 = -0.003399828308670, ...
    B_12 = -0.583070554936976, ...
    t = 0, ...
    x0 = 0, ...
    true_x = 0, ...
    omega = 2 * 60 * pi ...
);

sol_params = solver.SolverParameters( ...
    device = 'cuda', ...
    optim_factory = solver.default_optim_factory, ...
    batch_size = 20, ...
    batch_time = 30, ...
    ode_params = py.dict(method = 'rk4') ...
);

model = solver.EnvModelEstimator( ...
    scenario, ...
    sol_params ...
);

model.init();
params = model.get_current_params();

for itr = 0:2000

    if mod(itr, 5) == 0
        disp(params);
    end

    start = datetime('now');
    loss, grad_norm = model.iterate();
    dur = datetime('now');
    params = model.get_current_params();

    fprintf( ...
        'Iteration %d | Loss: %.6f | Grad Norm: %.6f | Time Elapsed: %.4fs\n', ...
        itr, loss, grad_norm, seconds(dur) ...
    );

    if loss < 1e-10
        disp('break')
        disp(params)
        break
    end

end

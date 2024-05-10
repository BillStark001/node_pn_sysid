env_vars = utils.env(File='.env');

% init python

pythonPath = '/opt/anaconda3/bin/python';

if isfield(env_vars, 'CONDA_PYTHON_EXE')
    pythonPath = env_vars.CONDA_PYTHON_EXE;
elseif isfield(env_vars, 'PYTHON_PATH')
    pythonPath = env_vars.PYTHON_PATH;
elseif isfield(env_vars, 'PYTHON_EXE')
    pythonPath = env_vars.PYTHON_EXE;
end

pyenv(Version = pythonPath);
pyenv(ExecutionMode = "OutOfProcess");

solver = py.importlib.import_module('solver_wrapper');

% init guilda

oldFolder = cd(env_vars.GUILDA_PRJ_PATH);
matlab.project.loadProject(env_vars.GUILDA_PRJ_PATH);
cd(oldFolder);

% create power network

net = network();
net.initialize();
x_init = net.x_equilibrium;
x_init(1) = x_init(1) + pi / 6;

% simulation
tend = 20;
omega = 2 * 60 * pi;
sim_res = net.simulate([0, tend], 'x0_sys', x_init);
sim_t = sim_res.t;

% resample
t = 0:1/30:tend;
t = t';
x = [
    interp1(sim_t, sim_res.X{1}.delta, t, 'linear'), ...
    interp1(sim_t, sim_res.X{1}.omega, t, 'linear') * omega, ...
    interp1(sim_t, sim_res.X{2}.delta, t, 'linear'), ...
    interp1(sim_t, sim_res.X{2}.omega, t, 'linear') * omega
];

% identify
factor = 1.05;
s_params = solver.ScenarioParameters( ...
    M_1 = 100, ...
    D_1 = 10, ...
    V_field_1 = 2.04379374735951, ...
    P_mech_1 = -0.513568531598284, ...
    M_2 = 12 * factor, ...
    D_2 = 10 * factor, ...
    V_field_2 = 1.98829642487374 * factor, ...
    P_mech_2 = 0.486559381709619 * factor, ...
    G_12 = -0.003399828308670 * factor, ...
    B_12 = -0.583070554936976 * factor, ...
    t = py.numpy.array(t), ...
    x0 = py.numpy.array(x_init), ...
    true_x = py.numpy.array(x), ...
    omega = 2 * 60 * pi ...
);

sol_params = solver.SolverParameters( ...
    device = 'cpu', ...
    optim_factory = solver.default_optim_factory, ...
    batch_size = py.int(20), ...
    batch_time = py.int(30), ...
    ode_params = py.dict(method = 'rk4') ...
);


model = solver.EnvModelEstimator( ...
    s_params, ...
    sol_params ...
);

model.init();
params = model.get_current_params();

for itr = 0:2000

    if mod(itr, 5) == 0
        disp(params);
    end

    start = datetime('now');
    iter_res = model.iterate();
    iter_res = cell(iter_res);
    loss = iter_res{1};
    grad_norm = iter_res{2};
    
    dur = datetime('now');
    params = model.get_current_params();

    fprintf( ...
        'Iteration %d | Loss: %.12f | Grad Norm: %.12f | Time Elapsed: %.4fs\n', ...
        itr, loss, grad_norm, second(dur) ...
    );

    if loss < 1e-10
        disp('break')
        disp(params)
        break
    end

end

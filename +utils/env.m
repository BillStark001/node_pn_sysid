function env_vars = env(varargin)
    p = inputParser();
    addParameter(p, 'SysMethod', 'system');
    addParameter(p, 'File', '');
    parse(p, varargin{:})
    r = p.Results;
    env_vars = env_sys(r.SysMethod);
    if ~isempty(r.File)
        env_vars_f = env_file(r.File);
        fn = fieldnames(env_vars_f);
        for k = 1:numel(fn);
            k_ = fn{k};
            env_vars.(k_) = env_vars_f.(k_);
        end
    end
end

function env_vars = env_sys(method)
    if nargin < 1, method = 'system'; end
    method = validatestring(method, {'java', 'system'});

    switch method
        case 'java'
            map = java.lang.System.getenv();  % returns a Java map
            keys = cell(map.keySet.toArray());
            vals = cell(map.values.toArray());
        case 'system'
            if ispc()
                %cmd = 'set "';  %HACK for hidden variables
                cmd = 'set';
            else
                cmd = 'env';
            end
            [~,out] = system(cmd);
            vars = regexp(strtrim(out), '^(.*)=(.*)$', ...
                'tokens', 'lineanchors', 'dotexceptnewline');
            vars = vertcat(vars{:});
            keys = vars(:,1);
            vals = vars(:,2);
    end

    % Windows environment variables are case-insensitive
    if ispc()
        keys = upper(keys);
    end

    env_vars = struct();
    for i = 1:numel(keys)
        k = keys(i);
        k = k{:};
        v = vals(i);
        v = v{:};
        if ~isvarname(k)
            continue
        end
        env_vars.(k) = v;
    end
end


function env_vars = env_file(path)

    fp = fopen(path);
    env_file = textscan(fp, '%s', 'delimiter', '\n');
    env_file = env_file{:};

    env_vars_ = struct();

    for i = 1:numel(env_file)
        line = env_file{i};

        if ~isempty(line) && ~startsWith(line, '#')
            [var_name, var_value] = strtok(line, '=');
            var_name = strtrim(var_name);
            var_value = strtrim(extractAfter(var_value, '='));
            env_vars_.(var_name) = var_value;
        end

    end
    env_vars = env_vars_;

end

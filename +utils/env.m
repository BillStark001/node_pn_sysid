function env_vars = env(path)

    fp = fopen(path);
    env_file = textscan(fp, '%s', 'delimiter', '\n');

    env_vars_ = struct();

    for i = 1:numel(env_file)
        line = env_file{i};

        if ~isempty(line) && ~startsWith(line, '#')
            [var_name, var_value] = strtok(line, '=');
            var_name = strtrim(var_name);
            var_value = strtrim(extractAfter(var_value, '='));
            for j = 1:numel(var_name)
                env_vars_.(var_name{i}) = var_value{i};
            end
        end

    end
    env_vars = env_vars_;

end

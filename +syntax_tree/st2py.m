function n_py = st2py(n, varargin)

    if nargin > 1
        cache = varargin{1};
    else
        cache = containers.Map();
    end

    if isa(n, "syntax_tree.ArrayWrapper")
        if isKey(cache, n.Uuid)
            n_py = cache(n.Uuid);
            return;
        end
        nodes = cellfun( ...
            @(x) syntax_tree.st2py(x, cache), ...
            n.Nodes, ...
            UniformOutput = false ...
        );
        n_st = struct();
        n_st.Uuid = utils.mat2py(n.Uuid);
        n_st.Name = utils.mat2py(n.Name);
        n_st.Type = utils.mat2py(n.Type);
        n_st.Data = utils.mat2py(n.Data);
        n_st.Nodes = nodes;

        n_py = utils.mat2py(n_st);
        cache(n.Uuid) = n_py;
    else
        n_py = utils.mat2py(n);
    end

end

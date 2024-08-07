function matVar = py2mat(pyVar)

    if isa(pyVar, 'py.numpy.ndarray')
        matVar = double(py.array.array('d', pyVar.flat));
        % matVar = reshape(matVar, pyVar.shape);
    elseif isa(pyVar, 'py.str')
        matVar = char(pyVar);
    elseif isa(pyVar, 'py.dict')
        matVar = struct();
        kvs = cell(py.list(pyVar.items()));

        for kv = kvs
            kv_cell = cell(kv{:});
            v = utils.py2mat(kv_cell{2});
            matVar.(string(kv_cell{1})) = utils.py2mat(v);
        end

    elseif isa(pyVar, 'py.list')
        matVar = cell(1, pyVar.len);

        for i = 1:pyVar.len
            matVar{i} = utils.py2mat(pyVar(i - 1));
        end

    elseif isa(pyVar, 'py.NoneType')
        matVar = false(1); % since there is no null in matlab
    else
        matVar = pyVar;
    end

end

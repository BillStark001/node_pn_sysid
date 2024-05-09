function matVar = py2mat(pyVar)

    if isa(pyVar, 'py.numpy.ndarray')
        matVar = double(py.array.array('d', pyVar.flat));
        matVar = reshape(matVar, pyVar.shape);
    elseif isa(pyVar, 'py.str')
        matVar = char(pyVar);
    elseif isa(pyVar, 'py.dict')
        matVar = struct();
        keys = pyVar.keys();

        for key = keys
            matVar.(char(key)) = py2mat(pyVar{key});
        end

    elseif isa(pyVar, 'py.list')
        matVar = cell(1, pyVar.len);

        for i = 1:pyVar.len
            matVar{i} = py2mat(pyVar(i - 1));
        end

    elseif isa(pyVar, 'py.NoneType')
        matVar = false(1); % since there is no null in matlab
    else
        matVar = pyVar;
    end

end

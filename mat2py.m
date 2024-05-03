function pyVar = mat2py(matVar)

    if isnumeric(matVar)
        pyVar = py.numpy.array(matVar);
    elseif ischar(matVar)
        pyVar = char(matVar);
    elseif isstruct(matVar)
        pyVar = py.dict();
        fields = fieldnames(matVar);

        for i = 1:numel(fields)
            key = fields{i};
            pyVar{key} = mat2py(matVar.(key));
        end

    elseif iscell(matVar)
        pyVar = py.list();

        for i = 1:numel(matVar)
            pyVar.append(mat2py(matVar{i}));
        end

    else
        error('Unsupported data type');
    end

end

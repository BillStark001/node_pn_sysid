function pyVar = mat2py(matVar)

    if isnumeric(matVar)
        s = size(matVar);
        s0 = py.int(s(1));
        s1 = py.int(s(2));
        pyVar = py.numpy.array(matVar).reshape(py.tuple({s0, s1}));
    elseif ischar(matVar) || isstring(matVar)
        pyVar = py.str(matVar);
    elseif isstruct(matVar)
        pyVar = py.dict();
        fields = fieldnames(matVar);

        for i = 1:numel(fields)
            key = fields{i};
            pyVar{key} = utils.mat2py(matVar.(key));
        end

    elseif iscell(matVar)
        pyVar = py.list();

        for i = 1:numel(matVar)
            pyVar.append(utils.mat2py(matVar{i}));
        end
    elseif isa(matVar, "py.object")
        pyVar = matVar;
    else
        error(strcat('Unsupported data type: ', class(matVar)));
    end

end

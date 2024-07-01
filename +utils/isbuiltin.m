function ret = isbuiltin(obj)

    classInfo = metaclass(obj);
    defPath = which(classInfo.Name);

    matlabRoot = matlabroot();

    ret = contains(defPath, matlabRoot) ...
        || startsWith(defPath, 'built-in') ...
        || endsWith(defPath, 'is a built-in method');
end
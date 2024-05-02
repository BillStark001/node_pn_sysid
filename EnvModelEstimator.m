classdef EnvModelEstimator < handle
    
    properties
        pyInterpreter
        pyModule
        pyClass
    end
    
    methods
        function obj = PythonInvoker(pyPath, moduleConfig)
            obj.pyInterpreter = py.sys.executable(pyPath);
            obj.init(moduleConfig);
        end
        
        function init(obj, moduleConfig)
            moduleName = moduleConfig.moduleName;
            className = moduleConfig.className;
            args = struct2cell(rmfield(moduleConfig, {'moduleName', 'className'}));
            obj.pyModule = py.importlib.import_module(moduleName);
            obj.pyClass = obj.pyModule.(className)(*args);
        end
        
        function result = invoke(obj, methodName, varargin)
            pyMethod = obj.pyClass.(methodName);
            pyArgs = cellfun(@(x) num2pyArray(x), varargin, 'UniformOutput', false);
            pyResult = pyMethod(pyArgs{:});
            result = py2matlabResult(pyResult);
        end
    end
end

function pyArray = num2pyArray(arr)
    pyArray = py.numpy.array(arr);
end

function result = py2matlabResult(pyResult)
    if isa(pyResult, 'py.dict')
        result = struct();
        pyKeys = pyResult.keys();
        for i = 1:length(pyKeys)
            key = char(pyKeys(i));
            result.(key) = py2matlabResult(pyResult.get(pyKeys(i)));
        end
    elseif isa(pyResult, 'py.numpy.ndarray')
        result = double(pyResult);
    else
        result = double(pyResult);
    end
end
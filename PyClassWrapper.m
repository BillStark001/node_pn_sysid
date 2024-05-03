classdef PyClassWrapper < handle

    properties
        moduleName
        className

        pyInterpreter
        pyModule
        pyClass
        pyClassCtor
    end

    methods

        function obj = PyClassWrapper(moduleName, className)
            obj.moduleName = moduleName;
            obj.className = className;
        end

        function init(obj, varargin)
            obj.pyModule = py.importlib.import_module(obj.moduleName);
            obj.pyClassCtor = obj.pyModule.(obj.className);
            obj.pyClass = obj.pyClassCtor(varargin{:});
        end

        function result = invoke(obj, methodName, varargin)
            pyResult = obj.pyClass.(methodName)(varargin{:});
            result = py2mat(pyResult);
        end

        function result = get(obj, fieldName)
            pyResult = obj.pyClass.(fieldName);
            result = py2mat(pyResult);
        end

    end

end

classdef MyClass
    properties
        Value
    end
    
    methods
        function obj = MyClass(val)
            obj.Value = val;
        end
        
        % 重载加法运算符
        function result = plus(obj1, obj2)
            varName1 = inputname(1);
            varName2 = inputname(2);
            
            if isempty(varName1)
                varName1 = 'unknown';
            end
            if isempty(varName2)
                varName2 = 'unknown';
            end
            
            fprintf('Variable names are: %s and %s\n', varName1, varName2);
            
            result = MyClass(obj1.Value + obj2.Value);
        end
    end
end
classdef Backend < handle

    methods

        function obj = Backend()
        end

        function result = arrayRaw(~, matlabArray)
            result = matlabArray;
        end

        function result = array(obj, matlabArray, varargin)
            resultRaw = obj.arrayRaw(matlabArray);
            suggestName = string(inputname(2));

            if nargin > 2
                v3 = varargin{1};
                suggestName = string(v3);
            end

            result = syntax_tree.ArrayWrapper(suggestName, "var", resultRaw, {});
        end

        function result = sin(obj, varargin)
            names = arrayfun('inputname', 2:nargin);
            disp(names);
            result = func_call("sin", varargin);
        end

        function result = cos(obj, varargin)
            result = func_call("cos", varargin);
        end

        function result = tanh(obj, varargin)
            % names = arrayfun(@(x) inputname(x), 2:nargin, UniformOutput=false);
            result = func_call("tanh", varargin);
        end

    end

end

function result = func_call(name, params)
    % params: cell array
    result = syntax_tree.ArrayWrapper("", "func", string(name), params);
end

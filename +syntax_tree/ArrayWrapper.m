classdef ArrayWrapper < handle

    properties (SetAccess = immutable)
        Uuid
        Data
        Type
    end

    properties (SetAccess = private)
        Name
        Nodes
    end

    methods

        function obj = ArrayWrapper(name, type, data, nodes)
            obj.Uuid = string(java.util.UUID.randomUUID());
            obj.Name = name; % str
            obj.Type = type; % str
            obj.Data = data; % str or mat(var case)
            obj.Nodes = nodes; % cell array of this class
        end

        function setName(obj, varargin)

            if obj.Name ~= ""
                return;
            end

            if nargin > 0
                suggestName = string(varargin{1});
            else
                suggestName = "";
            end

            if suggestName ~= ""
                obj.Name = string(suggestName);
            end

            if obj.Type == "var"
                n = "var";
            else
                n = [string(obj.Type)];
            end

            for i = 1:numel(obj.Nodes)
                nn = obj.Nodes{i}.Name;

                if nn == ""
                    nn = "n" + i;
                end

                n = [n, nn]; %#ok<*AGROW>
            end

            obj.Name = join(n, "_");
        end

        % arithmetic

        function result = plus(a, b)
            nameArr = [string(inputname(1)), string(inputname(2))];
            result = binaryOpr(a, b, "plus", nameArr);
        end

        function result = minus(a, b)
            nameArr = [string(inputname(1)), string(inputname(2))];
            result = binaryOpr(a, b, "minus", nameArr);
        end

        function result = uminus(a)
            nameArr = [string(inputname(1))];
            result = unaryOpr(a, b, "uminus", nameArr);
        end

        function result = uplus(a)
            nameArr = [string(inputname(1))];
            result = unaryOpr(a, b, "uplus", nameArr);
        end

        function result = times(a, b)
            nameArr = [string(inputname(1)), string(inputname(2))];
            result = binaryOpr(a, b, "times", nameArr);
        end

        function result = mtimes(a, b)
            nameArr = [string(inputname(1)), string(inputname(2))];
            result = binaryOpr(a, b, "mtimes", nameArr);
        end

        function result = rdivide(a, b)
            nameArr = [string(inputname(1)), string(inputname(2))];
            result = binaryOpr(a, b, "rdivide", nameArr);
        end

        function result = ldivide(a, b)
            nameArr = [string(inputname(1)), string(inputname(2))];
            result = binaryOpr(a, b, "ldivide", nameArr);
        end

        function result = mrdivide(a, b)
            nameArr = [string(inputname(1)), string(inputname(2))];
            result = binaryOpr(a, b, "mrdivide", nameArr);
        end

        function result = mldivide(a, b)
            nameArr = [string(inputname(1)), string(inputname(2))];
            result = binaryOpr(a, b, "mldivide", nameArr);
        end

        function result = power(a, b)
            nameArr = [string(inputname(1)), string(inputname(2))];
            result = binaryOpr(a, b, "power", nameArr);
        end

        function result = mpower(a, b)
            nameArr = [string(inputname(1)), string(inputname(2))];
            result = binaryOpr(a, b, "mpower", nameArr);
        end

        % comparison

        function result = lt(a, b)
            nameArr = [string(inputname(1)), string(inputname(2))];
            result = binaryOpr(a, b, "lt", nameArr);
        end

        function result = gt(a, b)
            nameArr = [string(inputname(1)), string(inputname(2))];
            result = binaryOpr(a, b, "gt", nameArr);
        end

        function result = le(a, b)
            nameArr = [string(inputname(1)), string(inputname(2))];
            result = binaryOpr(a, b, "le", nameArr);
        end

        function result = ge(a, b)
            nameArr = [string(inputname(1)), string(inputname(2))];
            result = binaryOpr(a, b, "ge", nameArr);
        end

        function result = ne(a, b)
            nameArr = [string(inputname(1)), string(inputname(2))];
            result = binaryOpr(a, b, "ne", nameArr);
        end

        function result = eq(a, b)
            nameArr = [string(inputname(1)), string(inputname(2))];
            result = binaryOpr(a, b, "eq", nameArr);
        end

        function result = and(a, b)
            nameArr = [string(inputname(1)), string(inputname(2))];
            result = binaryOpr(a, b, "and", nameArr);
        end

        function result = or(a, b)
            nameArr = [string(inputname(1)), string(inputname(2))];
            result = binaryOpr(a, b, "or", nameArr);
        end

        function result = not(a)
            nameArr = [string(inputname(1))];
            result = unaryOpr(a, b, "not", nameArr);
        end

        % misc

        function result = ctranspose(a)
            nameArr = [string(inputname(1))];
            result = unaryOpr(a, b, "ctranspose", nameArr);
        end

        function result = transpose(a)
            nameArr = [string(inputname(1))];
            result = unaryOpr(a, b, "transpose", nameArr);
        end

        function result = colon(a, d, b)
            nameArr = [string(inputname(1)), string(inputname(2)), string(inputname(3))];
            result = ternaryOpr(a, d, b, "colon", nameArr);
        end

        function result = horzcat(varargin)
            elems = {};

            for i = 1:nargin
                elem = varargin{i};
                if isnumeric(elem) elem = wrap(elem, inputname(i)); end
                elem.setName(inputname(i));
                elems = [elems(:)' {elem}];
            end

            result = syntax_tree.ArrayWrapper("", "opr", "horzcat", elems);
        end

        function result = vertcat(varargin)
            elems = {};

            for i = 1:nargin
                elem = varargin{i};
                if isnumeric(elem) elem = wrap(elem, inputname(i)); end
                elem.setName(inputname(i));
                elems = [elems(:)' {elem}];
            end

            result = syntax_tree.ArrayWrapper("", "opr", "vertcat", elems);
        end

        function result = subsref(a, s)
            s1 = s(1);
            nodesArray = {a, s1.subs};
            switch s1.type
                case '()'
                    s_opr = "subsref_arr";
                case '{}'
                    s_opr = "subsref_list";
                case '.'
                    if strcmp(s1.subs, 'Uuid') ...
                        || strcmp(s1.subs, 'Name') ... 
                        || strcmp(s1.subs, 'Data') ...
                        || strcmp(s1.subs, 'Type') ...
                        || strcmp(s1.subs, 'Nodes') ...
                        result = a.(s1.subs);
                        return;
                    end

                    s_opr = "subsref_obj";
                otherwise
                    error('Not a valid indexing expression.');
            end
            result = syntax_tree.ArrayWrapper("", "opr", s_opr, nodesArray);

            if length(s) > 1
                result = subsref(result, s(2:end));
            end

        end

        function a = subsasgn(a, s, b)

            a_res = a;
            if length(s) > 1
                a_res = syntax_tree.ArrayWrapper.subsref(a, s(1:end-1));
            end

            s1 = s(end);
            nodesArray = {a_res, s1.subs, b};
            switch s1.type
                case '()'
                    s_opr = "subsasgn_arr";
                case '{}'
                    s_opr = "subsasgn_list";
                case '.'
                    s_opr = "subsasgn_obj";
                otherwise
                    error('Not a valid indexing expression.');
            end
            result = syntax_tree.ArrayWrapper("", "opr", s_opr, nodesArray);

        end

        function index = subsindex(a)
            error("Unsupported");
        end

    end

end

function b = wrap(a, suggestName)

    if suggestName == ""
        suggestName = inputname(1);
    end

    b = syntax_tree.ArrayWrapper(string(suggestName), "var", a, {});
end

function result = unaryOpr(a, opr, inputName)
    setName(a, inputName(1));
    result = syntax_tree.ArrayWrapper("", "opr", string(opr), {a});
end

function result = binaryOpr(a, b, opr, inputName)
    if isnumeric(a) a = wrap(a, inputName(1)); end %#ok<*SEPEX>
    if isnumeric(b) b = wrap(b, inputName(2)); end
    setName(a, inputName(1));
    setName(b, inputName(2));
    result = syntax_tree.ArrayWrapper("", "opr", string(opr), {a, b});
end

function result = ternaryOpr(a, b, c, opr, inputName)
    if isnumeric(a) a = wrap(a, inputName(1)); end
    if isnumeric(b) b = wrap(b, inputName(2)); end
    if isnumeric(c) c = wrap(c, inputName(3)); end
    setName(a, inputName(1));
    setName(b, inputName(2));
    setName(c, inputName(3));
    result = syntax_tree.ArrayWrapper("", "opr", string(opr), {a, b, c});
end

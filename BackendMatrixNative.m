classdef BackendMatrixNative < BackendMatrix

    properties
        Data
    end

    methods

        function obj = BackendMatrixNative(data)

            if nargin > 0

                if isnumeric(data)
                    obj.Data = data;
                else
                    error('Input must be a numeric matrix');
                end

            else
                obj.Data = [];
            end

        end

        function result = plus(a, b)
            result = BackendMatrixNative(a.Data + b.Data);
        end

        function result = minus(a, b)
            result = BackendMatrixNative(a.Data - b.Data);
        end

        function result = uminus(a)
            result = BackendMatrixNative(-a.Data);
        end

        function result = uplus(a)
            result = a;
        end

        function result = times(a, b)
            result = BackendMatrixNative(a.Data .* b.Data);
        end

        function result = mtimes(a, b)
            result = BackendMatrixNative(a.Data * b.Data);
        end

        function result = rdivide(a, b)
            result = BackendMatrixNative(a.Data ./ b.Data);
        end

        function result = ldivide(a, b)
            result = BackendMatrixNative(a.Data .\ b.Data);
        end

        function result = mrdivide(a, b)
            result = BackendMatrixNative(a.Data / b.Data);
        end

        function result = mldivide(a, b)
            result = BackendMatrixNative(a.Data \ b.Data);
        end

        function result = power(a, b)
            result = BackendMatrixNative(a.Data .^ b.Data);
        end

        function result = mpower(a, b)
            result = BackendMatrixNative(a.Data ^ b.Data);
        end

        function result = lt(a, b)
            result = BackendMatrixNative(a.Data < b.Data);
        end

        function result = gt(a, b)
            result = BackendMatrixNative(a.Data > b.Data);
        end

        function result = le(a, b)
            result = BackendMatrixNative(a.Data <= b.Data);
        end

        function result = ge(a, b)
            result = BackendMatrixNative(a.Data >= b.Data);
        end

        function result = ne(a, b)
            result = BackendMatrixNative(a.Data ~= b.Data);
        end

        function result = eq(a, b)
            result = BackendMatrixNative(a.Data == b.Data);
        end

        function result = and(a, b)
            result = BackendMatrixNative(a.Data & b.Data);
        end

        function result = or(a, b)
            result = BackendMatrixNative(a.Data | b.Data);
        end

        function result = not(a)
            result = BackendMatrixNative(~a.Data);
        end

        function result = colon(a, d, b)

            if nargin == 3
                result = BackendMatrixNative(a:d:b);
            else
                result = BackendMatrixNative(1:a);
            end

        end

        function result = ctranspose(a)
            result = BackendMatrixNative(a.Data');
        end

        function result = transpose(a)
            result = BackendMatrixNative(a.Data.');
        end

        function result = horzcat(varargin)
            data = cellfun(@(x) x.Data, varargin, 'UniformOutput', false);
            concatenatedData = horzcat(data{:});
            result = BackendMatrixNative(concatenatedData);
        end

        function result = vertcat(varargin)
            data = cellfun(@(x) x.Data, varargin, 'UniformOutput', false);
            concatenatedData = vertcat(data{:});
            result = BackendMatrixNative(concatenatedData);
        end

        function result = subsref(a, s)

            switch s(1).type
                case '()'
                    result = BackendMatrixNative(a.Data(s.subs{:}));
                case '{}'
                    error('Cell contents reference from a non-cell array object.');
                case '.'
                    result = a.(s(1).subs);
                otherwise
                    error('Not a valid indexing expression.');
            end

            if length(s) > 1
                result = subsref(result, s(2:end));
            end

        end

        function a = subsasgn(a, s, b)

            switch s(1).type
                case '()'

                    if isa(b, 'BackendMatrixNative')
                        a.Data(s.subs{:}) = b.Data;
                    else
                        a.Data(s.subs{:}) = b;
                    end

                case '{}'
                    error('Cell contents assignment to a non-cell array object.');
                case '.'
                    a.(s(1).subs) = b;
                otherwise
                    error('Not a valid indexing expression.');
            end

        end

        function index = subsindex(a)
            index = double(a.Data) - 1;
        end

    end

end

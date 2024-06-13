classdef BackendMatrixTorch < BackendMatrix

    properties (SetAccess = immutable)
        torch
        Data
    end

    methods

        function obj = BackendMatrixTorch(torch, torchArray)
            obj.torch = torch;
            obj.Data = torchArray; % torch.Tensor
        end

        % arithmetic

        function result = plus(a, b)

            if isnumeric(b) b = w(b); end %#ok<*SEPEX>
            result = BackendMatrixTorch(a.torch, a.Data + b.Data);
        end

        function result = minus(a, b)
            if isnumeric(b) b = w(b); end
            result = BackendMatrixTorch(a.torch, a.Data - b.Data);
        end

        function result = uminus(a)
            resultData = a.torch.zeros_like(a.Data) - a.Data;
            result = BackendMatrixTorch(a.torch, resultData);
        end

        function result = uplus(a)
            resultData = a.torch.t_copy(a.Data);
            result = BackendMatrixTorch(a.torch, resultData);
        end

        function result = times(a, b)
            if isnumeric(b) b = w(b); end
            result = BackendMatrixTorch(a.torch, a.Data * b.Data);
        end

        function result = mtimes(a, b)
            if isnumeric(b) b = w(b); end
            resultData = a.torch.matmul(a.Data, b.Data);
            result = BackendMatrixTorch(a.torch, resultData);
        end

        function result = rdivide(a, b)
            if isnumeric(b) b = w(b); end
            result = BackendMatrixTorch(a.torch, a.Data / b.Data);
        end

        function result = ldivide(a, b)
            if isnumeric(b) b = w(b); end
            result = BackendMatrixTorch(a.torch, b.Data / a.Data);
        end

        function result = mrdivide(a, b)
            if isnumeric(b) b = w2(a.torch, b); end
            resultData = a.torch.matmul(a.Data, a.torch.inverse(b.Data));
            result = BackendMatrixTorch(a.torch, resultData);
        end

        function result = mldivide(a, b)
            if isnumeric(b) b = w2(a.torch, b); end
            resultData = a.torch.matmul(a.torch.inverse(b.Data), a.Data);
            result = BackendMatrixTorch(a.torch, resultData);
        end

        function result = power(a, b)
            result = BackendMatrixTorch(a.torch, a.Data ^ b.Data);
        end

        function result = mpower(a, b)
            if ~isnumeric(b) error('Not Implemented'); end
            resultData = a.torch.linalg.matrix_power(a.Data, b);
            result = BackendMatrixTorch(a.torch, resultData);
        end

        % comparison

        function result = lt(a, b)
            if isnumeric(b) b = w2(a.torch, b); end
            result = BackendMatrixTorch(a.torch, a.Data < b.Data);
        end

        function result = gt(a, b)
            if isnumeric(b) b = w2(a.torch, b); end
            result = BackendMatrixTorch(a.torch, a.Data > b.Data);
        end

        function result = le(a, b)
            if isnumeric(b) b = w2(a.torch, b); end
            result = BackendMatrixTorch(a.torch, a.Data <= b.Data);
        end

        function result = ge(a, b)
            if isnumeric(b) b = w2(a.torch, b); end
            result = BackendMatrixTorch(a.torch, a.Data >= b.Data);
        end

        function result = ne(a, b)
            if isnumeric(b) b = w2(a.torch, b); end
            result = BackendMatrixTorch(a.torch, a.Data ~= b.Data);
        end

        function result = eq(a, b)
            if isnumeric(b) b = w2(a.torch, b); end
            result = BackendMatrixTorch(a.torch, a.Data == b.Data);
        end

        function result = and(a, b)
            if isnumeric(b) b = w2(a.torch, b); end
            resultData = a.torch.logical_and(a.Data, b.Data);
            result = BackendMatrixTorch(a.torch, resultData);
        end

        function result = or(a, b)
            if isnumeric(b) b = w2(a.torch, b); end
            resultData = a.torch.logical_and(a.Data, b.Data);
            result = BackendMatrixTorch(a.torch, resultData);
        end

        function result = not(a)
            resultData = a.torch.logical_not(a.Data);
            result = BackendMatrixTorch(a.torch, resultData);
        end

        % misc

        function result = ctranspose(a)
            result = BackendMatrixTorch(a.Data.transpose(py.int(0), py.int(1)));
        end

        function result = transpose(a)
            result = BackendMatrixTorch(a.Data.transpose(py.int(0), py.int(1)));
        end

        function result = colon(a, d, b) %#ok<STOUT,INUSD>
            error('Unsupported');
        end


        function result = horzcat(varargin)
            data = cellfun(@(x) x.Data, varargin, 'UniformOutput', false);
            concatenatedData = horzcat(data{:});
            result = BackendMatrixTorch(concatenatedData);
        end

        function result = vertcat(varargin)
            data = cellfun(@(x) x.Data, varargin, 'UniformOutput', false);
            concatenatedData = vertcat(data{:});
            result = BackendMatrixTorch(concatenatedData);
        end

        function result = subsref(a, s)
            disp(s);
            switch s(1).type
                case '()'
                    result = BackendMatrixTorch(a.Data(s.subs{:}));
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

                    if isa(b, 'BackendMatrixTorch')
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

function b = w(a)
    bb = py.numpy.array(a);
    b = struct();
    b.Data = bb;
end

function b = w2(t, a)
    bb = py.numpy.array(a);
    bbb = t.array(bb);
    b = struct();
    b.Data = bbb;
end

classdef (Abstract) BackendMatrix

    properties (Abstract, SetAccess = immutable)
        Data
    end

    methods (Abstract)
        result = plus(a, b)
        result = minus(a, b)
        result = uminus(a)
        result = uplus(a)
        result = times(a, b)
        result = mtimes(a, b)
        result = rdivide(a, b)
        result = ldivide(a, b)
        result = mrdivide(a, b)
        result = mldivide(a, b)
        result = power(a, b)
        result = mpower(a, b)
        result = lt(a, b)
        result = gt(a, b)
        result = le(a, b)
        result = ge(a, b)
        result = ne(a, b)
        result = eq(a, b)
        result = and(a, b)
        result = or(a, b)
        result = not(a)
        result = colon(a, d, b)
        result = ctranspose(a)
        result = transpose(a)
        result = horzcat(varargin)
        result = vertcat(varargin)
        result = subsref(a, s)
        result = subsasgn(a, s, b)
        result = subsindex(a)
    end

end

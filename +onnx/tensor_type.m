classdef tensor_type
    enumeration
        UNDEFINED (0)
        FLOAT (1)
        UINT8 (2)
        INT8 (3)
        UINT16 (4)
        INT16 (5)
        INT32 (6)
        INT64 (7)
        STRING (8)
        BOOL (9)
        FLOAT16 (10)
        DOUBLE (11)
        UINT32 (12)
        UINT64 (13)
        COMPLEX64 (14)
        COMPLEX128 (15)
        BFLOAT16 (16)
        FLOAT8E4M3FN (17)
        FLOAT8E4M3FNUZ (18)
        FLOAT8E5M2 (19)
        FLOAT8E5M2FNUZ (20)
        UINT4 (21)
        INT4 (22)
    end
    
    properties(SetAccess=immutable)
        Value
    end
    
    methods
        function obj = tensor_type(val)
            obj.Value = val;
        end
    end
end
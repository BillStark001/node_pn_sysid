classdef node < handle
    properties(SetAccess=immutable)
        input % cell array of strings
        output % cell array of strings
        name % string
        op_type % string
        domain % string
        overload % string
        doc_string % string
        attribute
        metadata_props
    end
    
    methods
        function obj = node(varargin)
            % Constructor to initialize the properties
            if nargin == 0
                return;
            end
            p = inputParser;
            addParameter(p, 'input', {}, @(x) iscell(x) && all(cellfun(@ischar, x)));
            addParameter(p, 'output', {}, @(x) iscell(x) && all(cellfun(@ischar, x)));
            addParameter(p, 'name', '', @ischar);
            addParameter(p, 'op_type', '', @ischar);
            addParameter(p, 'domain', '', @ischar);
            addParameter(p, 'overload', '', @ischar);
            addParameter(p, 'attribute', [], @(x) isa(x, 'AttributeProto'));
            addParameter(p, 'doc_string', '', @ischar);
            addParameter(p, 'metadata_props', [], @(x) isa(x, 'StringStringEntryProto'));
            parse(p, varargin{:});
            
            obj.input = p.Results.input;
            obj.output = p.Results.output;
            obj.name = p.Results.name;
            obj.op_type = p.Results.op_type;
            obj.domain = p.Results.domain;
            obj.overload = p.Results.overload;
            obj.doc_string = p.Results.doc_string;
            obj.attribute = p.Results.attribute;
            obj.metadata_props = p.Results.metadata_props;
        end
        
    end
end
classdef value_info < handle
    properties(SetAccess=immutable)
        name % string
        type
        doc_string % string
        metadata_props
    end

    methods
        function obj = node(varargin)
            % Constructor to initialize the properties
            if nargin == 0
                return;
            end
            addParameter(p, 'name', '', @ischar);
            addParameter(p, 'type', '', @ischar);
            addParameter(p, 'doc_string', '', @ischar);
            addParameter(p, 'metadata_props', [], @(x) isa(x, 'StringStringEntryProto'));
            parse(p, varargin{:});
            
            obj.name = p.Results.name;
            obj.type = p.Results.type;
            obj.doc_string = p.Results.doc_string;
            obj.metadata_props = p.Results.metadata_props;
        end
        
    end


end
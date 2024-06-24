function w2r = prepare_weight_dict(w, cache)
    w2 = struct();
    fields = fieldnames(w);

    for i = 1:numel(fields)
        field = fields{i};
        w2.(field) = syntax_tree.st2py(w.(field), cache);
    end

    w2r = syntax_tree.st2py(w2);
end

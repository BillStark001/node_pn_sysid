a = BackendMatrixSyntaxTree("a", "var", [1 2;3 4], {});
b = BackendMatrixSyntaxTree("b", "var", [1 2;3 4], {});
c = BackendMatrixSyntaxTree("c", "var", [1 2;3 4], {});
d = BackendMatrixSyntaxTree("d", "var", [1 2;3 4], {});

slice = a:2:b;
arr = [1 2 3 4];

% g = arr(a);
g = a.gansihuangxudong;
d(slice) = 114514;
f = c(1:3:5);
e = c(slice);

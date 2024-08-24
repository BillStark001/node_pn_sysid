function result = test_mat_func(A)

  % check if is square matrix

  B = [1 2; 3 4];
  B(1, 1) = 5;

  [rows, cols] = size(A);
  if rows ~= cols
      error('Input must be a square matrix.');
  end
  
  % calc eig
  eigenvalues = eig(A);
  sumEig = sum(eigenvalues);
  
  % create result
  result = struct('SumOfEigenvalues', sumEig, 'Determinant', NaN);
  
  % check if is symmetric, calc determinant if yes
  if isequal(A, A')
      detA = det(A);
      result.Determinant = detA;

      for i = 1:numel(detA)
          disp(detA(i))
      end
  end
end
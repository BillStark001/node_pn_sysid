classdef TorchBackend < Backend

  properties(SetAccess = immutable)
    torch
  end

  methods

    function obj = TorchBackend(torch)
      obj.torch = torch;
    
    end

    function result = arrayRaw(obj, matlabArray)
      result = obj.torch.tensor(py.numpy.array(matlabArray));
    end

    function result = array(obj, matlabArray)
      resultRaw = obj.arrayRaw(matlabArray);
      result = BackendMatrixTorch(obj.torch, resultRaw);
    end


  end



end
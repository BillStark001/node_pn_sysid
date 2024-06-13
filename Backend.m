classdef (Abstract) Backend < handle


  methods (Abstract)

    result = array(matlabArray)
    result = arrayRaw(matlabArray)

  end


end
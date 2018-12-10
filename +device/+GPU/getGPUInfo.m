% Author: Zhao Mingxin
% Date:   2018/12/10
% Description: as below

function res = getGPUInfo()
    try
        res = gpuDevice;
    catch
        res = 0;
    end
end
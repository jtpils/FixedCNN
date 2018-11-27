% Author: Zhao Mingxin
% Date:   2018/11/25
% Description: Get current workers number.

function res = GetCurrentCore()
    p_info = gcp('nocreate'); 
    if isempty(p_info)
        res = 0;
    else
        res = p_info.NumWorkers;
    end
end
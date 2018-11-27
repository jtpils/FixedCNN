% Author: Zhao Mingxin
% Date:   2018/11/27
% Description: Get current workers number.

function res = TurnOnMultiCore()
    num_core = GetCurrentCore();
    if num_core == 0
        try
            parpool
        catch error
            warning('Can''t turn on multicore mode on this machine');
            res = 0;
        end
    else
        disp('Multicore mode is already on');
        res = 1;
    end
end
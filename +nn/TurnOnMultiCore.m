% Author: Zhao Mingxin
% Date:   2018/11/27
% Description: Turn on MultiCore mode. If fails, it will return 0.

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
        fprintf(2,'MultiCore mode is already ON.\n');
        res = 1;
    end
end
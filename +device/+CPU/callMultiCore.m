% Author: Zhao Mingxin
% Date:   2018/11/27
% Description: Turn on MultiCore mode. If fails, it will return 0.

function res = TurnOnMultiCore()
    num_core = GetCurrentCore();
    if num_core == 0
        try
            parpool;
            num_core = GetCurrentCore();
            if num_core>0
                fprintf(2,'Successfully Turn On MultiCore Mode.\n');
                fprintf(2,'%d Cores Detected ...\n',num_core);
                res = 1;
            end
        catch
            warning('Can''t Turn On MultiCore Mode On This Machine.\n');
            res = 0;
        end
    else
        fprintf(2,'MultiCore Mode is already ON.\n');
        fprintf(2,'%d Cores Detected ...\n',num_core);
        res = 1;
    end
end
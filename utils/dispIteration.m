function dispIteration(MsgBox, prt)
% Content: gate function to control the output of print info
% Input:
%       MsgBox: string, print info
%       prt: logical, 1 for printing info and 0 for not printing info
%
% Output:
%       None
%
%
%

if prt == 1
    disp(MsgBox);
end

end
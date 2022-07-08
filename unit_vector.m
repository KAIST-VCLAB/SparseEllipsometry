%=======================================================================
% Project: Microscopic SVBRDF
%
%   Normalize a vector
%   Input: vector
%   Output: normalized vector
%
% VCLAB, KAIST
% by Giljoo Nam
% Created in 2015/01/07
%=======================================================================
function [Np, denorm] = unit_vector(N, option)

if (strcmp(option, 'column')) % if column-wise vectors (3xN)
    denorm = sqrt(sum(N.^2,1));
    Np = N./ repmat(denorm, 3, 1);
elseif (strcmp(option, 'row')) % if row-wise vectors (Nx3)
    denorm = sqrt(sum(N.^2,2));
    Np = N./ repmat(denorm, 1, 3);
else
    disp('determine column/row vectors');
end

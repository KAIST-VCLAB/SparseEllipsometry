
function [dop,dop_partial]= compute_d_dop_partial( zenith, rfr_ind)

rfr_ind_sq = rfr_ind.^2;

if numel(rfr_ind) == 1
    rfr_ind = repmat(rfr_ind, size(zenith));
    rfr_ind_sq = repmat(rfr_ind_sq, size(zenith));
end

one = ones(size(rfr_ind));




sin_zenith = sin(zenith);
cos_zenith = cos(zenith);
sin_zenith_sq = (sin_zenith.^2);

rfr_ind_cos = sqrt(rfr_ind.^2 - sin_zenith_sq);

f = ((rfr_ind_sq - one).^2) .* sin_zenith_sq;
g = (2.*rfr_ind_sq + 2*rfr_ind_sq.^2 - ((rfr_ind_sq + one).^2) .* sin_zenith_sq + 4.*rfr_ind_sq.*cos_zenith.*rfr_ind_cos);
f_prime = 4.*rfr_ind.*(rfr_ind_sq - one) .* sin_zenith_sq;

g_prime = 4.*rfr_ind +8.*rfr_ind.*rfr_ind_sq -4.*rfr_ind.*(rfr_ind_sq+one).*sin_zenith_sq ...
    + 8.*rfr_ind.*cos_zenith.*rfr_ind_cos + 4.*rfr_ind.*rfr_ind_sq.*cos_zenith./rfr_ind_cos;


dop = f./g;
dop_partial = (f_prime.*g - f.*g_prime)./(g.^2);

%syms dop(rfr_ind,zenith)
%dop(rfr_ind,zenith) = (((rfr_ind.^2 - 1).^2) .* sin(zenith).^2)/(2.*rfr_ind.^2 + 2*rfr_ind.^4 - ((rfr_ind.^2 + 1).^2) .* sin(zenith).^2 + 4.*(rfr_ind.^2).*cos(zenith).*sqrt(rfr_ind.^2 - sin(zenith).^2));
% dop_drfr_ind = matlabFunction(diff(dop,rfr_ind));



% dop_partial = -sin(zenith).^2.*(rfr_ind.^2-1.0).^2.*1.0./(-sin(zenith).^2.*(rfr_ind.^2+1.0).^2+rfr_ind.^2.*2.0+rfr_ind.^4.*2.0+rfr_ind.^2.*cos(zenith).*sqrt(-sin(zenith).^2+rfr_ind.^2).*4.0).^2.*(rfr_ind.*4.0+rfr_ind.^3.*8.0+rfr_ind.^3.*cos(zenith).*1.0./sqrt(-sin(zenith).^2+rfr_ind.^2).*4.0-rfr_ind.*sin(zenith).^2.*(rfr_ind.^2+1.0).*4.0+rfr_ind.*cos(zenith).*sqrt(-sin(zenith).^2+rfr_ind.^2).*8.0)+(rfr_ind.*sin(zenith).^2.*(rfr_ind.^2-1.0).*4.0)./(-sin(zenith).^2.*(rfr_ind.^2+1.0).^2+rfr_ind.^2.*2.0+rfr_ind.^4.*2.0+rfr_ind.^2.*cos(zenith).*sqrt(-sin(zenith).^2+rfr_ind.^2).*4.0);

end

function dop = compute_d_dop( zenith, rfr_ind)

if numel(rfr_ind) == 1
    rfr_ind = repmat(rfr_ind, size(zenith));
end

one = ones(size(rfr_ind));

sin_zenith = sin(zenith);
cos_zenith = cos(zenith);

dop = ((rfr_ind - (one./rfr_ind)).^2) .* sin_zenith.^2;
dop = dop ./ (2 + 2*rfr_ind.^2 - ((rfr_ind + one./rfr_ind).^2) .* (sin_zenith.^2) + 4*cos_zenith.*sqrt(rfr_ind.^2 - sin_zenith.^2));

end
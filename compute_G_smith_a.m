% compute smith G
% INPUT
% - n: shading normal
% - v: view vector
% - i: illumination vector
% - m: roughness
% OUTPUT
% - G: smith G
function G = compute_G_smith_a(m, theta_i, theta_o)
tan_nv = tan(theta_o);
tan_ni = tan(theta_i);

two_mat = zeros(size(m,1),1);
two_mat(:) = 2;
G = (two_mat./(1 + sqrt(1 + (m.^2) .* (tan_ni.^2)))) .* (two_mat./(1 + sqrt(1 + (m.^2) .* (tan_nv.^2)))); % Geoemtric factor
end
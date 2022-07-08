% compute GGX NDF
% INPUT
% - m: roughness
% - n: shading normal
% - h: half-way vector between the view and the light
% OUTPUT
% - D: Beckmann GGX term 
function D = compute_D_GGX_a(m,theta_h)
cos_hn = cos(theta_h);
tan_hn = tan(theta_h);

%D = (m.^2) ./ max(eps, pi*(cos_hn.^4).*((m.^2 + tan_hn.^2).^2));
D = (m.^2) ./ (pi*(cos_hn.^4).*((m.^2 + tan_hn.^2).^2)); % sum of D should be one? not now.
end



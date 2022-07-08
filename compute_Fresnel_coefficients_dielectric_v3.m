% Compute Fresnel coefficient
% INPUT
% - cos_theta: if is_exitant_angle = 0, cosine of the incident angle 
%              if is_exitant_angle = 1, cosine of the exitant angle 
%       dim: [# of px, 1]
% - n1: refractive index of the incident medium
%       dim: [# of px, 1]
% - n2: refractive index of the exitant medium
%       dim: [# of px, 1]
% OUTPUT
% - Rs: perpendicular reflection coefficient 
%       dim: [# of px, 1]
% - Rp: parallel reflection coefficient 
%       dim: [# of px, 1]
% - Ts: perpendicular transmission coefficient 
%       dim: [# of px, 1]
% - Tp: parallel transmission coefficient 
%       dim: [# of px, 1]

function [Rs, Rp, Ts, Tp] = compute_Fresnel_coefficients_dielectric_v2(cos_theta, n1, n2, is_exitant_angle)

if is_exitant_angle == 0
    % theta_i is the incident angle
    cos_theta_i = cos_theta;
    sin_theta_i = sqrt(1 - cos_theta_i.^2);
else
    % theta_i is the exitant angle
    % employ snell's law
    cos_theta_e = cos_theta;
    sin_theta_e = sqrt(1 - cos_theta_e.^2);
    sin_theta_i = (n2./n1) .* sin_theta_e;
    cos_theta_i = sqrt(1 - sin_theta_i.^2);
end

Rs =((n1 .* cos_theta_i - n2 .* sqrt( 1 - ((n1./n2).*sin_theta_i).^2 ) ) ./ (n1 .* cos_theta_i + n2 .* sqrt( 1 - ((n1./n2).*sin_theta_i).^2 ) )).^2;
Rp =((n1 .* sqrt( 1 - ((n1./n2).*sin_theta_i).^2 ) - n2 .* cos_theta_i ) ./ (n1 .* sqrt( 1 - ((n1./n2).*sin_theta_i).^2 ) + n2 .* cos_theta_i )).^2;

Ts = 1 - Rs;
Tp = 1 - Rp;

end
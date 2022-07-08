function loss = obj_parameters_and_normal_azimuth_wo_linear(x,diffuse,specular,aolp,l,v,h,hi,y_v, psi, norm_mean_color)

normal = x(1:3)';
normal = normal./sqrt(sum(normal.^2));
m1 = x(4);
m2 = x(5);
eta = x(6);
ks1 = x(7);

%%
z_o = v;
y_o = y_v - sum(y_v.*z_o,1).*z_o;
y_o = unit_vector(y_o,'column');
x_o = cross(y_o,z_o,1);
sin_azimuth = sum(y_o.*normal,1);
cos_azimuth = sum(x_o.*normal,1);
azimuth_denom = sqrt(sin_azimuth.*sin_azimuth+cos_azimuth.*cos_azimuth);
sin_azimuth = sin_azimuth./azimuth_denom;
cos_azimuth = cos_azimuth./azimuth_denom;
alpha_o = -2*sin_azimuth.*cos_azimuth;
beta_o = 2*cos_azimuth.*cos_azimuth-1;

%%
ni = sum(normal.*l,1);
nh = sum(normal.*h,1);
no = sum(normal.*v,1);
theta_h = acos(nh);
theta_i = acos(ni);
theta_o = acos(no);

%%
[Rs_i,Rp_i,~,~] = compute_Fresnel_coefficients_dielectric_v3(hi, 1, eta, 0);
Rpos = (Rs_i + Rp_i)/2;
%% specular coefficient
D1 = compute_D_GGX_a(m1,theta_h);
D2 = compute_D_GGX_a(m2,theta_h);

G1 = compute_G_smith_a(m1, theta_i, theta_o);
G2 = compute_G_smith_a(m2, theta_i, theta_o);
% DG1= (D1.*G1);
DG1= ks1.*(D1.*G1);
% DG2= (D2.*G2);
% A = [DG2;zeros(3,size(DG2,2));DG2;zeros(3,size(DG2,2));DG2];
% A = reshape(A,3,[])';
% % A=[repmat(DG1',3,1),A];
% % b = specular.*psi./Rpos.*4.*cos(theta_o);
% b = specular.*psi./Rpos.*4.*cos(theta_o)-DG1;
% b = reshape(b',[],1);
% x_est = A\b;
% x_est(x_est<0)=0;
% % ks1 = x_est(1);
% % ks2 = x_est(2:4);
% ks2 = x_est(1:3);
ks2= ks2_lsq(D2,G2,DG1,Rpos,theta_o,psi,1,specular);

if any(isnan(ks2))
    ks2 = zeros(3:1);
    
end

DG1= ks1 .* (D1.*G1);
DG2= ks2 .* (D2.*G2);

% est_S = (DG1+DG2).*Rpos./4./cos(theta_o);
est_S = (DG2).*Rpos./4./cos(theta_o);

s_diff = est_S./psi-specular;
specular_loss = sum(s_diff(:).^2);

%%

% aolp_beta = -s_diff;
aolp_beta = specular-DG2.*Rpos./4./cos(theta_o)./psi;
% aolp_beta = specular;
diffuse_mono = sum(norm_mean_color.*diffuse,1);
aolp_beta_mono = sum(norm_mean_color.*aolp_beta,1); 
aolp_mono = sum(norm_mean_color.*aolp,1);
diffuse_polarization = sqrt(aolp_mono.^2+aolp_beta_mono.^2);

dop = diffuse_polarization./diffuse_mono;
% dop(dop>1) = 1;
predict_dop = compute_d_dop(theta_o,eta');
dop_loss = sum((dop - predict_dop).^2,'all');

%%
[~,~,Ts_i,Tp_i] = compute_Fresnel_coefficients_dielectric_v3(ni, 1, eta', 0);
[~,~,Ts_o,Tp_o] = compute_Fresnel_coefficients_dielectric_v3(no, eta', 1, 1);
Tpos_o = (Ts_o + Tp_o)/2;
Tpos_i = (Ts_i + Tp_i)/2;
TpoTpi = Tpos_o.*Tpos_i;

rho = mean(diffuse.*psi,2)./mean(TpoTpi.*ni,2);

rho_loss = sum((diffuse-TpoTpi.*ni.*rho./psi).^2,'all');
%% azimuth
azimuth_loss = sum((aolp_mono - alpha_o.*diffuse_polarization).^2 + (aolp_beta_mono - beta_o.*diffuse_polarization).^2,'all');

%%
loss = 10*dop_loss+specular_loss+rho_loss+azimuth_loss;

end
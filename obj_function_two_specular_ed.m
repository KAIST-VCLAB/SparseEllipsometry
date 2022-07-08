function loss = obj_function_two_specular_ed(x, theta_h, Rpos, S, angle_weight)

m1 = x(1);
m2 = x(2);
ks1 = x(3);
ks2 = x(4:6)';
%% distance sampling
angles_1 = (0.01:0.01:1-0.01);
angles = angles_1.*pi./2;
new_angle_spase = - log(abs(csc(angles)+ cot(angles)));
%% D term: NDF
D1 = compute_D_GGX_a(m1,angles);
D2 = compute_D_GGX_a(m2,angles);

%% G term
G1 = compute_G_smith_a(m1, angles, angles);
G2 = compute_G_smith_a(m2, angles, angles);

%% compute DG
DG1= ks1 .* (D1.*G1);
DG2= ks2 .* (D2.*G2);



%% sample the cluster region
est_S = (DG1+DG2).*Rpos./4./cos(angles);

%% find nearest neighbor
theta_h(theta_h<0.0001) = 0.0001;
new_theta_h = - log(abs(csc(theta_h)+ cot(theta_h)));
% [~,s_diff] = knnsearch([angles_1'*sqrt(10),est_S'],[theta_h'./(pi./2)*sqrt(10),S']);
% [~,s_diff] = knnsearch([angles'*sqrt(angle_weight),est_S'],[theta_h'*sqrt(angle_weight),S']);
[~,s_diff] = knnsearch([new_angle_spase'*sqrt(angle_weight),est_S'],[new_theta_h'*sqrt(angle_weight),S']);

%% compute the loss
loss = mean(s_diff(:).^2)+0.000001.*ks1.^2;


end
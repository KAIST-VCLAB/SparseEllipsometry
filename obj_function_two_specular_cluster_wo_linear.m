function loss = obj_function_two_specular_cluster_wo_linear(x, theta_h, theta_i, theta_o, Rpos, psi, S ,weight)

m1 = x(1);
m2 = x(2);
ks1 = x(3);

%% D term: NDF
D1 = compute_D_GGX_a(m1,theta_h);
D2 = compute_D_GGX_a(m2,theta_h);

%% G term
G1 = compute_G_smith_a(m1, theta_i, theta_o);
G2 = compute_G_smith_a(m2, theta_i, theta_o);

%% compute DG
DG1= ks1 .* (D1.*G1);

ks2 =  ks2_lsq(D2,G2,DG1,Rpos,theta_o,psi,weight,S);

DG2= ks2 .* (D2.*G2);

%% sample the cluster region
est_S = (DG1+DG2).*Rpos./4./cos(theta_o);

%% compute the loss
s_diff = est_S./psi-S;
weight_diff = s_diff.^2.*weight;
loss = sum(weight_diff(:),'omitnan');


end
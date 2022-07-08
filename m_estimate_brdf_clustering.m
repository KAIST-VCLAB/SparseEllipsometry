%% parameters
max_iteration = 3;
norm_exp = 1;
initial_ref_idx = 1.5;
initial_m1 = 0.5;
initial_m2 = 1;
initial_ks1 = ones(1,1);
initial_ks2 = ones(3,1);
batch_size = 5000;
opt_max_iter = 1000;
cluster_weight = 0.1;
weight_sigma = 0.1;
color_threshold = 0.2;
color_threshold_sq = color_threshold.^2;
FiniteDifferenceStepSize = sqrt(eps);
MaxFunctionEvaluations = 3000;
optAlgorithm = 'sqp';
options = optimoptions('fmincon','Display','off',...
    'MaxIterations',opt_max_iter, ...
    'FiniteDifferenceStepSize',FiniteDifferenceStepSize, ...
    'MaxFunctionEvaluations',MaxFunctionEvaluations, ...
    'Algorithm',optAlgorithm);

options_gradient = optimoptions('fmincon','Display','off',...
    'MaxIterations',opt_max_iter, ...
    'FiniteDifferenceStepSize',FiniteDifferenceStepSize, ...
    'MaxFunctionEvaluations',MaxFunctionEvaluations, ...
    'SpecifyObjectiveGradient',true, ...
    'Algorithm',optAlgorithm);

options_gradient_trust_region = optimoptions('fmincon','Display','off',...
    'MaxIterations',opt_max_iter, ...
    'FiniteDifferenceStepSize',FiniteDifferenceStepSize, ...
    'MaxFunctionEvaluations',MaxFunctionEvaluations, ...
    'SpecifyObjectiveGradient',true, ...
    'Algorithm','trust-region-reflective');

options_hessian = optimoptions('fmincon','Display','off',...
    'MaxIterations',opt_max_iter, ...
    'FiniteDifferenceStepSize',FiniteDifferenceStepSize, ...
    'MaxFunctionEvaluations',MaxFunctionEvaluations, ...
    'SpecifyObjectiveGradient',true, ...
    'HessianFcn','objective',...
    'Algorithm','trust-region-reflective');

intensity_threshold = 0.001;

meshCurrent.sample_mask = sample_mask;
%% load
% load(fullfile(path_figs,'cluster_brdf.mat'),'hist_interval','histogram_angles','hist_numer','hist_denom', ...
%     'hist_numer_virtual','hist_denom_virtual','result_m1','result_ks1','result_m2','result_ks2','result_eta','num_cluster','idx','distance_cluster')




%% Initialization
meshCurrent.eta = repmat(initial_ref_idx,1,N);
meshCurrent.m1 = zeros(1,N);
meshCurrent.m2 = zeros(1,N);
meshCurrent.ks1 = zeros(1,N);
meshCurrent.ks2 = zeros(3,N);
meshCurrent.rho = zeros(3,N);
eta_iter = zeros(1,N);
specular_iter = zeros(1,N);
local_eta_iter = zeros(1,batch_size);
local_specular_iter = zeros(1,batch_size);
local_eta = zeros(1,batch_size);
local_m1 = zeros(1,batch_size);
local_m2 = zeros(1,batch_size);
local_ks1 = zeros(1,batch_size);
local_ks2 = zeros(3,batch_size);
local_rho = zeros(3,batch_size);

num_angles = size(hist_numer,2);
angles = (1:num_angles) - 0.5;
angles = angles./num_angles.*pi./2;


%% 
time_record = zeros(6,ceil(N/batch_size));
for batch_idx = 1:ceil(N/batch_size)
% for batch_idx = 43
batch_offset =  (batch_idx-1)*batch_size;
time = clock;
fprintf("%d iter: %02d:%02d:%02.2f\n",batch_offset,time(4),time(5),time(6));
time_record(:,batch_idx) = time;
now_batch_size = min(batch_size,N-batch_offset);

local_V = V_outlier(batch_offset+1:batch_offset+now_batch_size,:);
local_V(:,~sample_mask(:)) = 0;
local_diffuse_I = diffuse_I(batch_offset+1:batch_offset+now_batch_size,:);
local_specular_I = specular_I(batch_offset+1:batch_offset+now_batch_size,:); 
local_aolp_I = aolp_I(batch_offset+1:batch_offset+now_batch_size,:);
local_zenith_mat = zenith_mat(batch_offset+1:batch_offset+now_batch_size,:);
local_azimuth_mat = azimuth_mat(batch_offset+1:batch_offset+now_batch_size,:);
local_vertices = meshCurrent.vertices(:,batch_offset+1:batch_offset+now_batch_size);
local_normals = meshCurrent.normals(:,batch_offset+1:batch_offset+now_batch_size);
local_psi = Psi(batch_offset+1:batch_offset+now_batch_size,:);
local_distance_cluster = distance_cluster(batch_offset+1:batch_offset+now_batch_size,:);
local_selected_exposure_indices = selected_exposure_indices(batch_offset+1:batch_offset+now_batch_size,:);
local_norm_mean_color = meshCurrent.norm_mean_color(:,batch_offset+1:batch_offset+now_batch_size);
parfor (point_idx_inbatch = 1:now_batch_size)
point_idx = batch_offset+point_idx_inbatch;
if ~region_mask(point_idx)
    continue
end
%% DoP calculation
warning('off','all');
visible = local_V(point_idx_inbatch,:);
if sum(visible)<8
    continue
end
diffuse = local_diffuse_I(point_idx_inbatch,[visible,visible,visible]);
specular = local_specular_I(point_idx_inbatch,[visible,visible,visible]);
aolp = local_aolp_I(point_idx_inbatch,[visible,visible,visible]);
psi = local_psi(point_idx_inbatch,visible)';
distance_cluster_vertex = local_distance_cluster(point_idx_inbatch,:)';
zenith = local_zenith_mat(point_idx_inbatch,visible)';
azimuth = local_azimuth_mat(point_idx_inbatch,visible)';
diffuse = reshape(squeeze(diffuse),[],3);
specular = reshape(squeeze(specular),[],3);
aolp = reshape(squeeze(aolp),[],3);

exposure_mask = diffuse(:,1)>intensity_threshold | diffuse(:,2)>intensity_threshold | diffuse(:,3)>intensity_threshold;


%% Albedo channel

color_norm = sqrt(sum(diffuse.^2,2));
normalized_color = diffuse./color_norm;

mean_color = mean(normalized_color,1,'omitnan');
mean_color = mean_color./sqrt(sum(mean_color.^2));
color_distance = sum((normalized_color-mean_color).^2,2);
color_mask = color_distance<color_threshold_sq;
for i = 1:3
    normalized_color_filtered = normalized_color(color_mask,:);
    mean_color = mean(normalized_color_filtered,1,'omitnan');
    mean_color = mean_color./sqrt(sum(mean_color.^2));
    color_distance = sum((normalized_color-mean_color).^2,2);
    color_mask = color_distance<color_threshold_sq;
end

exposure_mask = exposure_mask & color_mask;
if sum(exposure_mask)<8
    continue
end

norm_mean_color = mean_color.^norm_exp;
norm_mean_color = norm_mean_color./sum(mean_color);
norm_mean_color = norm_mean_color.^(1/norm_exp);


%%
diffuse_masked = diffuse(exposure_mask,:);
specular_masked = specular(exposure_mask,:);
aolp_masked = aolp(exposure_mask,:);
zenith_masked = zenith(exposure_mask);
azimuth_masked = azimuth(exposure_mask);
psi_masked = psi(exposure_mask);

diffuse_mono = sum(norm_mean_color.*diffuse_masked,2);
specular_mono = sum(norm_mean_color.*specular_masked,2);
aolp_mono = sum(norm_mean_color.*aolp_masked,2);
aolp_beta_mono = specular_mono;
%% geometric information
pPos = local_vertices(:,point_idx_inbatch);
normal = local_normals(:,point_idx_inbatch);
vVec = vPos(:,visible) - pPos;
vVec = unit_vector(vVec, 'column');
lVec = lPos(:,visible) - pPos;
lVec = unit_vector(lVec, 'column');
hVec = unit_vector(lVec+vVec, 'column');
ni = sum(normal.*lVec,1);
hi = sum(hVec.*lVec,1);
nh = sum(normal.*hVec,1);
no = sum(normal.*vVec,1);
theta_h = acos(nh);
theta_i = acos(ni);
theta_o = acos(no);
ni_masked = double(ni(:,exposure_mask));
no_masked = double(no(:,exposure_mask));


%% iterations
eta = initial_ref_idx;
m1 = initial_m1;
m2 = initial_m2;
ks1 = initial_ks1;
ks2 = initial_ks2;
for brdf_iter = 1:max_iteration

    
    dop_mono = sqrt(aolp_mono.^2+aolp_beta_mono.^2)./diffuse_mono;


%%

dop_mask = (dop_mono<=1)& specular_mono<diffuse_mono & specular_mono<0.5;
if sum(dop_mask)<3
    dop_mask(:) = 1;
    dop_mono(dop_mono>1)=1;
end

x0 = eta;
[x_est,fval,exitflag,output,~,~] = ...
    fmincon(@(x)obj_function_eta_dop_zenith_partial(x,double(dop_mono(dop_mask)),double(zenith_masked(dop_mask))),x0,[],[],[],[],[1.1],[3],[],options_gradient);


local_eta_iter(:,point_idx_inbatch) = output.iterations;

eta = x_est;



%% Specular reflection estimation

theta_h_virtual = double(theta_h);
theta_i_virtual = double(theta_i);
theta_o_virtual = double(theta_o);




[~,idx] = sort(distance_cluster_vertex);
save_idx = idx(1);
saved_m1 = result_m1(save_idx);
saved_ks1 = result_ks1(save_idx);
histogram_value = squeeze(hist_virtual(save_idx,:,:));
distance_cluster_saved = distance_cluster_vertex(save_idx);

histogram_value(histogram_value>1) = 1;

theta_h_virtual = [theta_h_virtual,repmat(angles(:),1,1)'];
theta_i_virtual = [theta_i_virtual,repmat(angles(:),1,1)'];
theta_o_virtual = [theta_o_virtual,repmat(angles(:),1,1)'];

[Rs_i,Rp_i,~,~] = compute_Fresnel_coefficients_dielectric_v3([double(hi),ones(1,num_angles)], 1, double(eta), 0);
Rpos = (Rs_i + Rp_i)/2;

psi_virtual = [double(psi'),double(virtual_psi)*ones(1,num_angles)];
specular_virtual = [double(specular'),histogram_value'];


weight = exp(-(distance_cluster_saved.^2)./2./(weight_sigma.^2));

direct_weight = weight;
if direct_weight(1)==0
    direct_weight(1) = 1;
end
direct_weight = direct_weight./sum(direct_weight(:));
min_m1 = min(saved_m1);
m1 = sum(direct_weight.*saved_m1);
ks1 = sum(direct_weight.*saved_ks1);

weight = weight.*(cos(angles));
weight = weight';
weight_mask = weight > 0.01;

theta_h_virtual = theta_h_virtual(:,[ones(1,size(specular,1),'logical'),weight_mask(:)']);
theta_i_virtual = theta_i_virtual(:,[ones(1,size(specular,1),'logical'),weight_mask(:)']);
theta_o_virtual = theta_o_virtual(:,[ones(1,size(specular,1),'logical'),weight_mask(:)']);
Rpos = Rpos(:,[ones(1,size(specular,1),'logical'),weight_mask(:)']);
psi_virtual = psi_virtual(:,[ones(1,size(specular,1),'logical'),weight_mask(:)']);
specular_virtual = specular_virtual(:,[ones(1,size(specular,1),'logical'),weight_mask(:)']);

real_weight = ones(size(specular,1),1).*(1-cluster_weight)./size(specular,1);
weight = weight(weight_mask);
weight = weight./sum(weight(:)).*cluster_weight;
weight = [real_weight(:);weight(:)];
weight = weight';

x0 = [m1,m2,ks1'];
[x_est,fval,exitflag,output,~,~] = ...
fmincon(@(x)obj_function_two_specular_cluster_wo_linear(x,theta_h_virtual,theta_i_virtual,theta_o_virtual,Rpos,psi_virtual,specular_virtual,weight),x0,[1 -1 0],[0],[],[],[min_m1./2 0 0],[1 1 double(virtual_psi).*5]',[],options);


m1 = x_est(1);
m2 = x_est(2);
ks1 = x_est(3)';

local_specular_iter(:,point_idx_inbatch) = output.iterations;


%%

D = compute_D_GGX_a(m1,theta_h);
G = compute_G_smith_a(m1, theta_i, theta_o);
DG1= ks1 .* (D.*G);
D = compute_D_GGX_a(m2,theta_h);
G = compute_G_smith_a(m2, theta_i, theta_o);
[Rs_i,Rp_i,~,~] = compute_Fresnel_coefficients_dielectric_v3(double(hi), 1, double(eta), 0);
Rpos = (Rs_i + Rp_i)/2;

ks2 = ks2_lsq(D,G,DG1,Rpos,theta_o,psi',1,specular');

DG2= ks2 .* (D.*G);

%%
[Rs_i,Rp_i,~,~] = compute_Fresnel_coefficients_dielectric_v3(hi, 1, double(eta), 0);
Rpos = (Rs_i + Rp_i)/2;

D = compute_D_GGX_a(m1,theta_h);
G = compute_G_smith_a(m1, theta_i, theta_o);
DG1= ks1 .* (D.*G);
D = compute_D_GGX_a(m2,theta_h);
G = compute_G_smith_a(m2, theta_i, theta_o);
DG2= ks2 .* (D.*G);
est_S = (DG2).*Rpos./4./cos(theta_o)./psi';


est_S_mono = sum(norm_mean_color'.*est_S,1);
aolp_beta_mono = specular_mono-est_S_mono(exposure_mask)';

%% albedo
[~,~,Ts_i,Tp_i] = compute_Fresnel_coefficients_dielectric_v3(ni_masked, 1, eta', 0);
[~,~,Ts_o,Tp_o] = compute_Fresnel_coefficients_dielectric_v3(no_masked, eta', 1, 1);
Tpos_o = (Ts_o + Tp_o)/2;
Tpos_i = (Ts_i + Tp_i)/2;
TpoTpi = Tpos_o.*Tpos_i;

rho = mean(diffuse_masked.*psi_masked)./mean(TpoTpi.*ni_masked);
if ~isreal(rho)
    continue;
end

end
warning('on','all');

local_m1(1,point_idx_inbatch) = m1;
local_m2(1,point_idx_inbatch) = m2;
local_ks1(:,point_idx_inbatch) = ks1;
local_ks2(:,point_idx_inbatch) = ks2;
local_eta(1,point_idx_inbatch) = eta;
local_rho(:,point_idx_inbatch) = rho;


end
meshCurrent.m1(:,batch_offset+1:batch_offset+now_batch_size) = local_m1(:,1:now_batch_size);
meshCurrent.m2(:,batch_offset+1:batch_offset+now_batch_size) = local_m2(:,1:now_batch_size);
meshCurrent.eta(:,batch_offset+1:batch_offset+now_batch_size) = local_eta(:,1:now_batch_size);
meshCurrent.ks1(:,batch_offset+1:batch_offset+now_batch_size) = local_ks1(:,1:now_batch_size);
meshCurrent.ks2(:,batch_offset+1:batch_offset+now_batch_size) = local_ks2(:,1:now_batch_size);
meshCurrent.rho(:,batch_offset+1:batch_offset+now_batch_size) = local_rho(:,1:now_batch_size);
eta_iter(:,batch_offset+1:batch_offset+now_batch_size) = local_eta_iter(:,1:now_batch_size);
specular_iter(:,batch_offset+1:batch_offset+now_batch_size) = local_specular_iter(:,1:now_batch_size);
end


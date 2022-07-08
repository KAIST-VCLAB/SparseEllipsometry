%% parameters

norm_exp = 1;
batch_size = 5000;
intensity_threshold = 0.001;
color_threshold = 0.2;
color_threshold_sq = color_threshold.^2;

FiniteDifferenceStepSize = sqrt(eps);
MaxFunctionEvaluations = 3000;
opt_max_iter = 1000;
InitBarrierParam = 0.1; 
    

optAlgorithm = 'sqp';
options = optimoptions('fmincon','Display','off',...
    'MaxIterations',opt_max_iter, ...
    'FiniteDifferenceStepSize',FiniteDifferenceStepSize, ...
    'MaxFunctionEvaluations',MaxFunctionEvaluations, ...
    'InitBarrierParam',InitBarrierParam,...
    'Algorithm',optAlgorithm);


meshCurrent.sample_mask = sample_mask;
meshCurrent.fval = zeros(1,N);

%% Initialization
initial_ref_idx = 1.5;
initial_m1 = 0.5;
initial_m2 = 1;
initial_ks1 = ones(1,1);
initial_ks2 = ones(3,1);
initial_rho = ones(3,1);

diffuse_I = 2 .* I_90;
specular_I = I_0 - I_90;
aolp_I = I_135-I_45;

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
local_new_normal = zeros(3,batch_size);
local_fval = zeros(1,batch_size);

y_v = zeros(3,J);
for i = 1:J
    if ~isempty(extrinsic{i})
        y_v(:,i) = - extrinsic{i}(2,1:3)';
    end
end



%% 
time_record = zeros(6,ceil(N/batch_size));
for batch_idx = 1:ceil(N/batch_size)
batch_offset =  (batch_idx-1)*batch_size;
time = clock;
fprintf("%d iter: %02d:%02d:%02.2f\n",batch_offset,time(4),time(5),time(6));
time_record(:,batch_idx) = time;
now_batch_size = min(batch_size,N-batch_offset);

local_V = V(batch_offset+1:batch_offset+now_batch_size,:);
local_V(:,~sample_mask(:)) = 0;
local_diffuse_I = diffuse_I(batch_offset+1:batch_offset+now_batch_size,:);
local_specular_I = specular_I(batch_offset+1:batch_offset+now_batch_size,:);
local_aolp_I = aolp_I(batch_offset+1:batch_offset+now_batch_size,:);
local_vertices = meshCurrent.vertices(:,batch_offset+1:batch_offset+now_batch_size);
local_normals = meshCurrent.normals(:,batch_offset+1:batch_offset+now_batch_size);
local_psi = Psi(batch_offset+1:batch_offset+now_batch_size,:);

local_m1 = meshCurrent.m1(:,batch_offset+1:batch_offset+now_batch_size);
local_m2 = meshCurrent.m2(:,batch_offset+1:batch_offset+now_batch_size);
local_ks1 = meshCurrent.ks1(:,batch_offset+1:batch_offset+now_batch_size);
local_ks2 = meshCurrent.ks2(:,batch_offset+1:batch_offset+now_batch_size);
local_eta = meshCurrent.eta(:,batch_offset+1:batch_offset+now_batch_size);
local_rho = meshCurrent.rho(:,batch_offset+1:batch_offset+now_batch_size);
parfor (point_idx_inbatch = 1:now_batch_size)
point_idx = batch_offset+point_idx_inbatch;
%% DoP calculation

visible = local_V(point_idx_inbatch,:);
if sum(visible)<12
    continue
end
diffuse = local_diffuse_I(point_idx_inbatch,[visible,visible,visible]);
specular = local_specular_I(point_idx_inbatch,[visible,visible,visible]);
aolp = local_aolp_I(point_idx_inbatch,[visible,visible,visible]);
psi = local_psi(point_idx_inbatch,visible)';

m1 = local_m1(1,point_idx_inbatch);
m2 = local_m2(1,point_idx_inbatch);
ks1 = local_ks1(:,point_idx_inbatch);
ks2 = local_ks2(:,point_idx_inbatch);
eta = local_eta(1,point_idx_inbatch);
rho = local_rho(:,point_idx_inbatch);

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

diffuse_masked = diffuse(exposure_mask,:);
specular_masked = specular(exposure_mask,:);
aolp_masked = aolp(exposure_mask,:);
psi_masked = psi(exposure_mask);

diffuse_mono = sum(norm_mean_color.*diffuse_masked,2);
specular_mono = sum(norm_mean_color.*specular_masked,2);
aolp_mono = sum(norm_mean_color.*aolp_masked,2);
aolp_beta_mono = specular_mono;

%% Position and normal

pPos = local_vertices(:,point_idx_inbatch);
normal = local_normals(:,point_idx_inbatch);
vVec = vPos(:,visible) - pPos;
vVec = unit_vector(vVec, 'column');
lVec = lPos(:,visible) - pPos;
lVec = unit_vector(lVec, 'column');
hVec = unit_vector(lVec+vVec, 'column');
hi = sum(hVec.*lVec,1);

y_v_visible = y_v(:,visible);

%% iterations


warning('off','all');

%% geometric info

ni = sum(normal.*lVec,1);
nh = sum(normal.*hVec,1);
no = sum(normal.*vVec,1);
 
%%
if eta>1
    eta = initial_ref_idx;
end
if m1<0
    m1 = initial_m1;
end
if ks1<0
    ks1 = initial_ks1;
end
if any(ks2<0)
    ks2 = initial_ks2;
end

%%
x0 = [double(normal'),m1,m2,eta,ks1];
[x_est,fval,exitflag,output] = ...
fmincon(@(x)obj_parameters_and_normal_azimuth_wo_linear(x,double(diffuse_masked'),double(specular_masked'),double(aolp_masked'), ...
double(lVec(:,exposure_mask)),double(vVec(:,exposure_mask)),double(hVec(:,exposure_mask)),double(hi(:,exposure_mask)),y_v_visible(:,exposure_mask),double(psi_masked'),double(norm_mean_color')), ...
x0,[zeros(1,3) 1 -1 zeros(1,2)],[0],[],[],[-1 -1 -1 0 0 1.1 0]',[1 1 1 1 1 Inf Inf]',@(x)nonlcon_normal(x),options);

% x0 = [double(normal'),m1,eta,ks1];
% [x_est,fval,exitflag,output] = ...
% fmincon(@(x)obj_parameters_and_normal_m2fix(x,double(diffuse_masked'),double(specular_masked'),double(aolp_masked'), ...
% double(lVec(:,exposure_mask)),double(vVec(:,exposure_mask)),double(hVec(:,exposure_mask)),double(hi(:,exposure_mask)),y_v_visible(:,exposure_mask),double(psi_masked'),double(norm_mean_color')), ...
% x0,[],[],[],[],[-1 -1 -1 0 1.1 0]',[1 1 1 1 3 Inf]',@(x)nonlcon_init_normal(x,double(normal')),options);


new_normal = x_est(1:3)';
new_normal = new_normal./sqrt(sum(new_normal.^2));
m1 = x_est(4);
m2 = x_est(5);
eta = x_est(6);
ks1 = x_est(7);
ni = sum(new_normal.*lVec,1);
nh = sum(new_normal.*hVec,1);
no = sum(new_normal.*vVec,1);
theta_h = acos(nh);
theta_i = acos(ni);
theta_o = acos(no);
D1 = compute_D_GGX_a(m1,theta_h(:,exposure_mask));
D2 = compute_D_GGX_a(m2,theta_h(:,exposure_mask));
G1 = compute_G_smith_a(m1, theta_i(:,exposure_mask), theta_o(:,exposure_mask));
G2 = compute_G_smith_a(m2, theta_i(:,exposure_mask), theta_o(:,exposure_mask));
DG1= ks1.*(D1.*G1);

[Rs_i,Rp_i,~,~] = compute_Fresnel_coefficients_dielectric_v3(hi, 1, eta, 0);
Rpos = (Rs_i + Rp_i)/2;


ks2= ks2_lsq(D2,G2,DG1,Rpos(:,exposure_mask),theta_o(:,exposure_mask),psi_masked',1,specular_masked');



warning('on','all');
local_specular_iter(:,point_idx_inbatch) = output.iterations;
%% initial albedo
[~,~,Ts_i,Tp_i] = compute_Fresnel_coefficients_dielectric_v3(ni, 1, eta', 0);
[~,~,Ts_o,Tp_o] = compute_Fresnel_coefficients_dielectric_v3(no, eta', 1, 1);
Tpos_o = (Ts_o + Tp_o)/2;
Tpos_i = (Ts_i + Tp_i)/2;
TpoTpi = Tpos_o.*Tpos_i;

rho = double(mean(diffuse.*psi)./mean(TpoTpi.*ni))';
%%


local_m1(1,point_idx_inbatch) = m1;
local_m2(1,point_idx_inbatch) = m2;
local_ks1(:,point_idx_inbatch) = ks1;
local_ks2(:,point_idx_inbatch) = ks2;
local_eta(1,point_idx_inbatch) = eta;
local_rho(:,point_idx_inbatch) = rho;
local_new_normal(:,point_idx_inbatch) = new_normal;

local_fval(:,point_idx_inbatch) = fval;
end
meshCurrent.m1(:,batch_offset+1:batch_offset+now_batch_size) = local_m1(:,1:now_batch_size);
meshCurrent.m2(:,batch_offset+1:batch_offset+now_batch_size) = local_m2(:,1:now_batch_size);
meshCurrent.eta(:,batch_offset+1:batch_offset+now_batch_size) = local_eta(:,1:now_batch_size);
meshCurrent.ks1(:,batch_offset+1:batch_offset+now_batch_size) = local_ks1(:,1:now_batch_size);
meshCurrent.ks2(:,batch_offset+1:batch_offset+now_batch_size) = local_ks2(:,1:now_batch_size);
meshCurrent.rho(:,batch_offset+1:batch_offset+now_batch_size) = local_rho(:,1:now_batch_size);
meshCurrent.new_normal(:,batch_offset+1:batch_offset+now_batch_size) = local_new_normal(:,1:now_batch_size);
eta_iter(:,batch_offset+1:batch_offset+now_batch_size) = local_eta_iter(:,1:now_batch_size);
specular_iter(:,batch_offset+1:batch_offset+now_batch_size) = local_specular_iter(:,1:now_batch_size);
meshCurrent.fval(:,batch_offset+1:batch_offset+now_batch_size) = local_fval(:,1:now_batch_size);
end


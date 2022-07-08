%%
ni_mat = zeros(N,J);
hi_mat = zeros(N,J);
nh_mat = zeros(N,J);
no_mat = zeros(N,J);
parfor point_idx = 1:N
        pPos = meshCurrent.vertices(:,point_idx);
        normal = meshCurrent.normals(:,point_idx);
%         normal = meshCurrent.new_normal(:,point_idx);
        vVec = vPos - pPos;
        lVec = lPos - pPos;
        vVec = unit_vector(vVec, 'column');
        lVec = unit_vector(lVec, 'column');
        hVec = unit_vector(lVec+vVec, 'column');
        ni_mat(point_idx,:) = sum(normal.*lVec,1);
        hi_mat(point_idx,:) = sum(hVec.*lVec,1);
        nh_mat(point_idx,:) = sum(normal.*hVec,1);
        no_mat(point_idx,:) = sum(normal.*vVec,1);
end

%%
angle_weight = 1;
hist_interval = 0.5;
histogram_angles = 0:hist_interval:90-hist_interval;
histogram_angles = histogram_angles./180*pi;
hist_interval = hist_interval./180*pi;
hist_numer = zeros(num_cluster,numel(histogram_angles),3);
hist_denom = zeros(num_cluster,numel(histogram_angles));
hist_virtual = zeros(num_cluster,numel(histogram_angles),3);
hist_denom_virtual = zeros(num_cluster,numel(histogram_angles));
result_m1 = zeros(num_cluster,1);
result_ks1 = zeros(num_cluster,1);
result_m2 = zeros(num_cluster,1);
result_ks2 = zeros(num_cluster,3);

FiniteDifferenceStepSize = sqrt(eps);
MaxFunctionEvaluations = 3000;
optAlgorithm = 'sqp';
options = optimoptions('fmincon','Display','off',...
    'FiniteDifferenceStepSize',FiniteDifferenceStepSize, ...
    'MaxFunctionEvaluations',MaxFunctionEvaluations, ...
    'Algorithm',optAlgorithm,...
    'UseParallel',true);

virtual_psi = mean(Psi(V));

for k = 1:num_cluster
    cluster_filter = idx==k;
    if sum(cluster_filter)<100
        continue
    end
    visible_cluster = V_outlier(cluster_filter,:);
    diffuse_cluster = diffuse_I(cluster_filter,:);
    diffuse_cluster = reshape(diffuse_cluster,[size(visible_cluster),3]);
    diffuse_cluster = reshape(diffuse_cluster,[],3);
    specular_cluster = specular_I(cluster_filter,:);
    specular_cluster = reshape(specular_cluster,[size(visible_cluster),3]);
    specular_cluster = reshape(specular_cluster,[],3);
    aolp_cluster = aolp_I(cluster_filter,:);
    aolp_cluster = reshape(aolp_cluster,[size(visible_cluster),3]);
    aolp_cluster = reshape(aolp_cluster,[],3);
    zenith_cluster = zenith_mat(cluster_filter,:);
    azimuth_cluster = azimuth_mat(cluster_filter,:);
    diffuse = diffuse_cluster(visible_cluster(:),:);
    specular = specular_cluster(visible_cluster(:),:);
    aolp = aolp_cluster(visible_cluster(:),:);
    zenith = zenith_cluster(visible_cluster(:));
    azimuth = azimuth_cluster(visible_cluster(:));
    intensity_threshold = 0.01;
    beta_mask = diffuse(:,1)>intensity_threshold | diffuse(:,2)>intensity_threshold | diffuse(:,3)>intensity_threshold;
    diffuse_masked = diffuse(beta_mask,:);
    specular_masked = specular(beta_mask,:);
    aolp_masked = aolp(beta_mask,:);
    zenith_masked = zenith(beta_mask);
    azimuth_masked = azimuth(beta_mask);
    dop = sqrt(aolp_masked.^2+specular_masked.^2)./diffuse_masked;
    
    
    
    sample_size = 2.^12;
    random_sample1 = randperm(numel(zenith_masked));
    random_sample1 = random_sample1<=sample_size;
    
    diffuse_sample1 = diffuse_masked(random_sample1,:);
    specular_sample1 = specular_masked(random_sample1,:);
    aolp_sample1 = aolp_masked(random_sample1,:);
    zenith_sample1 = zenith_masked(random_sample1);
    azimuth_sample1 = azimuth_masked(random_sample1);
    dop_sample1 = dop(random_sample1,:);
    
    %%
    color_norm = sqrt(sum(diffuse.^2,2));
    normalized_color = diffuse./color_norm;
    mean_color = mean(normalized_color,1,'omitnan');
    mean_color = mean_color./sqrt(sum(mean_color.^2));
    norm_exp = 1;
    norm_mean_color = mean_color.^norm_exp;
    norm_mean_color = norm_mean_color./sum(mean_color);
    norm_mean_color = norm_mean_color.^(1/norm_exp);
    diffuse_mono = sum(norm_mean_color.*diffuse_masked,2);
    specular_mono = sum(norm_mean_color.*specular_masked,2);
    aolp_mono = sum(norm_mean_color.*aolp_masked,2);
    dop_mono = sqrt(aolp_mono.^2+specular_mono.^2)./diffuse_mono;
    diffuse_mono_sample1 = diffuse_mono(random_sample1,:);
    dop_mono_sample1 = dop_mono(random_sample1,:);
    aolp_mono_sample1 = aolp_mono(random_sample1,:);
    specular_mono_sample1 = specular_mono(random_sample1,:);
    %%
    
    
    x0 = 1.5;
    [x_est,fval,exitflag,output,grad,hessian] = ...
        fmincon(@(x)obj_function_eta_dop_zenith(x,double(dop_mono_sample1),double(zenith_masked(random_sample1))),x0,[],[],[],[],[1],[3],[],options);
    
    eta = x_est;

    
    %%
    
    sample_size = 2^16;
    random_sample2 = randperm(numel(zenith_masked));
    random_sample2 = random_sample2<=sample_size;
    
    diffuse_sample2 = diffuse_masked(random_sample2,:);
    specular_sample2 = specular_masked(random_sample2,:);
    aolp_sample2 = aolp_masked(random_sample2,:);
    zenith_sample2 = zenith_masked(random_sample2);
    azimuth_sample2 = azimuth_masked(random_sample2);
    dop_sample2 = dop(random_sample2,:);
    
    
    

    %% Specular reflection estimation    
    ni_cluster = ni_mat(cluster_filter,:);
    hi_cluster = hi_mat(cluster_filter,:);
    nh_cluster = nh_mat(cluster_filter,:);
    no_cluster = no_mat(cluster_filter,:);
    ni = ni_cluster(visible_cluster(:));
    hi = hi_cluster(visible_cluster(:));
    nh = nh_cluster(visible_cluster(:));
    no = no_cluster(visible_cluster(:));
    ni_masked = ni(beta_mask)';
    hi_masked = hi(beta_mask)';
    nh_masked = nh(beta_mask)';
    no_masked = no(beta_mask)';
    ni_sample1 = ni_masked(:,random_sample1);
    hi_sample1 = hi_masked(:,random_sample1);
    nh_sample1 = nh_masked(:,random_sample1);
    no_sample1 = no_masked(:,random_sample1);
    ni_sample2 = ni_masked(:,random_sample2);
    hi_sample2 = hi_masked(:,random_sample2);
    nh_sample2 = nh_masked(:,random_sample2);
    no_sample2 = no_masked(:,random_sample2);
%%
    theta_h1 = acos(nh_sample1);
    theta_i1 = acos(ni_sample1);
    theta_o1 = acos(no_sample1);
    theta_h2 = acos(nh_sample2);
    theta_i2 = acos(ni_sample2);
    theta_o2 = acos(no_sample2);
    [Rs_i,Rp_i,~,~] = compute_Fresnel_coefficients_dielectric_v3(hi_sample1, 1, eta, 0);
    Rpos1 = (Rs_i + Rp_i)/2;
    [Rs_i,Rp_i,~,~] = compute_Fresnel_coefficients_dielectric_v3(hi_sample2, 1, eta, 0);
    Rpos2 = (Rs_i + Rp_i)/2;
    x0 = [0.5,1,1,1,1,1];
    fit_angles_1 = (0.01:0.01:1-0.01);
    fit_angles = fit_angles_1.*pi./2;
    [Rs_i,Rp_i,~,~] = compute_Fresnel_coefficients_dielectric_v3(repmat(hi_sample2(1),size(fit_angles)), 1, eta, 0);
    Rpos_virtual = (Rs_i + Rp_i)/2;
    
    [x_est,fval,exitflag,output,grad,hessian] = ...
        fmincon(@(x)obj_function_two_specular_ed(x,double(theta_h2),double(Rpos_virtual),double(specular_sample2'), angle_weight),x0,[-eye(6);eye(2),zeros(2,4);1 -1 0 0 0 0],[zeros(6,1);1;1;0],[],[],[],[],[],options);
    
    m1 = x_est(1);
    m2 = x_est(2);
    ks1 = x_est(3);
    ks2 = x_est(4:6)';
    
    D1 = compute_D_GGX_a(m1,theta_h1);
    G1 = compute_G_smith_a(m1, theta_i1, theta_o1);
    D2 = compute_D_GGX_a(m2,theta_h1);
    G2 = compute_G_smith_a(m2, theta_i1, theta_o1);
    DG1= ks1 .* (D1.*G1)+ ks2 .* (D2.*G2);
    est_S1 = DG1.*Rpos1./4./cos(theta_o1);
    D1 = compute_D_GGX_a(m1,theta_h2);
    G1 = compute_G_smith_a(m1, theta_i2, theta_o2);
    D2 = compute_D_GGX_a(m2,theta_h2);
    G2 = compute_G_smith_a(m2, theta_i2, theta_o2);
    DG2= ks1 .* (D1.*G1)+ ks2 .* (D2.*G2);
    est_S2 = DG2.*Rpos2./4./cos(theta_o2);
    
    
    
    predict_D1 = compute_D_GGX_a(m1,fit_angles);
    predict_G1 = compute_G_smith_a(m1, fit_angles, fit_angles);
    predict_D2 = compute_D_GGX_a(m2,fit_angles);
    predict_G2 = compute_G_smith_a(m2, fit_angles, fit_angles);
    predict_DG = ks1 .* (predict_D1.*predict_G1)+ ks2 .* (predict_D2.*predict_G2);
    [Rs_i,Rp_i,~,~] = compute_Fresnel_coefficients_dielectric_v3(repmat(hi_sample2(1),size(fit_angles)), 1, eta, 0);
    
    predict_Rpos = (Rs_i + Rp_i)/2;
    
    predict_S = predict_DG.*predict_Rpos./4./cos(fit_angles);
    
    result_m1(k,:) = m1;
    result_ks1(k,:) = ks1;
    result_m2(k,:) = m2;
    result_ks2(k,:) = ks2;
    
    
    
    %% Angle difference
    obs_azimuth = atan2(aolp_mono_sample1,specular_mono_sample1)./2;
    azimuth_diff = mod(obs_azimuth - azimuth_sample1 ,pi) -pi/2;
    
    
    est_S1_mono = sum(norm_mean_color.*est_S1',2);
    aolp_beta_mono_sample1 = specular_mono_sample1-est_S1_mono;
    obs_azimuth = atan2(aolp_mono_sample1,aolp_beta_mono_sample1)./2;
    azimuth_diff = mod(obs_azimuth - azimuth_sample1 ,pi) -pi/2;
    
    %% specular removal dop
    aolp_beta_sample1 = specular_sample1-est_S1';
    dop_specular_removed = sqrt(aolp_sample1.^2+aolp_beta_sample1.^2)./diffuse_mono_sample1;
    dop_mono_specular_removed = sqrt(aolp_mono_sample1.^2+aolp_beta_mono_sample1.^2)./diffuse_mono_sample1;
    
    options = optimoptions('fmincon','Display','off');
    
    x0 = 1.5;
    [x_est,fval,exitflag,output,grad,hessian] = ...
        fmincon(@(x)obj_function_eta_dop_zenith(x,double([dop_mono_specular_removed]),double(zenith_sample1)),x0,[],[],[],[],[1],[3],[],options);
    eta_specular_removed = x_est;
    
    %% histogram
    theta_h = acos(nh_masked);
    for hist_iter = 1:numel(histogram_angles)
        hist_mask = theta_h>=histogram_angles(hist_iter) & theta_h<histogram_angles(hist_iter)+hist_interval;
        hist_numer(k,hist_iter,:) = sum(specular_masked(hist_mask,:),1);
        hist_denom(k,hist_iter) = sum(hist_mask);
    end
    hist_mono = squeeze(mean(hist_numer(k,:,:),3));
    hist_mono = hist_mono./squeeze(hist_denom(k,:));
    hist_mono(hist_denom(k,:)==0) = 0;

    
    predict_D1 = compute_D_GGX_a(m1,histogram_angles);
    predict_G1 = compute_G_smith_a(m1, histogram_angles, histogram_angles);
    predict_D2 = compute_D_GGX_a(m2,histogram_angles);
    predict_G2 = compute_G_smith_a(m2, histogram_angles, histogram_angles);
    predict_DG = ks1 .* (predict_D1.*predict_G1)+ ks2 .* (predict_D2.*predict_G2);
    [Rs_i,Rp_i,~,~] = compute_Fresnel_coefficients_dielectric_v3(repmat(hi_sample2(1),size(histogram_angles)), 1, eta, 0);
    predict_Rpos = (Rs_i + Rp_i)/2;
    
    hist_virtual(k,:,:) = (predict_DG.*predict_Rpos./4./cos(histogram_angles))';
    
end

clear diffuse_cluster specular_cluster aolp_cluster ...
    zenith_cluster azimuth_cluster diffuse specular aolp zenith azimuth ...
    diffuse_masked specular_masked  aolp_masked zenith_masked azimuth_masked dop normalized_color psi psi_masked

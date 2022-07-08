%%
colors = rand(num_cluster,3);
colors = colors-0.5;
colors = colors./max(abs(colors),2)./2;
colors = colors+0.5;

input_rho = meshCurrent.rho(:,region_mask)';
input_rho(isnan(input_rho)) = 0;
% input_rho_norm = sqrt(sum(input_rho.^2,2));
% input_rho = input_rho./input_rho_norm;
% input_rho(input_rho_norm==0,:) = 0;

rho_threshold = max(median(input_rho,1))*10;
input_rho(input_rho>rho_threshold) = rho_threshold;
[coeff,score,latent] = pca(input_rho);
normalized_rho = input_rho./sqrt(latent(1));
input_eta = meshCurrent.eta(:,region_mask)';
input_eta(input_eta>3) = 3;
eta_weight = 2;
rho_eta = [normalized_rho,input_eta.*eta_weight];

[idx_masked,C,sumd,distance_cluster_masked] = kmeans(rho_eta,num_cluster);
idx = zeros(size(meshCurrent.rho,2),1);
idx(region_mask) = idx_masked;
distance_cluster = zeros(size(meshCurrent.rho,2),num_cluster);
distance_cluster(region_mask,:) = distance_cluster_masked;



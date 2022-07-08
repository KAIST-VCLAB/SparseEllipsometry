clear;
dir_root = './owl';

iter = 9;

%% load info.
fn_data = sprintf('%s/output_pbrdf/iter%.4d.mat', dir_root, iter);
load(fn_data);
%% geometry optimization

if iter~=0
    fn_prev = sprintf('%s/output_pbrdf/iter%.4d_meshCurrent.mat', dir_root, iter-1);
    prev = load(fn_prev,'meshCurrent');
    meshCurrent_old = prev.meshCurrent;
    m_load_old_mesh_info;
else
    initial_ref_idx = 1.5;
    initial_m1 = 0.5;
    initial_m2 = 1;
    initial_ks1 = ones(1,1);
    initial_ks2 = ones(3,1);
    initial_rho = ones(3,1);
    meshCurrent.eta = repmat(initial_ref_idx,1,N);
    meshCurrent.m1 = repmat(initial_m1,1,N);
    meshCurrent.m2 = repmat(initial_m2,1,N);
    meshCurrent.ks1 = repmat(initial_ks1,1,N);
    meshCurrent.ks2 = repmat(initial_ks2,1,N);
    meshCurrent.rho = repmat(initial_rho,1,N);
    meshCurrent.new_normal = meshCurrent.normals;
end
region_mask = ones(N,1);
m_opt_parameters_and_normal_step_wo_linear;

fn_mesh = sprintf('%s/output_pbrdf/iter%.4d_meshCurrent.mat', dir_root, iter);
save(fn_mesh,'meshCurrent');
%% specular augmentation and final pBRDF
region_mask = ones(N,1,'logical');
main_clustering;
m_outlier_removal;
main_clustered_specular_lobes_angle_error;
m_estimate_brdf_clustering;
save(fullfile(dir_root,'output_pbrdf','meshCurrent_clustering.mat'),'meshCurrent');


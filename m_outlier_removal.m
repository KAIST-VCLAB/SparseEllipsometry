%% parameters

norm_exp = 1;
virtual_img_size = 500;
batch_size = 100000;


color_threshold = 0.2;
color_threshold_sq = color_threshold.^2;

intensity_threshold = 0.001;





% path_figs = fullfile(dir_root,sprintf("angle_weight_%.2f",100));

meshCurrent.sample_mask = sample_mask;

%% Initialization
V_outlier = V;
selected_exposure_indices = zeros(N,1);
meshCurrent.norm_mean_color = zeros(3,N);

local_selected_exposure_indices = zeros(batch_size,1);
local_norm_mean_color = zeros(3,batch_size);



%% 
time_record = zeros(6,ceil(N/batch_size));
for batch_idx = 1:ceil(N/batch_size)
% for batch_idx = 43
batch_offset =  (batch_idx-1)*batch_size;
time = clock;
fprintf("%d iter: %02d:%02d:%02.2f\n",batch_offset,time(4),time(5),time(6));
time_record(:,batch_idx) = time;
now_batch_size = min(batch_size,N-batch_offset);

local_V = V(batch_offset+1:batch_offset+now_batch_size,:);
local_V(:,~sample_mask(:)) = 0;
outlier_V = local_V;
local_diffuse_I = diffuse_I(batch_offset+1:batch_offset+now_batch_size,:);
local_specular_I = specular_I(batch_offset+1:batch_offset+now_batch_size,:); 
local_aolp_I = aolp_I(batch_offset+1:batch_offset+now_batch_size,:);
local_vertices = meshCurrent.vertices(:,batch_offset+1:batch_offset+now_batch_size);
local_normals = meshCurrent.normals(:,batch_offset+1:batch_offset+now_batch_size);
parfor (point_idx_inbatch = 1:now_batch_size)
% for point_idx_inbatch = 9
% for point_idx_inbatch = 22178
point_idx = batch_offset+point_idx_inbatch;
if ~region_mask(point_idx)
    continue
end
%% DoP calculation

visible = local_V(point_idx_inbatch,:);
if ~any(visible)
    local_selected_exposure_indices(point_idx_inbatch) = 0;
    continue
end

diffuse = local_diffuse_I(point_idx_inbatch,[visible,visible,visible]);
diffuse = reshape(squeeze(diffuse),[],3);
relative_exposure_vis = relative_exposure(visible);
captured_diffuse = relative_exposure_vis.*diffuse;

%% exposure selection

intensity_captured = mean(captured_diffuse,2);
intensity_mask = intensity_captured>intensity_threshold;
mean_intensity = zeros(size(relative_exposures));
for i = 1:numel(relative_exposures)
    candidates = intensity_captured(relative_exposure_vis==relative_exposures(i));
    mean_intensity(i) = mean(candidates);
end

underexposure_mask = mean_intensity>0.1;
overexposure_mask = mean_intensity<0.8;
exposure_mask = overexposure_mask&underexposure_mask;
num_candidate_exposure = sum(exposure_mask);
if num_candidate_exposure == 0
    if ~any(underexposure_mask)
        [sorted_exposure,sorted_exposure_idx] = sort(relative_exposures,'descend');
    elseif ~any(overexposure_mask)
        [sorted_exposure,sorted_exposure_idx] = sort(relative_exposures);
    else
        continue;
    end
    for i = 1:numel(relative_exposures)
        selected_exposure_idx = sorted_exposure_idx(i);
        selected_exposure = sorted_exposure(i);
        exposure_filter = relative_exposure_vis==selected_exposure;
        if sum(exposure_filter)>2
            break;
        end
    end
elseif num_candidate_exposure >1
    exposure_score = zeros(num_candidate_exposure,1);
    exposure_indices = 1:numel(relative_exposures);
    candidate_exposure_indices = exposure_indices(exposure_mask);
    candidate_exposure = relative_exposures(exposure_mask);
    for i = 1:numel(candidate_exposure)
        candidates = intensity_captured(relative_exposure_vis==candidate_exposure(i));
        exposure_score(i) = sum(candidates>0.1 & candidates<0.8);
    end
    [~,selected_exposure_idx] = max(exposure_score);
    selected_exposure_idx = candidate_exposure_indices(selected_exposure_idx);
    selected_exposure = relative_exposures(selected_exposure_idx);
else
    selected_exposure_idx = find(exposure_mask,1);
    selected_exposure = relative_exposures(selected_exposure_idx);
end

% [~,selected_exposure_idx]= min(abs(mean_intensity-0.5));

%% Albedo channel

exposure_filter = relative_exposure_vis==selected_exposure;
% if sum(exposure_filter)==0
%     continue;
% end
diffuse_filtered = diffuse(exposure_filter,:);

color_norm = sqrt(sum(diffuse_filtered.^2,2));
normalized_color = diffuse_filtered./color_norm;

mean_color = mean(normalized_color,1,'omitnan');
mean_color = mean_color./sqrt(sum(mean_color.^2));
color_distance = sum((normalized_color-mean_color).^2,2);
adaptive_threshold = color_threshold_sq;
color_mask = color_distance<adaptive_threshold;
while sum(color_mask)<1
    adaptive_threshold = adaptive_threshold*2;
    color_mask = color_distance<adaptive_threshold;
    if(adaptive_threshold>1)
        break
    end
end
for i = 1:3
    normalized_color_filtered = normalized_color(color_mask,:);
    mean_color = mean(normalized_color_filtered,1,'omitnan');
    mean_color = mean_color./sqrt(sum(mean_color.^2));
    color_distance = sum((normalized_color-mean_color).^2,2);
    color_mask = color_distance<adaptive_threshold;
    while sum(color_mask)<1
        adaptive_threshold = adaptive_threshold*2;
        color_mask = color_distance<adaptive_threshold;
        if(adaptive_threshold>1)
            break
        end
    end
end

norm_mean_color = mean_color.^norm_exp;
norm_mean_color = norm_mean_color./sum(mean_color);
norm_mean_color = norm_mean_color.^(1/norm_exp);

color_norm = sqrt(sum(diffuse.^2,2));
normalized_color = diffuse./color_norm;
color_distance = sum((normalized_color-mean_color).^2,2);
color_mask = color_distance<color_threshold_sq;

intensity_mask = intensity_mask & color_mask;
visible(visible) = intensity_mask;
outlier_V(point_idx_inbatch,:) = visible;
local_selected_exposure_indices(point_idx_inbatch) = selected_exposure_idx;
local_norm_mean_color(:,point_idx_inbatch) = norm_mean_color;

% figure;scatter(diffuse(:,1),diffuse(:,2),'.');axis([0,1,0,1]);
% figure;scatter(diffuse(:,1),diffuse(:,3),'.');axis([0,1,0,1]);
% figure;scatter(diffuse(:,2),diffuse(:,3),'.');axis([0,1,0,1]);
% figure;scatter3(diffuse(:,1),diffuse(:,2),diffuse(:,3),'.');axis([0,1,0,1,0,1]);
% figure;scatter3(diffuse_filtered(:,1),diffuse_filtered(:,2),diffuse_filtered(:,3),'.');axis([0,1,0,1,0,1]);

% figure;scatter(normalized_color(:,1),normalized_color(:,2),'.');axis([0,1,0,1]);
% figure;scatter(normalized_color(:,1),normalized_color(:,3),'.');axis([0,1,0,1]);
% figure;scatter(normalized_color(:,2),normalized_color(:,3),'.');axis([0,1,0,1]);
% figure;scatter3(normalized_color(:,1),normalized_color(:,2),normalized_color(:,3),'.');axis([0,1,0,1,0,1]);

end
V_outlier(batch_offset+1:batch_offset+now_batch_size,:)= outlier_V(1:now_batch_size,:);
selected_exposure_indices(batch_offset+1:batch_offset+now_batch_size)= local_selected_exposure_indices(1:now_batch_size);
meshCurrent.norm_mean_color(:,batch_offset+1:batch_offset+now_batch_size)= local_norm_mean_color(:,1:now_batch_size);
end


function error_corr = Correspondence_Error_function(patch_input,patch_as_target,corespondence_type)
%% correspondence error between two patches, the measure is a simple difference between intensities or the luminance
%  patch_input - input patch
%  patch_as_target - target patch
%  corespondence_type - type of Correspondence

%%
if strcmp(corespondence_type,'intensity')
patch_input = rgb2gray(patch_input);
patch_as_target = rgb2gray(patch_as_target);
error_corr = Root_Mean_Sq_Error(patch_input,patch_as_target);
elseif strcmp(corespondence_type,'luminance')
patch_input = rgb2hsv(patch_input);
patch_as_target = rgb2hsv(patch_as_target);
patch_input = patch_input(:,:,3);
patch_as_target = patch_as_target(:,:,3);
error_corr = Root_Mean_Sq_Error(patch_input,patch_as_target);
end
end
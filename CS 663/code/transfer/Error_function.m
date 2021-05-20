
function error_val = Error_function(input_patch,patch_as_ref,type,degree_of_overlap,size_patch)
%% error in the overlap region
%  input_patch - input patch
%  patch_as_ref - reference patch
%  type - type of overlap
%  degree_of_overlap - size of overlap
%  size_patch - size of patch
%%
if strcmp(type,'vertical')
patch_L = patch_as_ref{1};
overlap_L = patch_L(:,size_patch-degree_of_overlap+1:size_patch,:);
overlap_main = input_patch(:,1:degree_of_overlap,:);
error_val = Root_Mean_Sq_Error(overlap_L,overlap_main);
elseif strcmp(type,'horizontal')
patch_T = patch_as_ref{2};
overlap_T = patch_T(size_patch-degree_of_overlap+1:size_patch,:,:);
overlap_main = input_patch(1:degree_of_overlap,:,:);
error_val = Root_Mean_Sq_Error(overlap_T,overlap_main);
else
patch_L = patch_as_ref{1};
overlap_L = patch_L(:,size_patch-degree_of_overlap+1:size_patch,:);
patch_T = patch_as_ref{2};
overlap_T = patch_T(size_patch-degree_of_overlap+1:size_patch,:,:);
patch_corner = patch_as_ref{3};
overlap_corner = patch_corner(size_patch-degree_of_overlap+1:size_patch,size_patch-degree_of_overlap+1:size_patch,:);
overlap_T_main = input_patch(1:degree_of_overlap,:,:);
overlap_L_main = input_patch(:,1:degree_of_overlap,:);
overlap_corner_main = input_patch(1:degree_of_overlap,1:degree_of_overlap,:);
error_L = Root_Mean_Sq_Error(overlap_L,overlap_L_main);
error_T = Root_Mean_Sq_Error(overlap_T,overlap_T_main);
error_corner = Root_Mean_Sq_Error(overlap_corner,overlap_corner_main);
error_val = error_L + error_T - error_corner;
end
end
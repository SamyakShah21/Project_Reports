function patch_output = Transfer_Patch_Nearest(patch_as_ref,patch_as_target,img_of_texture,max_error,type,degree_of_overlap,size_patch,alpha_val,corespondence_type)
%% Returns the patch which is nearest to the target patch, according to a distance metric, in the original image
%  patch_as_ref - reference patch
%  patch_as_target - target patch, closest to which output patch should be
%  img_of_texture - orginal img of texture
%  max_error - tolerated error
%  type - type of overlap
%  degree_of_overlap - size of overlap
%  size_patch - size of patch
% corespondence_type - Type of Correspondence
% alpha_val - alpha value
%%
[H,W,~] = size(img_of_texture);
nr = H-size_patch+1;
nc = W-size_patch+1;
patch_error_val = zeros([nr,nc]); % matrix of correspondence error of patch no. (i,j) whose left top pixel is at (i,j)
error_min_val = 22222222.0;
for i=1:H-size_patch+1
for j=1:W-size_patch+1
patch_main = img_of_texture(i:i+size_patch-1,j:j+size_patch-1,:); %% finding patch whose left top pixel is at (i,j)
% overlap error of patch no. (i,j) whose left top pixel is at (i,j)
error_overlap = Error_function(patch_main,patch_as_ref,type,degree_of_overlap,size_patch);
% correspondence error of patch no. (i,j) whose left top pixel is at (i,j)
error_corr = Correspondence_Error_function(patch_main,patch_as_target,corespondence_type);
% Total error of patch no. (i,j) whose left top pixel is at (i,j)
error_tot = alpha_val*error_overlap + (1-alpha_val)*error_corr;
patch_error_val(i,j) = error_tot;
if error_tot > 0  
error_min_val = min(error_tot,error_min_val);
end			
end
end
error_min_val = error_min_val*(1+max_error);
[i_patch, j_patch] = find(patch_error_val<error_min_val);
x = randi(length(i_patch),1);
i_min = i_patch(x);
j_min = j_patch(x);
patch_output = img_of_texture(i_min:i_min+size_patch-1, j_min:j_min+size_patch-1, :);
end
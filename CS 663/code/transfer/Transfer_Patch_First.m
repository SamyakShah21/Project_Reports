function patch_output = Transfer_Patch_First(img_of_texture,patch_as_target,size_patch,max_error,corespondence_type)
%% Returns the random first patch
%  patch_as_target - target patch, closest to which output patch should be
%  img_of_texture - orginal img of texture
%  max_error - tolerated error
%  size_patch - size of patch
% corespondence_type - Type of Correspondence

%%
[H,W,~] = size(img_of_texture);
nr = H-size_patch+1;
nc = W-size_patch+1;
patch_error_val = zeros([nr,nc]);
error_min_val = 22222222.0;
for i=1:H-size_patch+1
for j=1:W-size_patch+1
patch_main = img_of_texture(i:i+size_patch-1,j:j+size_patch-1,:);
error_corr = 	Correspondence_Error_function(patch_main,patch_as_target,corespondence_type);
error_tot = error_corr;
patch_error_val(i,j) = error_tot;
if error_tot > 0
error_min_val = min(error_tot,error_min_val);
end			
end
end
error_min_val = error_min_val*(1+max_error);
[i_patch, j_patch] = find(patch_error_val<=error_min_val);
x = randi(length(i_patch),1);
i_min = i_patch(x);
j_min = j_patch(x);
patch_output = img_of_texture(i_min:i_min+size_patch-1, j_min:j_min+size_patch-1, :);
end
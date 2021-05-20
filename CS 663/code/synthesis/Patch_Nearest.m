function output = Patch_Nearest(patch_as_ref,img_main,max_error,type,degree_of_overlap,size_patch)
%% Returns the patch which is nearest to the input patch, according to a distance metric
%  patch_as_ref - reference patch
%  img_main - orginal img
%  max_error - tolerated error
%  type - type of overlap
%  degree_of_overlap - size of overlap
%  size_patch - size of patch
%%
[height,width,~] = size(img_main);
nr = height-size_patch+1;
nc = width-size_patch+1;
error_val_min = 22222222.0;
error = zeros([nr,nc]);  % matrix of error of patch no. (i,j) whose left top pixel is at (i,j)
for i=1:height-size_patch+1
for j=1:width-size_patch+1
temp_patch = img_main(i:i+size_patch-1,j:j+size_patch-1,:);  % finding patch whose left top pixel is at (i,j)
temp = Error_function(temp_patch,patch_as_ref,type,degree_of_overlap,size_patch);
error(i,j) = temp;
if temp>0
error_val_min = min(temp,error_val_min);
end
end			
end
error_val_min = error_val_min*(1+max_error);
[i_patch, j_patch] = find(error<error_val_min);
x = randi(length(i_patch),1);
i_min = i_patch(x);
j_min = j_patch(x);
output = img_main(i_min:i_min+size_patch-1, j_min:j_min+size_patch-1, :);
end
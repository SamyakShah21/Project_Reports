function output_img = Texture_Transfer(img_target,img_texture)
% img_target - Target Image
% img_texture - Texture Image
%%
[H,W,num_chan] = size(img_target);
%% Defining the parameters of our algorithm
size_patch = 20;
degree_of_overlap = size_patch/4;
size_patch_final = size_patch-degree_of_overlap;
max_error = 0.1;
alpha_val = 0.2;
num_iter = 1;
correspondence_type = 'intensity';
%% Calculating the new generated image size 
new_H = size_patch_final*floor(H/size_patch_final) + degree_of_overlap;
new_W = size_patch_final*floor(W/size_patch_final) + degree_of_overlap;
img_target = imresize(img_target,[new_H new_W], 'bicubic');
img_target_copy = img_target;
output_img = zeros([new_H,new_W,num_chan]);
f = waitbar(0,"Transferring Texture");
max_i = (new_H-degree_of_overlap)/size_patch_final;
max_j = (new_W-degree_of_overlap)/size_patch_final;
%% Calculating the new generated image size 
new_H = size_patch_final*floor(H/size_patch_final) + degree_of_overlap;
new_W = size_patch_final*floor(W/size_patch_final) + degree_of_overlap;
img_target = imresize(img_target,[new_H new_W], 'bicubic');
output_img = zeros([new_H,new_W,num_chan]);
progress = 0;
max_i = (new_H-degree_of_overlap)/size_patch_final;
max_j = (new_W-degree_of_overlap)/size_patch_final;
for i = 1:max_i
for j = 1:max_j
if i==1 && j==1
patch_target = img_target(1:size_patch,1:size_patch,:);
output_img(1:size_patch,1:size_patch,:) = Transfer_Patch_First(img_texture,patch_target,size_patch,0.0,correspondence_type);
elseif i==1
index_begin = size_patch_final + (j-2)*size_patch_final;
patch_previous = output_img(1:size_patch,index_begin - size_patch_final + 1:index_begin - size_patch_final + size_patch,:);
patches_as_reference = cell(1,3);
patches_as_reference{1} = patch_previous;
patch_target = img_target(1:size_patch,index_begin+1:index_begin+size_patch,:);
patch_chosen = Transfer_Patch_Nearest(patches_as_reference, patch_target, img_texture, max_error, 'vertical', degree_of_overlap, size_patch, alpha_val, correspondence_type);
patch_final = Minimum_Errror_Boundary_cut(patches_as_reference,patch_chosen,degree_of_overlap,'vertical',size_patch);
output_img(1:size_patch,index_begin+1:index_begin+size_patch,:) = patch_final;
elseif j==1
index_begin = size_patch_final + (i-2)*size_patch_final;
patch_previous = output_img(index_begin - size_patch_final + 1:index_begin - size_patch_final + size_patch,1:size_patch,:);
patches_as_reference = cell(1,3);
patches_as_reference{2} = patch_previous;
patch_target = img_target(index_begin+1:index_begin+size_patch,1:size_patch,:);
patch_chosen = Transfer_Patch_Nearest(patches_as_reference, patch_target, img_texture, max_error, 'horizontal', degree_of_overlap, size_patch, alpha_val, correspondence_type);
patch_final = Minimum_Errror_Boundary_cut(patches_as_reference,patch_chosen,degree_of_overlap,'horizontal',size_patch);
output_img(index_begin+1:index_begin+size_patch,1:size_patch,:) = patch_final;
else
index_left = size_patch_final + (j-2)*size_patch_final;
index_top = size_patch_final + (i-2)*size_patch_final;
patch_L = output_img(index_top + 1 : index_top + size_patch,index_left - size_patch_final + 1:index_left - size_patch_final + size_patch,:);
patch_T = output_img(index_top - size_patch_final + 1:index_top - size_patch_final + size_patch,index_left + 1:index_left + size_patch,:);
patch_corner = output_img(index_top - size_patch_final + 1:index_top - size_patch_final + size_patch,index_left - size_patch_final + 1:index_left - size_patch_final + size_patch,:);
patches_as_reference = cell(1,3);
patches_as_reference{1} = patch_L;
patches_as_reference{2} = patch_T;
patches_as_reference{3} = patch_corner;
patch_target = img_target(index_top+1:index_top+size_patch,index_left+1:index_left+size_patch,:);
patch_chosen = Transfer_Patch_Nearest(patches_as_reference, patch_target, img_texture, max_error, 'both', degree_of_overlap, size_patch, alpha_val, correspondence_type);
patch_final = Minimum_Errror_Boundary_cut(patches_as_reference,patch_chosen,degree_of_overlap,'both',size_patch);
output_img(index_top+1:index_top+size_patch,index_left+1:index_left+size_patch,:) = patch_final;
end
progress = progress + 1;
waitbar(progress/(max_i*max_j),f,"Transferring Texture");
end
end
close(f);
end
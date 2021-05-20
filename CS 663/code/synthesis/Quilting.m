function img_output = Quilting(img_main,size_patch)
[H,W,num_chan] = size(img_main);
%% Required Params
degree_of_overlap = size_patch/6;
size_patch_final = size_patch-degree_of_overlap;
max_error = 0.1;
%% Generating new image 
new_H = 2*size_patch_final*floor(H/size_patch_final) + degree_of_overlap;
new_W = 2*size_patch_final*floor(W/size_patch_final) + degree_of_overlap;
img_output = zeros([new_H,new_W,num_chan]);
f = waitbar(0,"Quilting");
progress = 0;
max_i = (new_H-degree_of_overlap)/size_patch_final;
max_j = (new_W-degree_of_overlap)/size_patch_final;
for i = 1:max_i
for j = 1:max_j
if i==1 && j==1
img_output(1:size_patch,1:size_patch,:) = Patch_Random(img_main,size_patch);
elseif i==1
index_begin = size_patch_final + (j-2)*size_patch_final;
patch_previous = img_output(1:size_patch,index_begin - size_patch_final + 1:index_begin - size_patch_final + size_patch,:);
patches_as_reference = cell(1,3);
patches_as_reference{1} = patch_previous;
patch_chosen = Patch_Nearest(patches_as_reference, img_main, max_error, 'vertical', degree_of_overlap, size_patch);
patch_final = Minimum_Error_Boundary_cut(patches_as_reference,patch_chosen,degree_of_overlap,'vertical',size_patch);
img_output(1:size_patch,index_begin+1:index_begin+size_patch,:) = patch_final;
elseif j==1
index_begin = size_patch_final + (i-2)*size_patch_final;
patch_previous = img_output(index_begin - size_patch_final + 1:index_begin - size_patch_final + size_patch,1:size_patch,:);
patches_as_reference = cell(1,3);
patches_as_reference{2} = patch_previous;
patch_chosen = Patch_Nearest(patches_as_reference, img_main, max_error, 'horizontal', degree_of_overlap, size_patch);
patch_final = Minimum_Error_Boundary_cut(patches_as_reference,patch_chosen,degree_of_overlap,'horizontal',size_patch);
img_output(index_begin+1:index_begin+size_patch,1:size_patch,:) = patch_final;
else
index_left = size_patch_final + (j-2)*size_patch_final;
index_top = size_patch_final + (i-2)*size_patch_final;
patch_L = img_output(index_top + 1 : index_top + size_patch,index_left - size_patch_final + 1:index_left - size_patch_final + size_patch,:);
patch_T = img_output(index_top - size_patch_final + 1:index_top - size_patch_final + size_patch,index_left + 1:index_left + size_patch,:);
patch_corner = img_output(index_top - size_patch_final + 1:index_top - size_patch_final + size_patch,index_left - size_patch_final + 1:index_left - size_patch_final + size_patch,:);
patches_as_reference = cell(1,3);
patches_as_reference{1} = patch_L;
patches_as_reference{2} = patch_T;
patches_as_reference{3} = patch_corner;
patch_chosen = Patch_Nearest(patches_as_reference, img_main, max_error, 'both', degree_of_overlap, size_patch);
patch_final = Minimum_Error_Boundary_cut(patches_as_reference,patch_chosen,degree_of_overlap,'both',size_patch);
img_output(index_top+1:index_top+size_patch,index_left+1:index_left+size_patch,:) = patch_final;
end
progress = progress + 1;	
waitbar(progress/(max_i*max_j),f,"Quilting");
end
end
close(f);
end
function patch_output = Minimum_Error_Boundary_cut(patch_as_ref,patch_main,degree_of_overlap,type,size_patch)
%% Returns the final patched up section using the selected patch and neighbor patches
% patch_as_ref - reference patch
% patch_main - selected patch
% degree_of_overlap - size of overlap
% type - type of overlap
% size_patch - size of patch
%%
patch_output = zeros(size(patch_main));
if strcmp(type,'vertical')
patch_output(:,degree_of_overlap+1:size_patch,:) = patch_main(:,degree_of_overlap+1:size_patch,:);	
patch_L = patch_as_ref{1};
overlap_L = patch_L(:,size_patch-degree_of_overlap+1:size_patch,:);
overlap_sel = patch_main(:,1:degree_of_overlap,:);
patch_Diff = overlap_L-overlap_sel;
E_val = sum(abs(patch_Diff),3);
E_matrix = zeros(size(E_val));
E_matrix(1,:) = E_val(1,:);
for i = 2:size_patch
for j = 1:degree_of_overlap
if j==1
E_matrix(i,j) = E_val(i,j) + min([E_matrix(i-1,j), E_matrix(i-1,j+1)]);
elseif j==degree_of_overlap
E_matrix(i,j) = E_val(i,j) + min([E_matrix(i-1,j-1), E_matrix(i-1,j)]);
else
E_matrix(i,j) = E_val(i,j) + min([E_matrix(i-1,j-1), E_matrix(i-1,j), E_matrix(i-1,j+1)]);
end
end
end
[E_min_val, index_previous] = min(E_matrix(size_patch,:));
if index_previous==1
patch_output(size_patch,:,:) = patch_main(size_patch,:,:);
else
patch_output(size_patch,1:index_previous-1,:) = patch_L(size_patch,size_patch - degree_of_overlap + 1:size_patch - degree_of_overlap + index_previous-1,:);
patch_output(size_patch,index_previous:degree_of_overlap,:) = patch_main(size_patch,index_previous:degree_of_overlap,:);	
end
for i = size_patch-1:-1:1
[~, index_previous] = min(abs(E_matrix(i,:) - (E_min_val-E_val(i+1,index_previous))));
E_min_val = E_matrix(i,index_previous);
if index_previous==1
patch_output(i,:,:) = patch_main(i,:,:);
else
patch_output(i,1:index_previous-1,:) = patch_L(i,size_patch - degree_of_overlap + 1:size_patch - degree_of_overlap + index_previous-1,:);
patch_output(i,index_previous:degree_of_overlap,:) = patch_main(i,index_previous:degree_of_overlap,:);	
end
end
elseif strcmp(type,'horizontal')
patch_output(degree_of_overlap+1:size_patch,:,:) = patch_main(degree_of_overlap+1:size_patch,:,:);	
patch_T = patch_as_ref{2};
overlap_T = patch_T(size_patch-degree_of_overlap+1:size_patch,:,:);
overlap_sel = patch_main(1:degree_of_overlap,:,:);
patch_Diff = overlap_T-overlap_sel;
E_val = sum(patch_Diff.*patch_Diff,3);
E_matrix = zeros(size(E_val));
E_matrix(:,1) = E_val(:,1);
for j = 2:size_patch
for i = 1:degree_of_overlap
if i==1
E_matrix(i,j) = E_val(i,j) + min([E_matrix(i,j-1), E_matrix(i+1,j-1)]);
elseif i==degree_of_overlap
E_matrix(i,j) = E_val(i,j) + min([E_matrix(i-1,j-1), E_matrix(i,j-1)]);
else
E_matrix(i,j) = E_val(i,j) + min([E_matrix(i-1,j-1), E_matrix(i,j-1), E_matrix(i+1,j-1)]);
end
end
end
[E_min_val, index_previous] = min(E_matrix(:,size_patch));
if index_previous==1
patch_output(:,size_patch,:) = patch_main(:,size_patch,:);
else
patch_output(1:index_previous-1,size_patch,:) = patch_T(size_patch - degree_of_overlap + 1:size_patch - degree_of_overlap + index_previous-1,size_patch,:);
patch_output(index_previous:degree_of_overlap,size_patch,:) = patch_main(index_previous:degree_of_overlap,size_patch,:);
end
for i = size_patch-1:-1:1
[~, index_previous] = min(abs(E_matrix(:,i) - (E_min_val-E_val(index_previous,i+1))));
E_min_val = E_matrix(index_previous,i);
if index_previous==1
patch_output(:,i,:) = patch_main(:,i,:);
else
patch_output(1:index_previous-1,i,:) = patch_T(size_patch - degree_of_overlap + 1:size_patch - degree_of_overlap + index_previous-1,i,:);
patch_output(index_previous:degree_of_overlap,i,:) = patch_main(index_previous:degree_of_overlap,i,:);
end
end
else
patch_Horiz = Minimum_Error_Boundary_cut(patch_as_ref,patch_main,degree_of_overlap,'horizontal',size_patch);
patch_output = Minimum_Error_Boundary_cut(patch_as_ref,patch_Horiz,degree_of_overlap,'vertical',size_patch);
end
end
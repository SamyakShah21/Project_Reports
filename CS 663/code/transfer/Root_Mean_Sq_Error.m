function error_val = Root_Mean_Sq_Error(patch_no_1,patch_no_2)
%% Returns the squared error between two patches
% patch_no_1 - 1st patch
% patch_no_2 - 2nd patch

% Note that the error value is independent of order in which the two
% patches are provided
%%
temp1 = patch_no_1-patch_no_2;
temp2 = temp1.*temp1;
error_val = sum(sum(sum(temp2)));
end
function patch_rand = Patch_Random(img_main,size_patch)
%% Returns a random patch of specified size from a given image
%  img_main - Image from which a random patch will be taken
%  size - size of patch
%%
[H,W,~] = size(img_main);
rand_H = randi(H-size_patch+1); % 1st coordinate of left corner of patch
rand_W = randi(W-size_patch+1); % 2nd coordinate of left corner of patch
patch_rand = img_main(rand_H:rand_H+size_patch-1,rand_W:rand_W+size_patch-1,:);
end
%% myMainScript

%% Preprocessing for .gif extension images
i=6;
input_name = num2str(i);
input_folder = 'paper_images/';
output_name = strcat(input_name,'.jpg');
input_file = strcat(input_name,'.gif');
[texture_paper_pic,map] = imread(strcat('data/',input_folder,input_file));
texture_paper_pic = ind2rgb(texture_paper_pic,map);
original_pic = double(texture_paper_pic);
patch_size = 60;
modified_pic = Quilting(original_pic,patch_size);
%imshow(modified_pic);
output_folder = strcat('results/',num2str(patch_size),'/');
imwrite(modified_pic,strcat(output_folder, output_name));

%% preprocessing for .jpg extension images
% input_folder = 'extra_images/';
% for i=1:12
%     input_name = num2str(i);
%     output_name = strcat(input_name,'.jpg');
%     input_file = strcat(input_name,'.jpg');
%     texture_our_pic = imread(strcat('data/',input_folder,input_file));
%     original_pic = double(texture_our_pic)/255.0;
%     modified_pic = Quilting(original_pic);
%     %imshow(modified_pic);
%     imwrite(modified_pic,strcat('results/', output_name));
% end
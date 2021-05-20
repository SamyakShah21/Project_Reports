%% myMainScript

%% Loading the pictures
input_folder = 'transfer/';
input_name = 's3';
input_file = strcat(input_name,'.jpg');
texture_pic = imread(strcat('data/',input_folder,input_file));
texture_pic = double(texture_pic)/255.0;
input_file = strcat('Retro_Geometric_Pattern','.jpg');
target_pic =  imread(strcat('data/',input_folder,input_file));
target_pic = double(target_pic)/255.0;
target_pic = target_pic(1:300,1:300,:);
modified_pic = Texture_Transfer(target_pic,texture_pic);
%imshow(modified_pic);
output_name = 'Hybrid';
output_file = strcat(output_name,'.jpg');
imwrite(modified_pic,strcat('results/',output_file));
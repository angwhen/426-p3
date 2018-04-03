%% CMSC 426: Project 1 Starter Code
% Written by: Nitin J. Sanket (nitinsan@terpmail.umd.edu)
% PhD in CS Student at University of Maryland, College Park

%% Load Test Images
filename = sprintf('../%s/%d.jpg','Frames5',1); % CHANGE BACK TO i LATER
I = imread(filename);
%I = I_hold(330:390,650:710,:);
imshow(I);
figure
% https://blog.saush.com/2011/04/20/edge-detection-with-the-sobel-operator-in-ruby/
sobel = [-1 0 1; -2 0 2; -1 0 1];
big_first = conv2(sobel, make_gaussian_matrix(10,1,4));
small_first = conv2(sobel, make_gaussian_matrix(10,5,1));

filter_bank = cell(2,12);
for angle = 0:30:330
    filter_bank{1,angle/30+1} =  imrotate(big_first,angle,'bilinear','crop');
    filter_bank{2,angle/30+1} =  imrotate(small_first,angle,'bilinear','crop');
end

%save_filter_bank(gaussian_filter_bank,'../Images/GaussianFB_Combined_dif_sigma.png');

%% Generate Half-disk masks
% Display all the GHalf-disk masks and save image as HDMasks_ImageName.png
%  in the Images/ folder.  use command saveas

circle_diam1 = 5;
circle_diam2 = 10;

[cols rows] = meshgrid(1:circle_diam1, 1:circle_diam1);
half1 = (rows - circle_diam1/2).^2 + (cols - circle_diam1/2).^2 <= (circle_diam1/2).^2;

[cols rows] = meshgrid(1:circle_diam2, 1:circle_diam2);
half2 = (rows - circle_diam2/2).^2 + (cols - circle_diam2/2).^2 <= (circle_diam2/2).^2;

    
%iterate through angles, and remove that angle of the circle
half_disk_bank = cell(2,12);
for angle = 0:30:150
    slope = tand(angle);
     
    [cols rows] = meshgrid(1:circle_diam1, 1:circle_diam1);
    to_remove1 = rows-(circle_diam1/2) < slope*(cols-(circle_diam1/2));
    to_remove2 = rows-(circle_diam1/2) >= slope*(cols-(circle_diam1/2));
    half_disk_bank{1,angle/30+1} = half1.*to_remove1;
    half_disk_bank{1,angle/30+7} = half1.*to_remove2;

    [cols rows] = meshgrid(1:circle_diam2, 1:circle_diam2);
    to_remove1 = rows-(circle_diam2/2) < slope*(cols-(circle_diam2/2));
    to_remove2 = rows-(circle_diam2/2) >= slope*(cols-(circle_diam2/2));
    half_disk_bank{2,angle/30+1} = half2.*to_remove1;
    half_disk_bank{2,angle/30+7} = half2.*to_remove2;
end


% filter everything
[h,w] = size(filter_bank);
filter_bank = reshape(filter_bank,[1,h*w]);
test_images_filtered_cell = cell(1,h*w);
for f = 1:(w*h)
    test_images_filtered_cell{1,f} = imfilter(I,filter_bank{1,f});
end
[~,numfilts] = size(test_images_filtered_cell);
filtered_image = test_images_filtered_cell{1,1};
for f = 2:numfilts
   filtered_image = cat(3, filtered_image,test_images_filtered_cell{1,f});
end 

%kmeans the filtered image
bin_values = 10;
[x,y,z] = size(filtered_image);
shaped_image = reshape(filtered_image,[x*y,z]);

disp('starting k means')
idx = kmeans(double(shaped_image),bin_values,'MaxIter',250);
disp('ending kmeans')

texton_map = reshape(idx, [x,y]);
imagesc(texton_map); colormap(jet);

% Create Gaussian Matrices
function gauss = make_gaussian_matrix(size,sigma,elongation_factor)
    gauss = ones(size,size);
    for i = 1:size;
       for j = 1:size;
           gauss(i,j) = exp(-(((i-size/2)/elongation_factor).^2+(j-size/2).^2)/(2*sigma.^2));
       end
    end
end
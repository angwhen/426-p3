folder_name = 'Frames3';
num_images = size(dir(['../' folder_name '/*.jpg']),1); 
images_cell = cell(1,num_images);
for i=1:num_images
    filename = sprintf('../%s/%d.jpg',folder_name,i); % CHANGE BACK TO i LATER
    I  = imread(filename);
    images_cell{1,i} = I;
end

%imshow(images_cell{1,1});
%init_mask = roipoly();
init_mask = load('frames3_mask1'); init_mask = init_mask.init_mask;

%% Get transformations between frames 
%estimate whole object motion
transformation_cell = cell(1,num_images-1);
for i = 2:num_images
    gray_im1 = rgb2gray(images_cell{1,i-1});
    gray_im2 = rgb2gray(images_cell{1,i});
    points1 = detectSURFFeatures(gray_im1,'MetricThreshold',1500);
    points2 = detectSURFFeatures(gray_im2,'MetricThreshold',1500);
    [features1, validpts1]  = extractFeatures(gray_im1,points1);
    [features2, validpts2] = extractFeatures(gray_im2,points2);
    indexPairs = matchFeatures(features1,features2);
    matchedPoints1 = validpts1(indexPairs(:,1));
    matchedPoints2 = validpts2(indexPairs(:,2));
    transformation_cell{1,i-1} = estimateGeometricTransform(matchedPoints2,matchedPoints1,'affine');
end

halfw = 30; s = 80; % s is how many windows total

local_windows_center_cell_prev = get_window_pos_orig(init_mask,s);
local_windows_image_cell_prev = get_local_windows(images_cell{1,1},local_windows_center_cell_prev, halfw);
local_windows_mask_cell_prev = get_local_windows(init_mask,local_windows_center_cell_prev, halfw);
%[i, s] = size(local_windows_image_cell_prev); % s is number of windows
[combined_color_prob_cell_prev foreground_model_cell background_model_cell]= get_combined_color_prob_cell(local_windows_mask_cell_prev, local_windows_image_cell_prev, s);

color_model_confidence_cell_prev = get_color_model_confidence(local_windows_mask_cell_prev, combined_color_prob_cell_prev,halfw,s);
shape_model_confidence_mask_cell_prev = get_shape_model_confidence_mask_cell(local_windows_mask_cell_prev,color_model_confidence_cell_prev,halfw,s);
total_confidence_cell_prev = get_total_confidence_cell(shape_model_confidence_mask_cell_prev,combined_color_prob_cell_prev,local_windows_mask_cell_prev,s);

foreground_prob_prev = get_final_mask(rgb2gray(images_cell{1,1}),local_windows_center_cell_prev,total_confidence_cell_prev,halfw,s);
foreground_prev = foreground_prob_prev > 0.5;
%foreground_prev=imfill(foreground_prev,'holes');
imshow(foreground_prev);

save_image_with_boxes(images_cell{1,1},local_windows_center_cell_prev,halfw,s,sprintf('../MyOutput/%s/Windows_On_Image_1.png',folder_name));
imwrite(foreground_prev,sprintf('../MyOutput/%s/1.png',folder_name));

local_color_to_save = get_final_mask(rgb2gray(images_cell{1,1}),local_windows_center_cell_prev,combined_color_prob_cell_prev,halfw,s);
imwrite(local_color_to_save,sprintf('../MyOutput/%s/Local_Color_1.png',folder_name));
local_shape_to_save = get_final_mask(rgb2gray(images_cell{1,1}),local_windows_center_cell_prev,local_windows_mask_cell_prev,halfw,s);
imwrite(local_shape_to_save,sprintf('../MyOutput/%s/Local_Shape_1.png',folder_name));
local_shape_conf_to_save = get_final_mask(rgb2gray(images_cell{1,1}),local_windows_center_cell_prev,shape_model_confidence_mask_cell_prev,halfw,s);
imwrite(local_shape_conf_to_save,sprintf('../MyOutput/%s/Local_Shape_Conf_1.png',folder_name));

B = bwboundaries(foreground_prev);
imshow(images_cell{1,1});
hold on
for k = 1:length(B)
    boundary = B{k};
    plot(boundary(:,2), boundary(:,1), 'y', 'LineWidth', 2)
end
hold off
saveas(gcf,sprintf('../MyOutput/%s/Highlighted_%d.png',folder_name,1));
close all
figure

prev_mask = foreground_prev;

for frame =2:num_images
    fprintf("curr frame is %d\n",frame);
    %object motion, moving windows
    R = imref2d(size(prev_mask));
    %new_mask = imwarp(prev_mask,transformation_cell{1,frame-1},'OutputView',R);
    warped_image = imwarp(images_cell{1,frame-1},transformation_cell{1,frame-1},'OutputView',R);
    
    %local_windows_center_cell_prev = get_window_pos_orig(prev_mask,s);
    local_windows_center_cell_curr = get_new_windows_centers(local_windows_mask_cell_prev,local_windows_center_cell_prev,warped_image,images_cell{1,frame},transformation_cell{1,frame-1},halfw,s,frame,folder_name);
    
    save_image_with_boxes(images_cell{1,frame},local_windows_center_cell_curr,halfw,s,sprintf('../MyOutput/%s/Windows_On_Image_%d.png',folder_name,frame));
    
    local_windows_mask_cell_curr = get_local_windows(prev_mask,local_windows_center_cell_curr, halfw); %use warped mask or NOT???? IDKK
    
    imshow(get_final_mask(rgb2gray(images_cell{1,frame}),local_windows_center_cell_curr,local_windows_mask_cell_curr,halfw,s));

    %get cell with image in windows
    local_windows_image_cell_curr = get_local_windows(images_cell{1,frame},local_windows_center_cell_curr, halfw);
    
    %update color model
    [combined_color_prob_cell_curr,foreground_model_cell, background_model_cell] = get_combined_color_prob_cell2(local_windows_mask_cell_curr, local_windows_image_cell_curr, combined_color_prob_cell_prev, foreground_model_cell, background_model_cell,s);
    color_model_confidence_cell_curr = get_color_model_confidence(local_windows_mask_cell_curr, combined_color_prob_cell_curr,halfw,s);
    
    %update shape confidence based on change in color model confidence 
    shape_model_confidence_mask_cell_curr = get_shape_model_confidence_mask_cell(local_windows_mask_cell_curr,color_model_confidence_cell_curr,halfw,s);
    
    %combine shape and color model
    total_confidence_cell_curr = get_total_confidence_cell(shape_model_confidence_mask_cell_curr,combined_color_prob_cell_curr,local_windows_mask_cell_prev,s);

    %determine new mask
    foreground_prob_curr = get_final_mask(rgb2gray(images_cell{1,frame}),local_windows_center_cell_curr,total_confidence_cell_curr,halfw,s);
    %foreground_curr = get_snapped(foreground_prob_curr,foreground_prob_curr);
    foreground_curr = foreground_prob_curr >0.2;
    %foreground_curr=imfill(foreground_curr,'holes');
    
    %setting to prev
    prev_mask = foreground_curr;
    shape_model_confidence_mask_cell_prev =  shape_model_confidence_mask_cell_curr;
    local_windows_mask_cell_prev = local_windows_mask_cell_curr;
    local_windows_center_cell_prev = local_windows_center_cell_curr;
    local_windows_image_cell_prev = local_windows_image_cell_curr;
    combined_color_prob_cell_prev = combined_color_prob_cell_curr;
    total_confidence_cell_prev = total_confidence_cell_curr;
    
    %writing to files
    imwrite(foreground_curr,sprintf('../MyOutput/%s/%d.png',folder_name,frame));
    
    local_color_to_save = get_final_mask(rgb2gray(images_cell{1,frame}),local_windows_center_cell_curr,combined_color_prob_cell_curr,halfw,s);
    imwrite(local_color_to_save,sprintf('../MyOutput/%s/Local_Color_%d.png',folder_name,frame));
    local_shape_to_save = get_final_mask(rgb2gray(images_cell{1,frame}),local_windows_center_cell_prev,local_windows_mask_cell_prev,halfw,s);
    imwrite(local_shape_to_save,sprintf('../MyOutput/%s/Local_Shape_%d.png',folder_name,frame));
    local_shape_conf_to_save = get_final_mask(rgb2gray(images_cell{1,frame}),local_windows_center_cell_curr,shape_model_confidence_mask_cell_prev,halfw,s);
    imwrite(local_shape_conf_to_save,sprintf('../MyOutput/%s/Local_Shape_Conf_%d.png',folder_name,frame));
    
    B = bwboundaries(foreground_curr);
%     bsizes = cellfun('size',B,1);  
%     [M, maxes_ind] = max(bsizes); 
%     b = B{maxes_ind};
%     imshow(images_cell{1,frame});
%     hold on
%     for k = 1:length(b)
%         boundary = b(k,:);
%         plot(boundary(2), boundary(1), 'y', 'LineWidth', 2);
%     end
%     hold off

    imshow(images_cell{1,frame});
    hold on
    for k = 1:length(B)
        boundary = B{k};
        plot(boundary(:,2), boundary(:,1), 'y', 'LineWidth', 2)
    end
    hold off
    saveas(gcf,sprintf('../MyOutput/%s/Highlighted_%d.png',folder_name,frame));
    %imwrite(uint8(double(images_cell{1,frame}).*foreground_curr),sprintf('../MyOutput/%s/Image_Cutout_%d.png',folder_name,frame));
    
end
disp("DONE WITH ALL THE FRAMES!");

function local_windows_center_cell = get_window_pos_orig(init_mask,s)
    B = bwboundaries(init_mask); 
    % use the biggest thing in B
    bsizes = cellfun('size',B,1); 
    [M I] = max(bsizes);
    b = B{I};
    % also get second largest for sake of hole
%     bsizes(I) = 0;
%     [M I2] = max(bsizes);
%     b2 = B{I2};
%     b = cat(1,b,b2);
    
    [h w] = size(b);
    density = (h/s);
    if density < 1
        density = 1;
    end
    local_windows_center_cell = cell(1,s);
    for i = 1:s
        ind = uint32(i*density);
        r  = b(ind,1); c= b(ind,2);
        local_windows_center_cell{1,i} = [r c];
    end
end


function local_windows_image_cell = get_local_windows(I,local_windows_center_cell, halfwidth)
    [~, num_windows] = size(local_windows_center_cell);
    local_windows_image_cell = cell(1,num_windows);
    for i = 1:num_windows
        center = local_windows_center_cell{1,i};
        r = center(1); c = center(2);
        %fprintf("r is %d, c is %d\n",r,c);
        my_patch = I(r-halfwidth:r+halfwidth, c-halfwidth:c+halfwidth,:);
        local_windows_image_cell{1,i} = my_patch; 
        %imshow(my_patch);
    end
end


function [combined_color_prob_cell, foreground_model_cell, background_model_cell] = get_combined_color_prob_cell(local_windows_mask_cell_prev, local_windows_image_cell, s)
    combined_color_prob_cell = cell(1,s);
    foreground_model_cell = cell(1,s);
    background_model_cell = cell(1,s);
    options = statset('MaxIter',500);
    for i = 1:s
        curr_image = local_windows_image_cell{1,i};
        inverted = local_windows_mask_cell_prev{1,i}==0; %background becomes non zero
        
        [r c] = find(bwdist(inverted)>2);
        foreground_pix = rgb2lab(impixel(curr_image,c,r));
        foreground_model = fitgmdist(foreground_pix,3,'RegularizationValue',0.001,'Options',options);
        foreground_model_cell{1,i} = foreground_model;
        
        [r c] = find(bwdist(local_windows_mask_cell_prev{1,i})>2); %find all non zero that are more than 2 away from foreground 
        background_pix = rgb2lab(impixel(curr_image,c,r));
        background_model = fitgmdist(background_pix,3,'RegularizationValue',0.001,'Options',options);
        background_model_cell{1,i} = background_model;
        
        % probs
        [r c ~] = size(curr_image);
        values = rgb2lab(reshape(double(curr_image),[r*c 3]));

        fore_prob = pdf(foreground_model,values);
        back_prob = pdf(background_model,values);

        comb_prob = fore_prob./(fore_prob+back_prob);
        combined_color_prob_cell{1,i} = reshape(comb_prob,[r c]); 
        %imshow(combined_color_prob_cell{1,i});
    end
end

function color_model_confidence_cell = get_color_model_confidence(local_windows_mask_cell_prev, combined_color_prob_cell,halfw,s)
    color_model_confidence_cell = cell(1,s);
    for i = 1:s
        d1 = bwdist(local_windows_mask_cell_prev{1,i});
        d2 = bwdist(local_windows_mask_cell_prev{1,i}==0);
        d = max(d1,d2);
        wc = exp(-d.^2./halfw.^2);
        numer_window = abs(local_windows_mask_cell_prev{1,i} - combined_color_prob_cell{1,i}).*wc; 
        numer = sum(numer_window(:));
        denom = sum(wc(:));
        color_model_confidence_cell{1,i} = 1 - numer/denom;
        if i == 1
            fprintf("color model confidence for window 1 is %f\n",color_model_confidence_cell{1,1});
        end
    end
end

function shape_model_confidence_mask_cell = get_shape_model_confidence_mask_cell(local_windows_mask_cell_prev,color_model_confidence_cell,halfw,s)
    shape_model_confidence_mask_cell = cell(1,s);
    fcutoff = 0.6; sigma_min = 6; sigma_max = halfw*2+1; r=2; a = (sigma_max-sigma_min)/(1-fcutoff)^r; %paper sigma min =2
    for i = 1:s
        fc = color_model_confidence_cell{1,i};
        if fc > fcutoff
            sigma_s = sigma_min+a*(fc-fcutoff)^r;
        else
            sigma_s = sigma_min;
        end

        d1 = bwdist(local_windows_mask_cell_prev{1,i}); d2 = bwdist(local_windows_mask_cell_prev{1,i}==0);
        d = max(d1,d2);
        shape_model_confidence_mask_cell{1,i} = 1 - exp(-d.^2./sigma_s.^2); %eq 3
        %imshow(shape_model_confidence_mask_cell{1,i});
    end
end

function local_windows_center_cell2 = get_new_windows_centers(local_windows_mask_cell_prev,local_windows_center_cell,curr_image,next_image,my_tform,halfw,s,frame,folder_name)
    % DO CENTER MOVING WITH TRANSFORM
    [myh myw] = size(rgb2gray(curr_image));
    local_windows_center_cell2 = cell(1,s);
    for i = 1:s
        mycenter = local_windows_center_cell{1,i}; cr = double(mycenter(1)); cc = double(mycenter(2));
        [newr,newc] = transformPointsForward(my_tform,cr,cc);
     
        local_windows_center_cell2{1,i} = uint32([newr newc]);
    end

    opticFlow = opticalFlowFarneback;
    estimateFlow(opticFlow,rgb2gray(curr_image));
    flow = estimateFlow(opticFlow,rgb2gray(next_image));
    VxWindows = get_local_windows(flow.Vx,local_windows_center_cell2, halfw);
    VyWindows = get_local_windows(flow.Vy,local_windows_center_cell2, halfw);
    
%     imshow(imfuse(curr_image,next_image));
%     hold on
%     plot(flow,'DecimationFactor',[15 15],'ScaleFactor',5);
%     hold off;
%     saveas(gcf,sprintf('../MyOutput/%s/OpticalFlow_%d.png',folder_name,frame));
    
    for i = 1:s
        %get mean within foreground
        currVx = VxWindows{1,i} .* local_windows_mask_cell_prev{1,i};
        Vxmean = sum(currVx(:))/sum(currVx(:)~=0);
        currVy = VyWindows{1,i} .* local_windows_mask_cell_prev{1,i};
        Vymean = sum(currVy(:))/sum(currVy(:)~=0); 
        
        if isnan(Vxmean) || isnan(Vymean)
            Vxmean = 0; Vymean = 0;
        end

        new_center = uint32(double(local_windows_center_cell2{1,i}) + [Vymean Vxmean]);
        if new_center(1)+halfw >= myh || new_center(1)-halfw <= 1 || new_center(2)+halfw >= myw || new_center(2)-halfw <= 1
            new_center = local_windows_center_cell2{1,randi(i-1)}; 
        end
        local_windows_center_cell2{1,i} = new_center;
    end
   
end


%make new color models
function [combined_color_prob_cell2,foreground_model_cell, background_model_cell] = get_combined_color_prob_cell2(local_windows_mask_cell_prev2, local_windows_image_cell2,prev_combined_color_prob_cell, foreground_model_cell_prev, background_model_cell_prev,s)
    combined_color_prob_cell2 = cell(1,s);
    foreground_model_cell = cell(1,s);
    background_model_cell = cell(1,s);
    options = statset('MaxIter',500);
    for i = 1:s
        inverted = local_windows_mask_cell_prev2{1,i}==0;
        [r c] = find(bwdist(inverted)>5);
        foreground_pix = rgb2lab(impixel(local_windows_image_cell2{1,i},c,r));
        [a b] = size(foreground_pix);
        if a < 100 %if not enough data so loosen boundary
            [r c] = find(bwdist(inverted)>1);
            foreground_pix = rgb2lab(impixel(local_windows_image_cell2{1,i},c,r));
            [a b] = size(foreground_pix);
        end
        if a > b
            foreground_model = fitgmdist(foreground_pix,3,'RegularizationValue',0.001,'Options',options);
        else
            foreground_model =  foreground_model_cell_prev{1,i};
        end

        [r c] = find(bwdist(local_windows_mask_cell_prev2{1,i})>5);
        background_pix = rgb2lab(impixel(local_windows_image_cell2{1,i},c,r));
        [a b] = size(background_pix);
        if a < 100
            [r c] = find(bwdist(local_windows_mask_cell_prev2{1,i})>1);
            background_pix = rgb2lab(impixel(local_windows_image_cell2{1,i},c,r));
            [a b] = size(background_pix);
        end
        if a > b
            background_model = fitgmdist(background_pix,3,'RegularizationValue',0.001,'Options',options);
        else
            background_model =  background_model_cell_prev{1,i};
        end
        
        % probs
        [r c ~] = size(local_windows_image_cell2{1,i});
        values = rgb2lab(reshape(double(local_windows_image_cell2{1,i}),[r*c 3]));

        fore_prob = pdf(foreground_model,values);
        back_prob = pdf(background_model,values);

        comb_prob = fore_prob./(fore_prob+back_prob);
     
        % previous model probs
        fore_prob_prev = pdf(foreground_model_cell_prev{1,i},values);
        back_prob_prev = pdf(background_model_cell_prev{1,i},values);
        comb_prob_prev = fore_prob_prev./(fore_prob_prev+back_prob_prev);
        
        if sum(comb_prob >0.5) > sum(comb_prob_prev>0.2) % use prev model
            %disp("prev color");
            combined_color_prob_cell2{1,i} = reshape(comb_prob_prev,[r c]); 
            foreground_model_cell{1,i} = foreground_model_cell_prev{1,i};
            background_model_cell{1,i} = background_model_cell_prev{1,i};
        else
            %disp("new color");
            combined_color_prob_cell2{1,i} = reshape(comb_prob,[r c]); 
            foreground_model_cell{1,i} = foreground_model;
            background_model_cell{1,i} = background_model;
        end
%         imshow(combined_color_prob_cell2{1,i});
%         hold on
%         plot(uint32(c/2), uint32(r/2), 'r*', 'LineWidth', 2, 'MarkerSize', 5);
%         hold off
    end
end

function total_confidence_cell = get_total_confidence_cell(shape_model_confidence_mask_cell,combined_color_prob_cell2,local_windows_mask_cell_prev2,s)
    % part 7, combining shape and color models
    total_confidence_cell = cell(1,s);
    for i = 1:s
        fs = shape_model_confidence_mask_cell{1,i};
        pc = combined_color_prob_cell2{1,i};
        total_confidence_cell{1,i}= fs.*local_windows_mask_cell_prev2{1,i}+(1-fs).*pc;
        %imshow(total_confidence_cell{1,i});
    end
end


function foreground2 = get_final_mask(I,local_windows_center_cell,total_confidence_cell,halfw,s)
    [h w] = size(I); eps = 0.1;
    foreground_numer = zeros([h w]);
    foreground_denom = zeros([h w]);
    for i = 1:s
        center = local_windows_center_cell{1,i}; cr = center(1); cc = center(2);
        for a = 1:(halfw*2+1)
            for b = 1:(halfw*2+1)
                d = 1.0/(sqrt(double((a-cr).^2+(b-cc).^2))+eps);
                foreground_numer(cr-halfw+a,cc-halfw+b) = foreground_numer(cr-halfw+a,cc-halfw+b)+total_confidence_cell{1,i}(a,b)*d;
                foreground_denom(cr-halfw+a,cc-halfw+b) = foreground_denom(cr-halfw+a,cc-halfw+b)+d;
            end
        end
    end
    foreground2 = foreground_numer./foreground_denom;
    foreground2(isnan(foreground2))=0;
end

function BW_foreground = get_snapped(I,foreground)
    L = superpixels(I,500);
    f = find(foreground>0.8); b = find(foreground==0);
    BW_foreground = lazysnapping(I,L,f,b);
end

function save_image_with_boxes(I,local_windows_center_cell_curr,halfw,s,filename)
    [h w] = size(I);
    imshow(I);
    hold on
    for i = 1:s
        center = local_windows_center_cell_curr{1,i};
        cr = center(1); cc = center(2);
        rectangle('Position',[cc-halfw,cr-halfw,halfw*2+1,halfw*2+1],'EdgeColor','r');
        plot(cc, cr, 'r*', 'LineWidth', 2, 'MarkerSize', 5);
    end
    hold off 
    saveas(gcf,filename);
end


% function filter_bank = get_filter_bank()
%     sobel = [-1 0 1; -2 0 2; -1 0 1];
%     big_first = conv2(sobel, make_gaussian_matrix(10,1,4));
%     small_first = conv2(sobel, make_gaussian_matrix(10,5,1));
% 
%     filter_bank = cell(2,12);
%     for angle = 0:30:330
%         filter_bank{1,angle/30+1} =  imrotate(big_first,angle,'bilinear','crop');
%         filter_bank{2,angle/30+1} =  imrotate(small_first,angle,'bilinear','crop');
%     end
% end
% 
% function texton_map = get_texton_map(I,filter_bank)
%     % filter everything
%     [h,w] = size(filter_bank);
%     filter_bank = reshape(filter_bank,[1,h*w]);
%     test_images_filtered_cell = cell(1,h*w);
%     for f = 1:(w*h)
%         test_images_filtered_cell{1,f} = imfilter(I,filter_bank{1,f});
%     end
%     [~,numfilts] = size(test_images_filtered_cell);
%     filtered_image = test_images_filtered_cell{1,1};
%     for f = 2:numfilts
%        filtered_image = cat(3, filtered_image,test_images_filtered_cell{1,f});
%     end 
%     
%     %kmeans the filtered image
%     bin_values = 50;
%     [x,y,z] = size(filtered_image);
%     shaped_image = reshape(filtered_image,[x*y,z]);
% 
%     disp('starting k means')
%     idx = kmeans(double(shaped_image),bin_values,'MaxIter',250);
%     disp('ending kmeans')
% 
%     texton_map = reshape(idx, [x,y]);
%     %imagesc(texton_map); colormap(jet);
% end
% 
% % Create Gaussian Matrices
% function gauss = make_gaussian_matrix(size,sigma,elongation_factor)
%     gauss = ones(size,size);
%     for i = 1:size;
%        for j = 1:size;
%            gauss(i,j) = exp(-(((i-size/2)/elongation_factor).^2+(j-size/2).^2)/(2*sigma.^2));
%        end
%     end
% end
% 
% function texture_prob_cell =  get_texture_prob_cell(local_windows_mask_cell_prev, local_windows_texton_cell, s)
%     texture_prob_cell = cell(1,s);
%     for i = 1:s
%         curr_texton_map = local_windows_texton_cell{1,i};
%         % for foreground see how mnay of each texton
%         % for background see how many of each texton
%         % assign proportions to each texton
%         foreground = local_windows_mask_cell_prev{1,i}*curr_texton_map;
%         background = (local_windows_mask_cell_prev{1,i}==0)*curr_texton_map; %background becomes non zero
%         fore_prob_cell = cell(1,10);
%         for j = 1:50
%             fore_count = sum(foreground(:) == j);
%             back_count = sum(background(:) == j);
%             if fore_count + back_count > 0 
%                 fore_prob_cell{1,j} = fore_count/(back_count+fore_count);
%             else
%                 fore_prob_cell{1,j}  = 0;
%             end
%         end
%         texture_prob_cell{1,i} = curr_texton_map*-1;
%         for j = 1:50
%             texture_prob_cell{1,i} (texture_prob_cell{1,i} ==-j)=fore_prob_cell{1,j};
%         end
%     end
% end
folder_name = 'Frames4';
num_images = size(dir(['../' folder_name '/*.jpg']),1); 
images_cell = cell(1,num_images);
for i=1:num_images
    filename = sprintf('../%s/%d.jpg',folder_name,i); % CHANGE BACK TO i LATER
    images_cell{1,i}  = imread(filename);
end

%imshow(images_cell{1,1});
%init_mask = roipoly();
init_mask = load('frames4_mask1'); init_mask = init_mask.init_mask;

%% Get transformations between frames 
%estimate whole object motion
transformation_cell = cell(1,num_images-1);
for i = 2:num_images
    gray_im1 = rgb2gray(images_cell{1,i-1});
    gray_im2 = rgb2gray(images_cell{1,i});
    points1 = detectSURFFeatures(gray_im1,'MetricThreshold',200);
    points2 = detectSURFFeatures(gray_im2,'MetricThreshold',200);
    [features1, validpts1]  = extractFeatures(gray_im1,points1);
    [features2, validpts2] = extractFeatures(gray_im2,points2);
    indexPairs = matchFeatures(features1,features2);
    matchedPoints1 = validpts1(indexPairs(:,1));
    matchedPoints2 = validpts2(indexPairs(:,2));
    transformation_cell{1,i-1} = estimateGeometricTransform(matchedPoints2,matchedPoints1,'affine');
end

halfw = 25; s = 40; % s is how many windows total

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
imshow(images_cell{1,frame});
hold on
for k = 1:length(B)
    boundary = B{k};
    plot(boundary(:,2), boundary(:,1), 'y', 'LineWidth', 2)
end
hold off
saveas(gcf,sprintf('../MyOutput/%s/Highlighted_%d.png',folder_name,1));
%imwrite(uint8(double(images_cell{1,1}).*foreground_prev),sprintf('../MyOutput/%s/Image_Cutout_1.png',folder_name));

subplot(2,2,1), imshow(local_windows_image_cell_prev{1,1});
subplot(2,2,2), imshow(combined_color_prob_cell_prev{1,1});
subplot(2,2,3), imshow(shape_model_confidence_mask_cell_prev{1,1});
subplot(2,2,4), imshow(local_windows_mask_cell_prev{1,1});
saveas(gcf,sprintf('../MyOutput/OutputWindow/%s/Window1_1.png',folder_name));   
close all
figure

prev_mask = foreground_prev;
%imshow(get_final_mask(rgb2gray(images_cell{1,1}),local_windows_center_cell_prev,get_local_windows(prev_mask,local_windows_center_cell_prev, halfw),0,halfw,s));

for frame =2:num_images
    fprintf("curr frame is %d\n",frame);
    %object motion, moving windows
    R = imref2d(size(prev_mask));
    %new_mask = imwarp(prev_mask,transformation_cell{1,frame-1},'OutputView',R);
    warped_image = imwarp(images_cell{1,frame-1},transformation_cell{1,frame-1},'OutputView',R);
    
    %idk adding stuff ... changing centers based on new mask first?
    local_windows_center_cell_prev = get_window_pos_orig(prev_mask,s);
    %FIX WRPING, WRONG EVEN FOR EXACT SAME IMAGE ALL THE TIME
    local_windows_center_cell_curr = get_new_windows_centers(local_windows_mask_cell_prev,local_windows_center_cell_prev,warped_image,images_cell{1,frame},transformation_cell{1,frame-1},halfw,s);
    
    save_image_with_boxes(images_cell{1,frame},local_windows_center_cell_curr,halfw,s,sprintf('../MyOutput/%s/Windows_On_Image_%d.png',folder_name,frame));
    
    local_windows_mask_cell_curr = get_local_windows(prev_mask,local_windows_center_cell_curr, halfw); %use warped mask or NOT???? IDKK
   
%     for j = 1:s
%         myimage = images_cell{1,frame};
%         center = local_windows_center_cell_curr{1,j}; cr = center(1); cc = center(2);
%         curr = local_windows_mask_cell_curr{1,j};
%         for a = 1:(halfw*2+1)
%             for b =  1:(halfw*2+1)
%                 val = uint8(curr(a,b)*256);
%                 myimage(cr-halfw+a,cc-halfw+b,:) = [val val val];
%             end
%         end
%         imshow(myimage);
%         hold on
%         plot(cc, cr, 'r*', 'LineWidth', 2, 'MarkerSize', 10);
%         hold off 
%     end
    
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
    foreground_curr = foreground_prob_curr >0.5;
    
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
    imshow(images_cell{1,frame});
    hold on
    for k = 1:length(B)
        boundary = B{k};
        plot(boundary(:,2), boundary(:,1), 'y', 'LineWidth', 2)
    end
    hold off
    saveas(gcf,sprintf('../MyOutput/%s/Highlighted_%d.png',folder_name,frame));
    %imwrite(uint8(double(images_cell{1,frame}).*foreground_curr),sprintf('../MyOutput/%s/Image_Cutout_%d.png',folder_name,frame));
    
    %save for just one window
    subplot(2,2,1), imshow(local_windows_image_cell_curr{1,1});
    subplot(2,2,2), imshow(combined_color_prob_cell_curr{1,1});
    subplot(2,2,3), imshow(shape_model_confidence_mask_cell_prev{1,1});
    subplot(2,2,4), imshow(local_windows_mask_cell_prev{1,1});
    saveas(gcf,sprintf('../MyOutput/OutputWindow/%s/Window1_%d.png',folder_name,frame));   
    close all
    figure
end

function local_windows_center_cell = get_window_pos_orig(init_mask,s)
    imshow(init_mask);
    b = bwboundaries(init_mask); b = b{1,1}; %coords of edge of mask
    [h w] = size(b);
    density = h/s;
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
    for i = 1:s
        curr_image = local_windows_image_cell{1,i};
        inverted = local_windows_mask_cell_prev{1,i}==0; %background becomes non zero
        
        [r c] = find(bwdist(inverted)>5);
        foreground_pix = rgb2lab(impixel(curr_image,c,r));
        foreground_model = fitgmdist(foreground_pix,1,'CovarianceType','diagonal','RegularizationValue',0.001);
        foreground_model_cell{1,i} = foreground_model;
        
        [r c] = find(bwdist(local_windows_mask_cell_prev{1,i})>5); %find all non zero that are more than 2 away from foreground 
        background_pix = rgb2lab(impixel(curr_image,c,r));
        background_model = fitgmdist(background_pix,1,'CovarianceType','diagonal','RegularizationValue',0.001);
        background_model_cell{1,i} = background_model;
        
        % probs
        [r c ~] = size(curr_image);
        values = rgb2lab(reshape(double(curr_image),[r*c 3]));

        fore_prob = pdf(foreground_model,values);
        back_prob = pdf(background_model,values);

        comb_prob = fore_prob./(fore_prob+back_prob);
        combined_color_prob_cell{1,i} = reshape(comb_prob,[r c]); 
        imshow(combined_color_prob_cell{1,i});
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
    fcutoff = 0.8; sigma_min = 2; sigma_max = halfw*2+1; r=2; a = (sigma_max-sigma_min)/(1-fcutoff)^r;
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

function local_windows_center_cell2 = get_new_windows_centers(local_windows_mask_cell_prev,local_windows_center_cell,curr_image,next_image,my_tform,halfw,s)
    % DO CENTER MOVING WITH TRANSFORM
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
    
    for i = 1:s
        %get mean within foreground
        currVx = VxWindows{1,i} .* local_windows_mask_cell_prev{1,i};
        Vxmean = sum(currVx(:))/sum(currVx(:)~=0);
        currVy = VyWindows{1,i} .* local_windows_mask_cell_prev{1,i};
        Vymean = sum(currVy(:))/sum(currVy(:)~=0);
        
        if isnan(Vxmean) || isnan(Vymean)
            Vxmean = 0; Vymean = 0;
        end

        local_windows_center_cell2{1,i} = uint32(double(local_windows_center_cell2{1,i}) + [Vxmean Vymean]);
    end
    
   
end


%make new color models
function [combined_color_prob_cell2,foreground_model_cell, background_model_cell] = get_combined_color_prob_cell2(local_windows_mask_cell_prev2, local_windows_image_cell2,prev_combined_color_prob_cell, foreground_model_cell_prev, background_model_cell_prev,s)
    combined_color_prob_cell2 = cell(1,s);
    for i = 1:s
        inverted = local_windows_mask_cell_prev2{1,i}==0;
        [r c] = find(bwdist(inverted)>5);
        foreground_pix = rgb2lab(impixel(local_windows_image_cell2{1,i},c,r));
        [a b] = size(foreground_pix);
        if a > b
            foreground_model = fitgmdist(foreground_pix,1,'CovarianceType','diagonal','RegularizationValue',0.001);
        else
            foreground_model =  foreground_model_cell_prev{1,i};
        end

        [r c] = find(bwdist(local_windows_mask_cell_prev2{1,i})>5);
        background_pix = rgb2lab(impixel(local_windows_image_cell2{1,i},c,r));
        [a b] = size(background_pix);
        if a > b
            background_model = fitgmdist(background_pix,1,'CovarianceType','diagonal','RegularizationValue',0.001);
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
        
        if sum(comb_prob >0.5) > sum(comb_prob_prev>0.5)
            combined_color_prob_cell2{1,i} = reshape(comb_prob_prev,[r c]); 
            foreground_model_cell{1,i} = foreground_model_cell_prev{1,i};
            background_model_cell{1,i} = background_model_cell_prev{1,i};
        else
            combined_color_prob_cell2{1,i} = reshape(comb_prob,[r c]); 
            foreground_model_cell{1,i} = foreground_model;
            background_model_cell{1,i} = background_model;
        end
        imshow(combined_color_prob_cell2{1,i});
        hold on
        plot(uint32(c/2), uint32(r/2), 'r*', 'LineWidth', 2, 'MarkerSize', 5);
        hold off
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
%         if mod(i,3) == 0 
%             plot(cc, cr, 'r*', 'LineWidth', 2, 'MarkerSize', 5);
%         elseif mod(i,3) == 1 
%             plot(cc, cr, 'y*', 'LineWidth', 2, 'MarkerSize', 5);
%         else
%             plot(cc, cr, 'g*', 'LineWidth', 2, 'MarkerSize', 5);
%         end
    end
    hold off 
    saveas(gcf,filename);
end

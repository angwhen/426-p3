folder_name = 'Frames1';
num_images = size(dir(['../' folder_name '/*.jpg']),1);
images_cell = cell(1,num_images);
for i=1:num_images
    filename = sprintf('../%s/%d.jpg',folder_name,i);
    images_cell{1,i}  = imread(filename);
end

%imshow(images_cell{1,1});
%init_mask = roipoly();
init_mask = load('frames1_mask1'); init_mask = init_mask.init_mask;

%% Get transformations between frames 
%estimate whole object motion
transformation_cell = cell(1,num_images-1);
for i = 2:num_images
    gray_im1 = rgb2gray(images_cell{1,i-1});
    gray_im2 = rgb2gray(images_cell{1,i});
    points1 = detectSURFFeatures(gray_im1,'MetricThreshold',100);
    points2 = detectSURFFeatures(gray_im2,'MetricThreshold',100);
    [features1, validpts1]  = extractFeatures(gray_im1,points1);
    [features2, validpts2] = extractFeatures(gray_im2,points2);
    indexPairs = matchFeatures(features1,features2);
    matchedPoints1 = validpts1(indexPairs(:,1));
    matchedPoints2 = validpts2(indexPairs(:,2));
    transformation_cell{1,i-1} = estimateGeometricTransform(matchedPoints2,matchedPoints1,'affine');
end

halfw = 15; density = 10;

local_windows_center_cell_prev = get_window_pos_orig(init_mask,density);
local_windows_image_cell_prev = get_local_windows(images_cell{1,1},local_windows_center_cell_prev, halfw);
local_windows_mask_cell_prev = get_local_windows(init_mask,local_windows_center_cell_prev, halfw);
[i, s] = size(local_windows_image_cell_prev); % s is number of windows
combined_color_prob_cell_prev = get_combined_color_prob_cell(local_windows_mask_cell_prev, local_windows_image_cell_prev, s);
color_model_confidence_cell_prev = get_color_model_confidence(local_windows_mask_cell_prev, combined_color_prob_cell_prev,halfw,s);
shape_model_confidence_mask_cell_prev = get_shape_model_confidence_mask_cell(local_windows_mask_cell_prev,color_model_confidence_cell_prev,halfw,s);
total_confidence_cell_prev = get_total_confidence_cell(shape_model_confidence_mask_cell_prev,combined_color_prob_cell_prev,local_windows_mask_cell_prev,s);

foreground_prob_prev = get_final_mask(rgb2gray(images_cell{1,1}),local_windows_center_cell_prev,total_confidence_cell_prev,init_mask,halfw,s);
foreground_prev = foreground_prob_prev > 0.5;
imshow(foreground_prev);

save_image_with_boxes(images_cell{1,1},local_windows_center_cell_prev,halfw,s,sprintf('../Output/Windows_On_Image_%s_1.png',folder_name));
% imwrite(foreground_prev,sprintf('../Output/%s_1.png',folder_name));

local_color_to_save = get_final_mask(rgb2gray(images_cell{1,1}),local_windows_center_cell_prev,combined_color_prob_cell_prev,foreground_prev,halfw,s);
imwrite(local_color_to_save,sprintf('../Output/Local_Color_%s_1.png',folder_name));
% local_shape_to_save = get_final_mask(rgb2gray(images_cell{1,1}),local_windows_center_cell_prev,local_windows_mask_cell_prev,foreground_prev,halfw,s);
% imwrite(local_shape_to_save,sprintf('../Output/Local_Shape_%s_1.png',folder_name));
% local_shape_conf_to_save = get_final_mask(rgb2gray(images_cell{1,1}),local_windows_center_cell_prev,shape_model_confidence_mask_cell_prev,foreground_prev,halfw,s);
% imwrite(local_shape_conf_to_save,sprintf('../Output/Local_Shape_Conf_%s_1.png',folder_name));


prev_mask = foreground_prev;
for frame =2:num_images
    fprintf("curr frame is %d\n",frame);
    %object motion, moving windows
    R = imref2d(size(prev_mask));
    new_mask = imwarp(prev_mask,transformation_cell{1,frame-1},'OutputView',R);
    warped_image = imwarp(images_cell{1,frame-1},transformation_cell{1,frame-1},'OutputView',R);
    local_windows_center_cell_curr = get_new_windows_centers(local_windows_mask_cell_prev,local_windows_center_cell_prev,warped_image,images_cell{1,frame},transformation_cell{1,frame-1},halfw,s);
    save_image_with_boxes(images_cell{1,frame},local_windows_center_cell_curr,halfw,s,sprintf('../Output/Windows_On_Image_%s_%d.png',folder_name,frame));

    local_windows_mask_cell_curr = get_local_windows(new_mask,local_windows_center_cell_curr, halfw);
    
    %get cell with image in windows
    local_windows_image_cell_curr = get_local_windows(images_cell{1,frame},local_windows_center_cell_curr, halfw);
    
    %update color model
    combined_color_prob_cell_curr = get_combined_color_prob_cell2(local_windows_mask_cell_curr, local_windows_image_cell_curr, combined_color_prob_cell_prev, s);
    color_model_confidence_cell_curr = get_color_model_confidence(local_windows_mask_cell_curr, combined_color_prob_cell_curr,halfw,s);

    %combine shape and color model
    total_confidence_cell_curr = get_total_confidence_cell(shape_model_confidence_mask_cell_prev,combined_color_prob_cell_curr,local_windows_mask_cell_prev,s);

    %determine new mask
    foreground_prob_curr = get_final_mask(rgb2gray(images_cell{1,frame}),local_windows_center_cell_curr,total_confidence_cell_curr,new_mask,halfw,s);
    %foreground_curr = get_snapped(foreground_prob_curr,foreground_prob_curr);
    foreground_curr = foreground_prob_curr >0.5; %not showing the cutoff because WTV
    imshow(foreground_curr);
    
    %setting to prev
    prev_mask = foreground_curr;
    local_windows_mask_cell_prev = local_windows_mask_cell_curr;
    local_windows_center_cell_prev = local_windows_center_cell_curr;
    local_windows_image_cell_prev = local_windows_image_cell_curr;
    combined_color_prob_cell_prev = combined_color_prob_cell_curr;
    total_confidence_cell_prev = total_confidence_cell_curr;
    
    %writing to files
    imwrite(foreground_curr,sprintf('../Output/%s_%d.png',folder_name,frame));
    
    local_color_to_save = get_final_mask(rgb2gray(images_cell{1,frame}),local_windows_center_cell_curr,combined_color_prob_cell_curr,new_mask,halfw,s);
    imwrite(local_color_to_save,sprintf('../Output/Local_Color_%s_%d.png',folder_name,frame));
    local_shape_to_save = get_final_mask(rgb2gray(images_cell{1,frame}),local_windows_center_cell_prev,local_windows_mask_cell_prev,new_mask,halfw,s);
    imwrite(local_shape_to_save,sprintf('../Output/Local_Shape_%s_%d.png',folder_name,frame));
    local_shape_conf_to_save = get_final_mask(rgb2gray(images_cell{1,frame}),local_windows_center_cell_curr,shape_model_confidence_mask_cell_prev,new_mask,halfw,s);
    imwrite(local_shape_conf_to_save,sprintf('../Output/Local_Shape_Conf_%s_%d.png',folder_name,frame));
end

function local_windows_center_cell = get_window_pos_orig(init_mask,density)
    b = bwboundaries(init_mask); b = b{1,1}; %coords of edge of mask
    [h w] = size(b);
    local_windows_center_cell = cell(1,floor(h/density));
    for i = 1:density:h %floor(h/density)
        r  = b(i,1); c= b(i,2);
        local_windows_center_cell{1,(i-1)/density+1} = [r c];
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
    end
end

function combined_color_prob_cell = get_combined_color_prob_cell(local_windows_mask_cell_prev, local_windows_image_cell, s)
    combined_color_prob_cell = cell(1,s);
    for i = 1:s
        curr_image = local_windows_image_cell{1,i};
        inverted = local_windows_mask_cell_prev{1,i}==0; %background becomes non zero
        
        [r c] = find(bwdist(inverted)>2);
        foreground_pix = rgb2lab(impixel(curr_image,r,c));
        foreground_model = fitgmdist(foreground_pix,1,'CovarianceType','diagonal','RegularizationValue',0.001);

        
        [r c] = find(bwdist(local_windows_mask_cell_prev{1,i})>2); %find all non zero that are more than 2 away from foreground 
        background_pix = rgb2lab(impixel(curr_image,c,r));
        background_model = fitgmdist(background_pix,1,'CovarianceType','diagonal','RegularizationValue',0.001);

        % probs
        [r c ~] = size(curr_image);
        values = rgb2lab(reshape(double(curr_image),[r*c 3]));

        fore_prob = pdf(foreground_model,values);
        back_prob = pdf(background_model,values);

        comb_prob = fore_prob./(fore_prob+back_prob);
        %norm_comb_prob = comb_prob - min(comb_prob); 
        %norm_comb_prob = norm_comb_prob ./ max(norm_comb_prob(:));
        combined_color_prob_cell{1,i} = reshape(comb_prob,[r c]); 
        imshow(mat2gray(reshape(fore_prob,[r c])));
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
    % move each point with the affine transformation
    %get average flow for each window
    opticFlow = opticalFlowHS;
    estimateFlow(opticFlow,rgb2gray(curr_image));
    flow = estimateFlow(opticFlow,rgb2gray(next_image));

    VxWindows = get_local_windows(flow.Vx,local_windows_center_cell, halfw);
    VyWindows = get_local_windows(flow.Vy,local_windows_center_cell, halfw);
    local_windows_center_cell2 = cell(1,s);
    for i = 1:s
        % get mean within foreground
        currVx = VxWindows{1,i}*local_windows_mask_cell_prev{1,i};
        Vxmean = sum(currVx(:))/sum(currVx(:)~=0);
        currVy = VyWindows{1,i}*local_windows_mask_cell_prev{1,i};
        Vymean = sum(currVy(:))/sum(currVy(:)~=0);
        if isnan(Vxmean) || isnan(Vymean)
            Vxmean = 0; Vymean = 0;
        end

        mycenter = local_windows_center_cell{1,i}; cr = double(mycenter(1)); cc = double(mycenter(2));
        [newr,newc] = transformPointsForward(my_tform,cr,cc);
     
        local_windows_center_cell2{1,i} = uint32([newr newc]+[Vymean Vxmean]);
        if local_windows_center_cell2{1,i}(1) == 0 || local_windows_center_cell2{1,i}(2) == 0
            disp("zeros here");
            disp(newr)
            disp(newc)
            disp(Vxmean)
            disp(Vymean)
            dbstop
        end
    end
end


%make new color models
function combined_color_prob_cell2 = get_combined_color_prob_cell2(local_windows_mask_cell_prev2, local_windows_image_cell2,prev_combined_color_prob_cell,s)
    combined_color_prob_cell2 = cell(1,s);
    for i = 1:s
        inverted = local_windows_mask_cell_prev2{1,i}==0;
        [r c] = find(bwdist(inverted)>2);
        foreground_pix = rgb2lab(impixel(local_windows_image_cell2{1,i},c,r));
        [a b] = size(foreground_pix);
        if a > b
            foreground_model = fitgmdist(foreground_pix,1,'CovarianceType','diagonal','RegularizationValue',0.001);
        else
            combined_color_prob_cell2{1,i} = prev_combined_color_prob_cell{1,i};
            continue
        end

        [r c] = find(bwdist(local_windows_mask_cell_prev2{1,i})>2);
        background_pix = rgb2lab(impixel(local_windows_image_cell2{1,i},c,r));
        [a b] = size(background_pix);
        if a > b
            background_model = fitgmdist(background_pix,1,'CovarianceType','diagonal','RegularizationValue',0.001);
        else
            combined_color_prob_cell2{1,i} = prev_combined_color_prob_cell{1,i};
            continue
        end
        
        % probs
        [r c ~] = size(local_windows_image_cell2{1,i});
        values = rgb2lab(reshape(double(local_windows_image_cell2{1,i}),[r*c 3]));

        fore_prob = pdf(foreground_model,values);
        back_prob = pdf(background_model,values);

        comb_prob = fore_prob./(fore_prob+back_prob);
        %norm_comb_prob = comb_prob - min(comb_prob); 
        %norm_comb_prob = norm_comb_prob ./ max(norm_comb_prob(:));

        if sum(comb_prob >0.5) > sum(prev_combined_color_prob_cell{1,i}(:)>0.5)
            combined_color_prob_cell2{1,i} = prev_combined_color_prob_cell{1,i};
        else
            combined_color_prob_cell2{1,i} = reshape(comb_prob,[r c]); 
        end
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

function foreground2 = get_final_mask(I,local_windows_center_cell2,total_confidence_cell,whole_mask,halfw,s)
    [h w] = size(I); eps = 0.1;
    foreground2 = zeros([h w]);%double(whole_mask); %idea wass to fill in middle
    for r =1:h
        for c = 1:w
            numer = 0; denom = 0;
            for i = 1:s
                center =  local_windows_center_cell2{1,i}; cr = center(1); cc = center(2);
                if (r <= cr+halfw && r >= cr-halfw) && (c <= cc+halfw && c >= cc-halfw)
                    d = 1/(sqrt((r-halfw).^2 +(c-halfw).^2)+eps);
                    window_r = (r-cr)+halfw+1; window_c = (c-cc)+halfw+1;
                    numer = numer + total_confidence_cell{1,i}(window_r,window_c) * d;
                    denom = denom + d;
                end
            end
            if denom > 0
                foreground2(r,c) = numer/denom;
            end
        end
    end
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
        rectangle('Position',[cc-halfw,cr-halfw,halfw*2+1,halfw*2+1]);
    end
    hold off 
    saveas(gcf,filename);
end

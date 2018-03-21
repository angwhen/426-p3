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
    transformation_cell{1,i-1} = estimateGeometricTransform(matchedPoints1,matchedPoints2,'affine');
end

halfw = 15; density = 4;


local_windows_center_cell = get_window_pos(init_mask,density);
local_windows_image_cell = get_local_windows(images_cell{1,1},local_windows_center_cell, halfw);
local_windows_mask_cell = get_local_windows(init_mask,local_windows_center_cell, halfw);
[i, s] = size(local_windows_image_cell); % s is number of windows
combined_color_prob_cell = get_combined_color_prob_cell(local_windows_mask_cell, local_windows_image_cell, s);
color_model_confidence_cell = get_color_model_confidence(local_windows_mask_cell, combined_color_prob_cell,halfw,s);
shape_model_confidence_mask_cell = get_shape_model_confidence_mask_cell(local_windows_mask_cell,color_model_confidence_cell,halfw,s);

new_mask = imwarp(init_mask,transformation_cell{1,1});
local_windows_center_cell2 = get_new_windows_centers(local_windows_center_cell,images_cell{1,1},images_cell{1,2},halfw,s);
local_windows_image_cell2 = get_local_windows(images_cell{1,2},local_windows_center_cell2, halfw);
combined_color_prob_cell2 = get_combined_color_prob_cell2(local_windows_mask_cell, local_windows_image_cell2, combined_color_prob_cell, s);
color_model_confidence_cell2 = get_color_model_confidence(local_windows_mask_cell, combined_color_prob_cell2,halfw,s);

local_windows_mask_cell2 = get_local_windows(new_mask,local_windows_center_cell, halfw);
total_confidence_cell = get_total_confidence_cell(shape_model_confidence_mask_cell,combined_color_prob_cell2,local_windows_mask_cell2,s);

foreground2 = get_final_mask(rgb2gray(images_cell{1,2}),local_windows_center_cell2,total_confidence_cell,halfw,s);
L = superpixels(images_cell{1,2},400);
foreground = find(foreground2>0.8); background = find(foreground2==0);
BW_foreground = lazysnapping(images_cell{1,2},L,foreground,background);
imshow(BW_foreground);


function local_windows_center_cell = get_window_pos(init_mask,density)
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
        my_patch = I(r-halfwidth:r+halfwidth, c-halfwidth:c+halfwidth,:);
        local_windows_image_cell{1,i} = my_patch; 
    end
end

function combined_color_prob_cell = get_combined_color_prob_cell(local_windows_mask_cell, local_windows_image_cell, s)
    combined_color_prob_cell = cell(1,s);
    for i = 1:s
        [r c] = find(bwdist(local_windows_mask_cell{1,i})>2);
        foreground_pix = impixel(local_windows_image_cell{1,i},r,c);
        foreground_model = fitgmdist(foreground_pix,1);

        inverted = local_windows_mask_cell{1,i}==0;
        [r c] = find(bwdist(inverted)>2);
        background_pix = impixel(local_windows_image_cell{1,i},r,c);
        %imshow(local_windows_mask_cell{1,i});
        background_model = fitgmdist(background_pix,1);

        % probs
        [r c ~] = size(local_windows_image_cell{1,i});
        values = reshape(double(local_windows_image_cell{1,i}),[r*c 3]);

        fore_prob = pdf(foreground_model,values);
        back_prob = pdf(background_model,values);

        comb_prob = fore_prob./(fore_prob+back_prob);
        norm_comb_prob = comb_prob - min(comb_prob); 
        norm_comb_prob = norm_comb_prob ./ max(norm_comb_prob(:));
        combined_color_prob_cell{1,i} = reshape(norm_comb_prob,[r c]); 
        %imshow(combined_prob_cell{1,i});
    end
end

function color_model_confidence_cell = get_color_model_confidence(local_windows_mask_cell, combined_color_prob_cell,halfw,s)
    color_model_confidence_cell = cell(1,s);
    for i = 1:s
        d1 = bwdist(local_windows_mask_cell{1,i});
        d2 = bwdist(local_windows_mask_cell{1,i}==0);
        d = max(d1,d2);
        wc = exp(-d.^2./halfw.^2);
        numer_window = abs(local_windows_mask_cell{1,i} - combined_color_prob_cell{1,i}).*wc; 
        numer = sum(numer_window(:));
        denom = sum(wc(:));
        color_model_confidence_cell{1,i} = 1 - numer/denom;
    end
end

function shape_model_confidence_mask_cell = get_shape_model_confidence_mask_cell(local_windows_mask_cell,color_model_confidence_cell,halfw,s)
    shape_model_confidence_mask_cell = cell(1,s);
    fcutoff = 0.8; sigma_min = 2; sigma_max = halfw*2+1; r=2; a = (sigma_max-sigma_min)/(1-fcutoff)^r;
    for i = 1:s
        fc = color_model_confidence_cell{1,i};
        if fc > fcutoff
            sigma_s = sigma_min+a*(fc-fcutoff)^r;
        else
            sigma_s = sigma_min;
        end

        d1 = bwdist(local_windows_mask_cell{1,i}); d2 = bwdist(local_windows_mask_cell{1,i}==0);
        d = max(d1,d2);
        shape_model_confidence_mask_cell{1,i} = 1 - exp(-d.^2./sigma_s.^2); %eq 3
        %imshow(shape_model_confidence_mask_cell{1,i});
    end
end

function local_windows_center_cell2 = get_new_windows_centers(local_windows_center_cell,curr_image,next_image,halfw,s)
    %get average flow for each window
    opticFlow = opticalFlowHS;
    estimateFlow(opticFlow,rgb2gray(curr_image));
    flow = estimateFlow(opticFlow,rgb2gray(next_image));

    VxWindows = get_local_windows(flow.Vx,local_windows_center_cell, halfw);
    VyWindows = get_local_windows(flow.Vy,local_windows_center_cell, halfw);
    local_windows_center_cell2 = cell(1,s);
    for i = 1:s
        Vxmean = mean(VxWindows{1,i}(:));
        Vymean = mean(VyWindows{1,i}(:));
        local_windows_center_cell2{1,i} = uint32(local_windows_center_cell{1,i}+[Vxmean Vymean]);
    end
end


%make new color models
function combined_color_prob_cell2 = get_combined_color_prob_cell2(local_windows_mask_cell2, local_windows_image_cell2,prev_combined_color_prob_cell,s)
    combined_color_prob_cell2 = cell(1,s);
    for i = 1:s
        [r c] = find(bwdist(local_windows_mask_cell2{1,i})>2);
        foreground_pix = impixel(local_windows_image_cell2{1,i},r,c);
        foreground_model = fitgmdist(foreground_pix,1);

        inverted = local_windows_mask_cell2{1,i}==0;
        [r c] = find(bwdist(inverted)>2);
        background_pix = impixel(local_windows_image_cell2{1,i},r,c);
        background_model = fitgmdist(background_pix,1);

        % probs
        [r c ~] = size(local_windows_image_cell2{1,i});
        values = reshape(double(local_windows_image_cell2{1,i}),[r*c 3]);

        fore_prob = pdf(foreground_model,values);
        back_prob = pdf(background_model,values);

        comb_prob = fore_prob./(fore_prob+back_prob);
        norm_comb_prob = comb_prob - min(comb_prob); 
        norm_comb_prob = norm_comb_prob ./ max(norm_comb_prob(:));

        if sum(norm_comb_prob >0.5) > sum(prev_combined_color_prob_cell{1,i}(:)>0.5)
            combined_color_prob_cell2{1,i} = prev_combined_color_prob_cell{1,i};
        else
            combined_color_prob_cell2{1,i} = reshape(norm_comb_prob,[r c]); 
        end
    end
end

function total_confidence_cell = get_total_confidence_cell(shape_model_confidence_mask_cell,combined_color_prob_cell2,local_windows_mask_cell2,s)
    % part 7, combining shape and color models
    total_confidence_cell = cell(1,s);
    for i = 1:s
        fs = shape_model_confidence_mask_cell{1,i};
        pc = combined_color_prob_cell2{1,i};
        total_confidence_cell{1,i}= fs.*local_windows_mask_cell2{1,i}+(1-fs).*pc;
        %imshow(total_confidence_cell{1,i});
    end
end

function foreground2 = get_final_mask(I,local_windows_center_cell2,total_confidence_cell,halfw,s)
    [h w] = size(I); eps = 0.1;
    foreground2 = zeros([h w]);
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

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

halfw = 15; density = 6;
local_windows_center_cell = get_window_pos(init_mask,density);
local_windows_image_cell = get_local_windows(images_cell{1,1},local_windows_center_cell, halfw);
local_windows_mask_cell = get_local_windows(init_mask,local_windows_center_cell, halfw);

[i s] = size(local_windows_image_cell);
foreground_model_cell = cell(1,s);
background_model_cell = cell(1,s);
combined_prob_cell = cell(1,s);
for i = 1:s
    [r c] = find(bwdist(local_windows_mask_cell{1,i})>4);
    foreground_pix = impixel(local_windows_image_cell{1,i},r,c);
    foreground_model_cell{1,i} = fitgmdist(foreground_pix,1);
    
    inverted = local_windows_mask_cell{1,i}==0;
    [r c] = find(bwdist(inverted)>4);
    background_pix = impixel(local_windows_image_cell{1,i},r,c);
    background_model_cell{1,i} = fitgmdist(background_pix,1);
    
    % probs
    [r c ~] = size(local_windows_image_cell{1,i});
    values = reshape(double(local_windows_image_cell{1,i}),[r*c 3]);
    
    fore_prob = pdf(foreground_model_cell{1,i},values);
    back_prob = pdf(background_model_cell{1,i},values);
    
    comb_prob = fore_prob./(fore_prob+back_prob);
    norm_comb_prob = comb_prob - min(comb_prob); 
    norm_comb_prob = norm_comb_prob ./ max(norm_comb_prob(:));
    combined_prob_cell{1,i} = reshape(norm_comb_prob,[r c]); 
    %imshow(combined_prob_cell{1,i});
end


%color model confidence
color_model_confidence_cell = cell(1,s);
for i = 1:s
    d1 = bwdist(local_windows_mask_cell{1,i});
    d2 = bwdist(local_windows_mask_cell{1,i}==0);
    d = max(d1,d2);
    wc = exp(-d.^2./halfw.^2);
    numer_window = abs(local_windows_mask_cell{1,i} - combined_prob_cell{1,i}).*wc; 
    numer = sum(numer_window(:));
    denom = sum(wc(:));
    color_model_confidence_cell{1,i} = 1 - numer/denom;
end

%shape model
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


new_mask = imwarp(init_mask,transformation_cell{1,1});
imshow(new_mask);

%get average flow for each window
opticFlow = opticalFlowHS;
estimateFlow(opticFlow,rgb2gray(images_cell{1,1}));
flow = estimateFlow(opticFlow,rgb2gray(images_cell{1,2}));

VxWindows = get_local_windows(flow.Vx,local_windows_center_cell, halfw);
VyWindows = get_local_windows(flow.Vy,local_windows_center_cell, halfw);
local_windows_center_cell2 = cell(1,s);
for i = 1:s
    Vxmean = mean(VxWindows{1,i}(:));
    Vymean = mean(VyWindows{1,i}(:));
    local_windows_center_cell2{1,i} = uint32(local_windows_center_cell{1,i}+[Vxmean Vymean]);
end
local_windows_image_cell2 = get_local_windows(images_cell{1,2},local_windows_center_cell2, halfw);

%make new color models
[i s] = size(local_windows_image_cell2);
combined_prob_cell2 = cell(1,s);
for i = 1:s
    [r c] = find(bwdist(local_windows_mask_cell{1,i})>4);
    foreground_pix = impixel(local_windows_image_cell{1,i},r,c);
    foreground_model = fitgmdist(foreground_pix,1);
    
    inverted = local_windows_mask_cell{1,i}==0;
    [r c] = find(bwdist(inverted)>4);
    background_pix = impixel(local_windows_image_cell{1,i},r,c);
    background_model = fitgmdist(background_pix,1);
    
    % probs
    [r c ~] = size(local_windows_image_cell2{1,i});
    values = reshape(double(local_windows_image_cell2{1,i}),[r*c 3]);
    
    fore_prob = pdf(foreground_model,values);
    back_prob = pdf(background_model,values);
    
    comb_prob = fore_prob./(fore_prob+back_prob);
    norm_comb_prob = comb_prob - min(comb_prob); 
    norm_comb_prob = norm_comb_prob ./ max(norm_comb_prob(:));
    if sum(norm_comb_prob >0.5) > sum(combined_prob_cell2{1,i}(:)>0.5)
        combined_prob_cell2{1,i} = combined_prob_cell2{1,i};
    else
        combined_prob_cell2{1,i} = reshape(norm_comb_prob,[r c]); 
    end
end


function local_windows_center_cell = get_window_pos(init_mask,density)
    b = bwboundaries(init_mask); b = b{1,1}; %coords of edge of mask
    [h w] = size(b);
    local_windows_center_cell = cell(1,floor(h/density));
    for i = 1:floor(h/density)
        r  = b(i,1); c= b(i,2);
        local_windows_center_cell{1,i} = [r c];
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


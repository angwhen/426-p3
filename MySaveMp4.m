workingDir = '../Output';
shuttleVideo = VideoReader('shuttle.avi');
outputVideo = VideoWriter(fullfile(workingDir,'results1.avi'));
outputVideo.FrameRate = shuttleVideo.FrameRate;
open(outputVideo);
imageNames = dir(fullfile(workingDir,'images','%d.jpg'));
imageNames = {imageNames.name}';
for ii = 1:length(imageNames)
   img = imread(fullfile(workingDir,'images',imageNames{ii}));
   writeVideo(outputVideo,img);
end
close(outputVideo);
function [corrValMapAvgCenter, probCorrValMapAvgCenter, origMapAvgCenter, xposCrop, yposCrop] = ...
    filterPointingMaps( fnames, PLOT, nSweepPts, searchSigma)
% filter a series of time series scans to identify the most likely position
% of the source on the sky. 
% 
% INPUTS
% fnames: a cell array of the names of the mat files containing the
%   time-series scans of a source obtained using exportmat in loclib.py
%   ex: fnames = {'scan_59811', 'scan_59812', 'scan_59813', 'scan_59814', 'scan_59815', 'scan_59816'}; 
% PLOT: true if you want it to disply the plots, otherwise false. Default = true
% nSweepts: The density of the grid that you search for the center of the
%   source in each direction. Default = 60
% searchSigma: the standard deviation of the beam in radians. Default = 5.3328e-5/2.355 radians
%
% OUTPUT
% corrValMapAvgCenter: the average of the filtered maps (in voltage)
% probCorrValMapAvgCenter: the probablity of the most likely position of the source
% origMapAvgCenter: the average of the orignal maps
% xposCrop: the positions of the sampling points in the maps in the x-direction
% yposCrop: the positions of the sampling points in the maps in the y-direction

%% set defaults if not provided 

if(nargsin < 2)
    PLOT = true; 
end

if(nargsin < 3 ) 
    nSweepPts = 60; 
end

if(nargsin < 4 ); 
    searchSigma = 5.3328e-5/2.355;
end

sourceAmp = 1;
nFiles = length(fnames); 


% set the grid points for searching
gridSelector = 1; 
load(fnames{gridSelector});
minXSweep = min(x(:)) + 2e-5;
minYSweep = min(y(:)) + 2e-5;
maxXSweep = max(x(:)) - 2e-5;
maxYSweep = max(y(:)) - 2e-5;
xpos = linspace(minXSweep, maxXSweep, nSweepPts);
ypos = linspace(minYSweep, maxYSweep, nSweepPts); 

%% load data and estimate covariance from each time series

for f=1:nFiles
    
    % load a scan of data
    load(fnames{f});
    
    % decide if you want to look at polarization a (a) or polarization b (b)
    data = b;

    % linearly detrend the data
    data = detrend(data(:));
    nSamples = length(data);
    
    % compute the crossCorrelation for estimating the covariance matrix
    crossCorr = conv(data(:), flipud(data(:)));
    
    midIdx = length(crossCorr)/2 + 0.5;
    crossCorrSide = crossCorr(midIdx:end);
    
    warning('didnt normalize covariance...maybe should do this');
    %crossCorrNorm = conv(ones(size(data(:))), ones(size(data(:))));
    %crossCorrSide = crossCorrSide./crossCorrNorm(midIdx:end);
    
    % fill in the covariance matrix for this data run
    estCovarianceSingle = zeros(nSamples, nSamples);
    for i=1:nSamples
        for j=1:nSamples
            estCovarianceSingle(i,j) = crossCorrSide(abs(i-j)+1);
        end
    end
    
    % save the data, estimated covariance, and the length of the data
    timeSeries{f} = data; 
    estCov{f} = estCovarianceSingle;
    datalen(f) = length(data);
    
    
    % get grid locations and grid the sweep values onto those locations
    [ygrid, xgrid]= ndgrid(ypos , xpos);
    origMap(:,:,f) = griddata(x, y, data, xgrid, ygrid, 'linear');
    
    if PLOT
        
        FigHandle = figure(f);
        set(FigHandle, 'Position', [100, 100, 1049, 895]);
        
        % plot the sweep points along with the grid points
        subplot(241); plot(xgrid(:), ygrid(:), 'r+'); hold on; plot(x,y,'lineWidth', 2);
        title('Sweep Positions and Gridding Positions');
        xlabel('radians'); ylabel('radians'); 
        
        % plot the time stream
        subplot(242); plot(t,data); xlim([min(t) max(t)]); 
        title('Time Series Data');
        xlabel('Time (seconds)'); ylabel('Voltage'); 

        % plot the map
        subplot(223); imagesc(origMap(:,:,f));
        imagesc(origMap(:,:,f),'XData',xpos,'YData',ypos); 
        title('Original Map'); 
        ylabel('radians'); xlabel('radians'); 
    end
    
end

%% compute average covariance

% each data stream is of different length, make a covariance matrix that is
% the largest of the data streams
maxDataLen = max(datalen); 
estCovarianceAll = nan(maxDataLen, maxDataLen, length(fnames)); 

% place the indiviually estimated covariance into the matrix
for f=1:nFiles
    estCovarianceAll(1:datalen(f),1:datalen(f),f) = estCov{f}; 
end

% average the covariance matrices 
estCovariance = nanmean(estCovarianceAll, 3); 


%% match filter

for f=1:nFiles
    
    % load the file
    load(fnames{f});
    corrVals = nan(nSweepPts, nSweepPts);
    
    % invert the inverse covariance
    invEstCovariance = inv(estCovariance(1:datalen(f), 1:datalen(f)));
    
    iCount = 1;
    for i= ypos
        
        jCount = 1;
        for j= xpos
            
            % generate the ideal signal
            propReal = genSampPos(x, y, searchSigma, [i j], sourceAmp); 
            % compute the match filter
            h =  invEstCovariance * propReal;
            
            % normalization
            h = h * (1/(propReal'*invEstCovariance*propReal)); 
            
            % evaluate the match filter
            corrVals(iCount, jCount) = sum(timeSeries{f}(:).*h(:));
            
            jCount = jCount + 1;
        end
        
        %plot the filter response
        figure(f); 
        subplot(122); imagesc(corrVals,'XData',xpos,'YData',ypos); 
        title('Filtered Map'); xlabel('radians'); ylabel('radians'); colorbar; drawnow; 
        
        iCount = iCount + 1;
        
        % save the filtered map
        corrValMap(:,:,f) = corrVals; 
    end
    
end

%% prepare and return results

% remove negative snr's 
corrValMapZeroed = corrValMap;
corrValMapZeroed(corrValMap<0) = 0;

% average the maps
origMapAvg = nanmean(origMap, 3);
corrValMapAvg = nanmean(corrValMapZeroed, 3);
corrValMapSqAvg = nanmean(corrValMapZeroed.^2, 3);

% figure out the crops for removing 3 sigma
deltaX = (maxXSweep - minXSweep)./nSweepPts;
deltaY = (maxYSweep - minYSweep)./nSweepPts;
nPixelCropX  = ceil( (3*searchSigma)./deltaX ) + 1;
nPixelCropY  = ceil( (3*searchSigma)./deltaY ) + 1;

% crop the maps
xposCrop = xpos(nPixelCropX:end-nPixelCropX);
yposCrop = ypos(nPixelCropY:end-nPixelCropY);
origMapAvgCenter = origMapAvg(nPixelCropY:end-nPixelCropY,nPixelCropX:end-nPixelCropX);
corrValMapAvgCenter = corrValMapAvg(nPixelCropY:end-nPixelCropY,nPixelCropX:end-nPixelCropX);
probCorrValMapAvgCenter = exp(0.5*corrValMapSqAvg(nPixelCropY:end-nPixelCropY,nPixelCropX:end-nPixelCropX);

% plot results
if PLOT

    FigHandle = figure();
    set(FigHandle, 'Position', [100, 100, 1300, 300]);
    
    % dispaly estimated covariance
    subplot(141); imagesc(estCovariance); title('Estimated Covariance'); axis square;
    
    % display average map
    subplot(142); imagesc(origMapAvgCenter,'XData',xposCrop,'YData',yposCrop);
    title('Averaged Original Maps'); xlabel('radians'); ylabel('radians'); axis square;
    
    % display average filtered map
    subplot(143); imagesc(corrValMapAvgCenter,'XData',xposCrop,'YData',yposCrop);
    title('Averaged Filtered Maps');  xlabel('radians'); ylabel('radians'); axis square;
    
    % display average map corresponding to probablity
    subplot(144); imagesc(probCorrValMapAvgCenter,'XData',xposCrop,'YData',yposCrop);
    title('Probablity Averaged Filtered Maps'); xlabel('radians'); ylabel('radians'); axis square; 
end


end


function [sampReal] = genSampPos(X, Y, sigma, center, amp)

sampReal =amp*exp(-((Y-center(1)).^2+(X-center(2)).^2)/(2*sigma.^2));
sampReal = sampReal(:); 

end


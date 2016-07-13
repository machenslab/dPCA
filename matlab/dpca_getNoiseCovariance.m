function Cnoise = dpca_getNoiseCovariance(Xfull, Xtrial, numOfTrials, varargin)

options = struct('simultaneous', false, ...
                 'type',         'pooled');

% read input parameters
optionNames = fieldnames(options);
if mod(length(varargin),2) == 1
	error('Please provide propertyName/propertyValue pairs')
end
for pair = reshape(varargin,2,[])    % pair is {propName; propValue}
	if any(strcmp(pair{1}, optionNames))
        options.(pair{1}) = pair{2};
    else
        error('%s is not a recognized parameter name', pair{1})
	end
end

if ~options.simultaneous
    SSnoise = nansum(Xtrial.^2, ndims(Xtrial)) - bsxfun(@times, Xfull.^2, numOfTrials);
    
    SSnoiseSumOverT = sum(SSnoise, ndims(SSnoise));
    
    numOfTrialsAverage = numOfTrials(:,:);
    numOfTrialsAverage(numOfTrialsAverage==0) = nan;
    numOfTrialsAverage = nanmean(numOfTrialsAverage, 2);
    
    if strcmp(options.type, 'pooled')
        Cnoise = diag(bsxfun(@times, nansum(SSnoiseSumOverT(:,:),2), 1./numOfTrialsAverage));
    elseif strcmp(options.type, 'averaged')
        Cnoise = diag(nansum(bsxfun(@times, SSnoiseSumOverT(:,:), 1./numOfTrials(:,:)),2));
    end
else
    if strcmp(options.type, 'pooled')
        Xnoise = bsxfun(@minus, Xtrial, Xfull);
        Xnoise = Xnoise(:,:);
        Xnoise = Xnoise(:, ~isnan(Xnoise(1,:)));
        SSnoise = Xnoise*Xnoise';
        Cnoise = SSnoise/size(Xnoise,2);
        dims = size(Xfull);
        dimProd = prod(dims(2:end));
        Cnoise = Cnoise * dimProd;
    elseif strcmp(options.type, 'averaged')
        display('Averaged noise covariance computation is not yet implemented')
        display('for the simultaneously recorded data. Returning the pooled')
        display('noise covariance matrix instead.')
    end
end
function [SSnoise, CnoisePooled, CnoiseAveraged] = dpca_getNoiseCovariance(Xfull, Xtrial, numOfTrials)

SSnoise = nansum(Xtrial.^2, ndims(Xtrial)) - bsxfun(@times, Xfull.^2, numOfTrials);      

SSnoiseSumOverT = sum(SSnoise, ndims(SSnoise));

numOfTrialsAverage = numOfTrials(:,:);
numOfTrialsAverage(numOfTrialsAverage==0) = nan;
numOfTrialsAverage = nanmean(numOfTrialsAverage, 2);

CnoisePooled = diag(bsxfun(@times, nansum(SSnoiseSumOverT(:,:),2), 1./numOfTrialsAverage));
CnoiseAveraged = diag(nansum(bsxfun(@times, SSnoiseSumOverT(:,:), 1./numOfTrials(:,:)),2));


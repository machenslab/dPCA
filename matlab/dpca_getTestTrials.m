function Xtest = dpca_getTestTrials(firingRatesPerTrial, numOfTrials)

dim = size(firingRatesPerTrial);

neuronsConditions = numOfTrials(:);
testTrials = ceil(rand([length(neuronsConditions) 1]) .* neuronsConditions);

ind = reshape(testTrials, size(numOfTrials));
ind = bsxfun(@times, ones(dim(1:end-1)), ind);
ind = ind(:);
    
Xtest = firingRatesPerTrial(sub2ind([prod(dim(1:end-1)) dim(end)], (1:prod(dim(1:end-1)))', ind));
Xtest = reshape(Xtest, dim(1:end-1));
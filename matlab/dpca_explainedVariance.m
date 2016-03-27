function explVar = dpca_explainedVariance(Xfull, W, V, varargin)

% explVar = dpca_explainedVariance(X, W, V) computes various measures of
% explained variance and returns them in a structure explVar. X is the data
% matrix, W is the decoder matrix, V is the encoder matrix. Returned values:
%
%  * explVar.totalVar             - total variance
%  * explVar.totalMarginalizedVar - total variance in each marginalization
%  * explVar.componentVar         - variance of each component (%)
%  * explVar.margVar              - variance of each component in each marginalization (%)
%  * explVar.cumulativePCA        - cumulative variance of the PCA components (%)
%  * explVar.cumulativeDPCA       - cumulative variance of the dPCA components (%)
%
% [...] = dpca(..., 'PARAM1',val1, 'PARAM2',val2, ...) 
% specifies optional parameter name/value pairs:
%
%  'combinedParams' - cell array of cell arrays specifying 
%                     which marginalizations should be added up together,
%                     e.g. for the three-parameter case with parameters
%                           1: stimulus
%                           2: decision
%                           3: time
%                     one could use the following value:
%                     {{1, [1 3]}, {2, [2 3]}, {3}, {[1 2], [1 2 3]}}.
%
%  'X_trial'        - array of single trials. Has one extra dimension as 
%                     compared with X and stores individual single trial
%                     firing rates, as opposed to the trial average. If
%                     provided, "signal variance" will be computed:
%
%  * explVar.totalVar_signal             - total signal variance
%  * explVar.totalVar_noise              - total residual noise variance
%  * explVar.totalMarginalizedVar_signal - total signal variance in each marginalization
%  * explVar.cumulativePCA_signal        - cumulative signal variance of the PCA components (%)
%  * explVar.cumulativeDPCA_signal       - cumulative signal variance of the dPCA components (%)
%
%  'numOfTrials'    - must be provided together with X_trial. Has one
%                     dimension fewer than X and for each neuron and
%                     combination of parameters (without time) specifies
%                     the number of available trials in X_trial. All
%                     entries have to be larger than 1.
%
% 'Cnoise'          - Cnoise matrix, as obtained by
%                     dpca_getNoiseCovariance(). Can be provided INSTEAD of
%                     X_trial to compute the noise estimate via the new
%                     method. numOfTrials still needed.


% default input parameters
options = struct('combinedParams', [], ...   
                 'X_trial',        [], ...
                 'numOfTrials',    [], ...
                 'Cnoise',         []);

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

% centering
X = Xfull(:,:);
Xfull = bsxfun(@minus, Xfull, mean(X,2));
X = bsxfun(@minus, X, mean(X,2));

% marginalizing
Xmargs = dpca_marginalize(Xfull, 'combinedParams', options.combinedParams, 'ifFlat', 'yes');

% total variance
explVar.totalVar = sum(sum(X.^2));

% total marginalized variance
for i=1:length(Xmargs)
    explVar.totalMarginalizedVar(i) = sum(Xmargs{i}(:).^2);
end

% variance of each component ("captured variance"! not the same as 
% "explained variance". Deprecated and not used anymore.)

% for i=1:length(Xmargs)
%     explVar.margVar(i,:) = sum((W' * Xmargs{i}).^2, 2)' / explVar.totalVar * 100;
% end
% explVar.componentVar = sum(explVar.margVar);

% PCA explained variance
[~,S,Wpca] = svd(X', 'econ');
S = diag(S);
S = S(1:size(W,2));
explVar.cumulativePCA = cumsum(S.^2'/ explVar.totalVar * 100);

% dPCA explained variance
Z = W'*X;
for i=1:size(W,2)
    explVar.cumulativeDPCA(i) = 100 - sum(sum((X - V(:,1:i)*Z(1:i,:)).^2)) / explVar.totalVar * 100;    
    explVar.componentVar(i) = 100 - sum(sum((X - V(:,i)*Z(i,:)).^2)) / explVar.totalVar * 100;    
   
    for j=1:length(Xmargs)
        ZZ = Xmargs{j} - V(:,i)*(W(:,i)'*Xmargs{j});
        explVar.margVar(j,i) = (explVar.totalMarginalizedVar(j) - sum(ZZ(:).^2)) / explVar.totalVar * 100;    
    end
end

% OPTIONAL part : OLD APPROACH
if ~isempty(options.X_trial) && ~isempty(options.numOfTrials)
    
    % subtract two trials in each condition to get a sample of noise data
    dims = size(options.X_trial);
    neuronsConditions = options.numOfTrials(:);
    
    ind1 = zeros(1, length(neuronsConditions));
    ind2 = zeros(1, length(neuronsConditions));
    for i=1:length(neuronsConditions)
        tmp = randperm(neuronsConditions(i), 2);
        ind1(i) = tmp(1);
        ind2(i) = tmp(2);
    end
    ind1 = reshape(ind1, size(options.numOfTrials));
    ind2 = reshape(ind2, size(options.numOfTrials));
    ind1 = bsxfun(@times, ones(dims(1:end-1)), ind1);
    ind2 = bsxfun(@times, ones(dims(1:end-1)), ind2);
    ind1 = ind1(:);
    ind2 = ind2(:);
    
    dif = options.X_trial(sub2ind([prod(dims(1:end-1)) dims(end)], (1:prod(dims(1:end-1)))', ind1)) - ...
          options.X_trial(sub2ind([prod(dims(1:end-1)) dims(end)], (1:prod(dims(1:end-1)))', ind2));
    
    % scale appropriately
    XfullNN = bsxfun(@times, ones(dims(1:end-1)), options.numOfTrials);
    dif = dif ./ sqrt(2*XfullNN(:));
    XnoiseFull = reshape(dif, dims(1:end-1));
    
    % process noise data
    Xnoise = XnoiseFull(:,:);
    XnoiseFull = bsxfun(@minus, XnoiseFull, mean(Xnoise,2));
    Xnoise = bsxfun(@minus, Xnoise, mean(Xnoise,2));
    XmargsNoise = dpca_marginalize(XnoiseFull, 'combinedParams', options.combinedParams);

    % total noise variance
    explVar.totalVar_noise = sum(sum(Xnoise.^2));
    
    % total signal variance
    explVar.totalVar_signal = explVar.totalVar - explVar.totalVar_noise;
    
    % total marginalized signal variance
    for i=1:length(Xmargs)
        marginalizedVarNoise(i) = sum(XmargsNoise{i}(:).^2);
    end
    explVar.totalMarginalizedVar_signal = explVar.totalMarginalizedVar - marginalizedVarNoise;
    
%     % PCA explained SIGNAL variance
%     [~,Snoise,~] = svd(Xnoise');
%     Snoise = diag(Snoise);
%     Snoise = Snoise(1:size(W,2));
%     pcaSignal = S.^2 - Snoise.^2;
%     explVar.cumulativePCA_signal = cumsum(pcaSignal' / explVar.totalVar_signal * 100);
%     
%     % dPCA explained SIGNAL variance
%     Z = W'*X;
%     for i=1:size(W,2)
%         dpcaVar(i) = explVar.totalVar - sum(sum((X - V(:,1:i)*Z(1:i,:)).^2)) - sum(Snoise(1:i).^2);
%     end
%     explVar.cumulativeDPCA_signal = dpcaVar / explVar.totalVar_signal * 100;
end

% OPTIONAL part : NEW APPROACH
if ~isempty(options.Cnoise) && ~isempty(options.numOfTrials)
    Ktilde = mean(reshape(options.numOfTrials, size(options.numOfTrials,1),[]),2);
    explVar.totalVar_noise = sum(diag(options.Cnoise)./Ktilde);
    explVar.totalVar_signal = explVar.totalVar - explVar.totalVar_noise;
    
    % dirty hack!! do it properly later!
    S = size(Xfull,2);
    Q = size(Xfull,3);
    T = size(Xfull,4);
    explVar.totalMarginalizedVar_signal = explVar.totalMarginalizedVar - ...
        [(S*T-T) (Q*T-T) (T-1) S*Q*T-S*T-Q*T+T]/(S*Q*T-1) * explVar.totalVar_noise;

    % X = randn(1000,S,Q,T);
    % Xmarg = dpca_marginalize(X, 'combinedParams', combinedParams);
    % [sum(Xmarg{1}(:).^2) sum(Xmarg{2}(:).^2) sum(Xmarg{3}(:).^2) sum(Xmarg{4}(:).^2)]/sum(X(:).^2)
end

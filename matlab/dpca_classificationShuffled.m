function [accuracyShuffle, brierShuffle] = dpca_classificationShuffled(Xtrial, numOfTrials, varargin)

% accuracyShuffle = dpca_classificationShuffled(Xtrial, numOfTrials)
% performs repeated shuffling to sample from the null distribution of the
% cross-validated classification accuracies. Xtrial is an array storing
% single trials, numOfTrials specifies the number of available trials, see
% dpca_classificationAccuracy() for description.
%
% This function assumes that time parameter is stored in the last dimension of
% X. For datasets without time, some other cross-validation needs to be
% used.
%
% [accuracyShuffle, brierShuffle] = dpca_classificationShuffled(X) returns
% Brier scores together with classification accuracy.
%
% [...] = dpca_classificationShuffled(..., 'PARAM1',val1, 'PARAM2',val2, ...) 
% specifies optional parameter name/value pairs:
%
% 'combinedParams'  - cell array of cell arrays specifying 
%                     which marginalizations should be added up together,
%                     e.g. for the three-parameter case with parameters
%                           1: stimulus
%                           2: decision
%                           3: time
%                     one could use the following value:
%                     {{1, [1 3]}, {2, [2 3]}, {3}, {[1 2], [1 2 3]}}.
%
%  'lambda'         - regularization parameter. It's going to be multiplied
%                     by the total variance of Xfull. Default value is
%                     zero. To use different lambdas for different
%                     marginalizations, provide an array instead of one
%                     number.
%
%  'numRep'         - number of cross-validation iterations. Default: 100
%
%  'numShuffles'    - number of shuffles. Default: 100
%
%  'decodingClasses'- specifies classes for each marginalization.
%                     E.g. for the three-parameter case with parameters
%                           1: stimulus  (3 values)
%                           2: decision  (2 values)
%                           3: time
%                     and combinedParams as specified above:
%                     {{1, [1 3]}, {2, [2 3]}, {3}, {[1 2], [1 2 3]}}
%                     one could use the following decodingClasses:
%                     {[1 1; 2 2; 3 3], [1 2; 1 2; 1 2], [], [1 2; 3 4; 5 6]}
%                     Default value is to use separate class for each
%                     condition, i.e.
%                     {[1 2; 3 4; 5 6], [1 2; 3 4; 5 6], [], [1 2; 3 4; 5 6]}
%
%  'verbose'        - If 'yes' (default) progress status is displayed. Set
%                     to 'no' to turn it off.
%
%  'filename'       - If provided, accuracy and brier outputs will be saved
%                     in this file.
%
%  'timeSplits'     - an array of K integer numbers specifying time splits
%                     for time period splitting. All marginalizations will
%                     be additionally split into K+1 marginalizations,
%                     apart from the one corresponding to the last
%                     parameter (which is assumed to be time).
%
% 'timeParameter'   - is only used together with 'timeSplits', and must be
%                     provided. Specifies the time parameter. In the
%                     example above it is equal to 3.
%
% 'notToSplit'      - is only used together with 'timeSplits'. A cell array
%                     of cell arrays specifying which marginalizations
%                     should NOT be split. If not provided, all
%                     marginalizations will be split.
%
% 'simultaneous'    - if the dataset is simultaneously recorded (true) or
%                     not (false - DEFAULT)
%
% 'noiseCovType'    - two possible ways to compute noise covariance matrix:
%                        'averaged'   - average over conditions
%                        'pooled'     - pooled over conditions (DEFAULT)


% default input parameters
options = struct('numShuffles',    100,                   ...  
                 'lambda',         0,                   ...
                 'numRep',         100,                 ...
                 'verbose',        'yes',               ...
                 'combinedParams', [],                  ...
                 'decodingClasses', [],                 ...
                 'timeSplits',      [],                 ...
                 'timeParameter',  [],                  ...
                 'notToSplit',     [],                  ...
                 'filename',       [],                  ...
                 'simultaneous',   false,               ...
                 'noiseCovType',   'pooled');
             
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

tic

if strcmp(options.verbose, 'yes')
    display('Preprocessing...')
end

D = size(numOfTrials,1);
dim = size(Xtrial);
T = dim(end-1);
maxTrialN = size(Xtrial, ndims(Xtrial));
numCond = prod(dim(2:end-2));

% find missing trials
numOfTrialsCond = reshape(numOfTrials, D, []);
trialsMissing = zeros(D, maxTrialN*numCond);
for n=1:D
    this = zeros(numCond, maxTrialN);
    for c=1:numCond
        this(c,:) = [zeros(1,numOfTrialsCond(n,c)) ones(1,maxTrialN-numOfTrialsCond(n,c))];
    end
    trialsMissing(n,:) = this(:);
end

% collapsing conditions
orderDim = 1:ndims(Xtrial);
orderDim(end-1:end) = orderDim([end end-1]);
XtrialCond = permute(Xtrial, orderDim); % time shifted to the end
XtrialCond = reshape(XtrialCond, D, [], T);

for shuffle = 1:options.numShuffles
    if strcmp(options.verbose, 'yes')
        repTic = tic;
        fprintf(['Shuffle #' num2str(shuffle) ' out of ' num2str(options.numShuffles) ': shuffling... '])
    end
        
    % shuffling PSTHs inside each neuron (preserving time and numOfTrials)
    XtrialCondShuffle = zeros(size(XtrialCond));
    if ~options.simultaneous
        for n = 1:D
            presentTrials = find(trialsMissing(n,:) == 0);
            shuffledOrder = presentTrials(randperm(length(presentTrials)));
            XtrialCondShuffle(n,presentTrials,:) = ...
                XtrialCond(n,shuffledOrder,:);
        end
    else
        presentTrials = find(trialsMissing(1,:) == 0);
        shuffledOrder = presentTrials(randperm(length(presentTrials)));
        XtrialCondShuffle(:,presentTrials,:) = ...
            XtrialCond(:,shuffledOrder,:);
    end
    XtrialShuffle = permute(reshape(XtrialCondShuffle, dim(orderDim)), ...
        orderDim);
    clear XtrialCondShuffle

    firingRatesAverageShuffle = sum(XtrialShuffle, ndims(XtrialShuffle));
    firingRatesAverageShuffle = bsxfun(@times, firingRatesAverageShuffle, 1./numOfTrials);
    
    if strcmp(options.verbose, 'yes')
        fprintf('cross-validating')
        verbosity = 'dots';
    else
        verbosity = 'no';
    end
    
    [accuracy, brier] = dpca_classificationAccuracy(firingRatesAverageShuffle, XtrialShuffle, numOfTrials, ...
        'numComps', 1, ...
        'combinedParams', options.combinedParams, ...
        'lambda', options.lambda, ...
        'numRep', options.numRep, ...
        'decodingClasses', options.decodingClasses,  ...
        'verbose', verbosity,  ...
        'timeSplits', options.timeSplits, ...
        'timeParameter', options.timeParameter, ...
        'notToSplit', options.notToSplit, ...
        'simultaneous', options.simultaneous, ...
        'noiseCovType', options.noiseCovType);
    
    accuracyShuffle(:,:,shuffle) = squeeze(accuracy(:,1,:));
    brierShuffle(:,:,shuffle) = squeeze(brier(:,1,:));
    
    if strcmp(options.verbose, 'yes')
        repTime = toc(repTic);
        fprintf([' [' num2str(round(repTime)) ' s]\n'])
    end
    clear XtrialShuffle firingRatesAverageShuffle
    
    if ~isempty(options.filename)
        if exist(options.filename, 'file')
            save(options.filename, 'accuracyShuffle', 'brierShuffle', '-append')
        else
            save(options.filename, 'accuracyShuffle', 'brierShuffle')
        end
    end
end

timeTaken = toc;
if ~isempty(options.filename)
    save(options.filename, 'timeTaken', '-append')
end

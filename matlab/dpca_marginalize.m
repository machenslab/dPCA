function [YY, margNums] = dpca_marginalize(X, varargin)

% YY = dpca_marginalize(X) computes data marginalized over all combinations
% of parameters. X is a multi-dimensional array of dimensionality D+1, where
% first dimension corresponds to neurons and the rest D dimensions --
% to various parameters. YY is a cell array of marginalized datasets,
% containing 2^D-1 arrays, marginalized over all combinations of D
% parameters, excluding empty set. For each i size(YY{i}) equals size(X).
%
% [...] = dpca_marginalize(..., 'PARAM1',val1, 'PARAM2',val2, ...) 
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
%  'ifFull'         - can be "yes" or "no" [default is "yes"] and specifies
%                     if the size of YY{i} should be equal to size(X) or
%                     reduced as a result of averaging some parameters out
%
%  'ifFlat'         - can be "yes" or "no" [default is "no"] and specifies
%                     if each YY{i} should be flattened to the 2D shape:
%                     first dimension (neurons) stays the same, all the
%                     others are pooled together: YY{i} = YY{i}(:,:);
%
%  'timeSplits'     - an array of K integer numbers specifying time splits
%                     for time period splitting. All marginalizations will
%                     be additionally split into K+1 marginalizations,
%                     apart from the one corresponding to the last
%                     parameter (which is assumed to be time).
%                       
%                     [YY, margNums] = marginalize(...) returns an
%                     additional variable margNums, an array with integers
%                     specifying the 'type' of each marginalization in YY,
%                     i.e. the number of the corresponding parameter
%                     combination. If timeSplits is not used then margNums
%                     is simply 1:length(YY). But if timesSplits is used,
%                     then individual marginalizations are further
%                     splitted, and margNums tells which of the splits
%                     belong to one original marginalization.
%
% 'timeParameter'   - is only used together with 'timeSplits', and must be
%                     provided. Specifies the time parameter. In the
%                     example above it is equal to 3.
%
% 'notToSplit'      - is only used together with 'timeSplits'. A cell array
%                     of cell arrays specifying which marginalizations
%                     should NOT be split. If not provided, all
%                     marginalizations will be split.


% default input parameters
options = struct('combinedParams', [],      ...   
                 'ifFull',         'yes',   ...
                 'ifFlat',         'no',    ...
                 'timeSplits',     [],      ...
                 'timeParameter',  [],      ...
                 'notToSplit',     []);

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

% mean zero
X = bsxfun(@minus, X, mean(X(:,:),2));

% all parameter combinations
params = 2:length(size(X));
paramsubsets = subsets(params);

% the core computation
alreadyProcessed = containers.Map;
for subs = 1:length(paramsubsets)
    indRest = paramsubsets{subs};
    indMarg = setdiff(params, indRest);
    YY{subs} = nanmmean(X, indMarg);
    alreadyProcessed(num2str(sort(indMarg))) = subs;
    
    % subtracting all marginalizations from the subsets
    paramsubsubsets = subsets(indRest);
    for s = 1:length(paramsubsubsets)-1
        indM = [paramsubsubsets{s} indMarg];
        YY{subs} = bsxfun(@plus, YY{subs}, -YY{alreadyProcessed(num2str(sort(indM)))});
    end
end

paramsubsets = subsets(params-1);
margNums = 1:length(YY);

% combining some marginalizations together (OPTIONAL)
if ~isempty(options.combinedParams)
    for i=1:length(options.combinedParams)
        margsToAdd = [];
        for j=1:length(options.combinedParams{i})
            for k=1:length(paramsubsets)
                if length(paramsubsets{k}) == length(options.combinedParams{i}{j}) ...
                   && all(sort(paramsubsets{k}) == sort(options.combinedParams{i}{j}))
                    margsToAdd = [margsToAdd k];
                    continue
                end
            end
        end
        
        YYY{i} = YY{margsToAdd(1)};
        for j=2:length(margsToAdd)
            YYY{i} = bsxfun(@plus, YYY{i}, YY{margsToAdd(j)});
        end
    end
    
    YY = YYY;
    paramsubsets = options.combinedParams;
    margNums = 1:length(options.combinedParams);
end

% making all marginalizations the same size as X (OPTIONAL)
if strcmp(options.ifFull, 'yes')
    for i=1:length(YY)
        YY{i} = bsxfun(@times, YY{i}, ones(size(X)));
    end
end

% splitting marginalizations into time periods (OPTIONAL)
if ~isempty(options.timeSplits)
    ZZ = {};
    margNums = [];
    
    timeSplitsBeg = [1 options.timeSplits];
    timeSplitsEnd = [options.timeSplits-1 size(X, length(size(X)))];
    
    % using the same format for paramsubsets as in options.combinedParams
    for k=1:length(paramsubsets)
        if ~iscell(paramsubsets{k})
            paramsubsets{k} = {paramsubsets{k}};
        end
    end
    
    for i=1:length(YY)
        % skip marginalizations that do not have to be split (OPTIONAL)
        if ~isempty(options.notToSplit)
            toSkip = 0;
            for k = 1:length(options.notToSplit)
                if length(paramsubsets{i}) == length(options.notToSplit{k})
                    identical = 1;
                    for zz = 1:length(paramsubsets{i})
                        found = 0;
                        for zzz = 1:length(options.notToSplit{k})
                            if length(paramsubsets{i}{zz}) == length(options.notToSplit{k}{zzz}) ...
                               && all(sort(paramsubsets{i}{zz}) == sort(options.notToSplit{k}{zzz}))
                                found = 1;
                                break
                            end
                        end
                        if ~found
                            identical = 0;
                            break
                        end
                    end
                            
                    if identical
                        toSkip = 1;
                        break
                    end
                end
            end
            if toSkip
                ZZ{end+1} = YY{i};
                margNums(end+1) = i;
                continue
            end
        end
        
        ind = {};
        for a=1:length(size(X))-1
            ind{a} = ':';
        end
        
        for split = 1:length(options.timeSplits)+1
            ind{1+options.timeParameter} = timeSplitsBeg(split):timeSplitsEnd(split);
            ZZ{end+1} = zeros(size(YY{i}));
            ZZ{end}(ind{:}) = YY{i}(ind{:});
            margNums(end+1) = i;
        end   
    end
    
    YY = ZZ;
end

% flattening (OPTIONAL)
if strcmp(options.ifFlat, 'yes')
    for i = 1:length(YY)
        YY{i} = YY{i}(:,:);
    end
end

%%%%%%%%%%%%%% HELPER FUNCTIONS %%%%%%%%%%%%%%%%%%%

function S = subsets(X)

% S = subsets(X) returns a cell array of all subsets of vector X apart
% from the empty set. Subsets are ordered by the number of elements in
% ascending order.
%
% subset([1 2 3]) = {[1], [2], [3], [1 2], [1 3], [2 3], [1 2 3]}

d = length(X);
pc = dec2bin(1:2^d-1) - '0';
[~, ind] = sort(sum(pc, 2));
pc = fliplr(pc(ind,:));
for i=1:length(pc)
    S{i} = X(find(pc(i,:)));
end

%%%%%%%%%%%%%%

function Y = nanmmean(X, dimlist)

% Y = nanmmean(X, DIMLIST) computes the average over pooled dimensions
% specified in DIMLIST ignoring NaN values. Y has the same dimensionality
% as X.

if isempty(dimlist)
    Y = X;
    return
end

dims = size(X);
dimrest = setdiff(1:length(dims), dimlist);

X = permute(X, [dimrest dimlist]);
X = reshape(X, [dims(dimrest) prod(dims(dimlist))]);
X = nanmean(X, length(size(X)));
X = ipermute(X, [dimrest dimlist]);

Y = X;


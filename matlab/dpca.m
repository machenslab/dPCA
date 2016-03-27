function [W, V, whichMarg] = dpca(Xfull, numComps, varargin)

% [W, V, whichMarg] = dpca(X, numComp, ...) performs dPCA on the data in X
% and returns decoder matrix W and encoder matrix V. X is a multi-dimensional
% array of dimensionality D+1, where first dimension corresponds to N neurons 
% and the rest D dimensions -- to various parameters. numComp specifies
% the number of dPCA components to be extracted (can be either one number
% of a list of numbers for each marginalization). whichMarg is an array of
% integers providing the 'type' of each component (which marginalization it
% describes). If the total number of required components is S=sum(numComp),
% then W and V are of NxS size, and whichMarg has length S.
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
%  'lambda'         - regularization parameter. It's going to be multiplied
%                     by the total variance of Xfull. Default value is
%                     zero. To use different lambdas for different
%                     marginalizations, provide an array instead of one
%                     number.
%
%  'order'          - can be 'yes' (default) or 'no' and specifies whether
%                     the components should be ordered by decreasing 
%                     variance. If length(numComp)==1, components will
%                     always be sorted.
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
% 'scale'           - if 'yes', decoder of each component will be scaled to
%                     have an optimal length (leading to the minimal
%                     reconstruction error). Default is 'no'.
%
% 'Cnoise'          - if provided, will be used in a cost function to
%                     penalize captured noise variance

% default input parameters
options = struct('combinedParams', [],       ...   
                 'lambda',         0,        ...
                 'order',          'yes',    ...
                 'timeSplits',     [],       ...
                 'timeParameter',  [],       ...
                 'notToSplit',     [],       ...
                 'scale',          'no',     ...
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
X = bsxfun(@minus, X, mean(X,2));
XfullCen = reshape(X, size(Xfull));

% total variance
totalVar = sum(X(:).^2);

% marginalize
[Xmargs, margNums] = dpca_marginalize(XfullCen, 'combinedParams', options.combinedParams, ...
                    'timeSplits', options.timeSplits, ...
                    'timeParameter', options.timeParameter, ...
                    'notToSplit', options.notToSplit, ...
                    'ifFlat', 'yes');

% initialize
decoder = [];
encoder = [];
whichMarg = [];

% noise covariance
if isempty(options.Cnoise)
    options.Cnoise = zeros(size(X,1));
end

% loop over marginalizations
for i=1:length(Xmargs)
    if length(numComps) == 1
        nc = numComps;
    else
        nc = numComps(margNums(i));
    end
    
    if length(options.lambda) == 1
        thisLambda = options.lambda;
    else
        thisLambda = options.lambda(margNums(i));
    end
    
    if nc == 0
        continue
    end
    
    % catching possible warning
    s1 = warning('error','MATLAB:singularMatrix');
    s2 = warning('error','MATLAB:nearlySingularMatrix');
    try
        C = Xmargs{i}*X'/(X*X' + options.Cnoise + (totalVar*thisLambda)^2*eye(size(X,1)));
    catch exception
        display('Matrix close to singular, using tiny regularization, lambda = 1e-10')
        thisLambda = 1e-10;
        C = Xmargs{i}*X'/(X*X' + options.Cnoise + (totalVar*thisLambda)^2*eye(size(X,1)));
    end
    warning(s1)
    warning(s2)
    
    M = C*X;
    [U,~,~] = eigs(M*M', nc);
    P = U;
    D = U'*C;
    
    if strcmp(options.scale, 'yes')
        for uu = 1:size(D,1)
            A = Xmargs{i};
            B = P(:,uu)*D(uu,:)*X;
            scalingFactor = (A(:)'*B(:))/(B(:)'*B(:));
            D(uu,:) = scalingFactor * D(uu,:);
        end
    end
    
    decoder = [decoder; D];
    encoder = [encoder P];    
    whichMarg = [whichMarg i*ones(1, nc)];
end

% transposing
V = encoder;
W = decoder';

% flipping axes such that all encoders have more positive values
toFlip = find(sum(sign(V))<0);
W(:, toFlip) = -W(:, toFlip);
V(:, toFlip) = -V(:, toFlip);

% if there were timeSplits, join the components from one marginalization
% together (ordering by variance)
if ~isempty(options.timeSplits)
    toKeep = [];
    for i=1:max(margNums)
        components = find(ismember(whichMarg, find(margNums==i)));
        
        Z = W(:,components)'*X;
        explVar = sum(Z.^2,2);
        [~, order] = sort(explVar, 'descend');
        
        if length(numComps) == 1
            nc = numComps;
        else
            nc = numComps(i);
        end
        
        toKeep = [toKeep components(order(1:nc))];
    end
    W = W(:, toKeep);
    V = V(:, toKeep);
    whichMarg = whichMarg(toKeep);
    whichMarg = margNums(whichMarg);
end

% ordering components by explained variance (or not)
if length(numComps) == 1 || strcmp(options.order, 'yes')
    
    % The next two lines would order the components based on the "captured"
    % variance, not "explained" variance. We used to do it in earlier
    % drafts, but switched to the explained variance in the final version.
    
%     Z = W'*X;
%     explVar = sum(Z.^2,2);

    for i=1:size(W,2)
        Z = X - V(:,i)*(W(:,i)'*X);
        explVar(i) = 1 - sum(Z(:).^2)/totalVar;
    end
    [~ , order] = sort(explVar, 'descend');
    
    if length(numComps) == 1
        L = numComps;
    else
        L = sum(numComps);
    end
    
    W = W(:, order(1:L));
    V = V(:, order(1:L));
    whichMarg = whichMarg(order(1:L));
end
    

function dpca_perMarginalization(Xfull, plotFunction, varargin)

% dpca_perMarginalization(X, plotFunction, ...) performs PCA in each
% marginalization of X and plots the components using plotFunction, a
% pointer to the function that plots one component (see dpca_plot_default() for
% the template).

% dpca_perMarginalization(..., 'PARAM1',val1, 'PARAM2',val2, ...) 
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
% 'timeEvents'      - time-points that should be marked on each subplot
%  'marginalizationNames'   - names of each marginalization
%  'time'                   - time axis
%
% 'timeSplits'      - an array of K integer numbers specifying time splits
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


% default input parameters
options = struct('combinedParams', [],       ...   
                 'timeEvents',     [],       ...
                 'time',           [], ...   
                 'marginalizationNames', [], ...
                 'timeSplits',     [],       ...
                 'timeParameter',  [],       ...
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

PCs = [];
vars = [];
margs = [];

ncompsPerMarg = 3;

for m=1:length(Xmargs)
    %[~,S,V] = svd(Xmargs{m});      % this is very slow!
    margVar(m) = sum(Xmargs{m}(:).^2)/totalVar*100;
    
    %tic
    XX = Xmargs{m}*Xmargs{m}';
    [U,S] = eig(XX);
    S = diag(sqrt(fliplr(diag(S)')));
    U = fliplr(U);
    SV = U'*Xmargs{m};
    %toc
    
    %PCs = [PCs; S(1:10,1:10)*V(:,1:10)'];
    PCs = [PCs; SV(1:ncompsPerMarg,:)];
    vars = [vars; diag(S(1:ncompsPerMarg,1:ncompsPerMarg)).^2];
    margs = [margs repmat(m, [1 ncompsPerMarg])];
end
[vars,ind] = sort(vars,'descend');
PCs = PCs(ind,:);
margs = margs(ind);
%PCs = PCs(1:15,:);
vars = vars / totalVar * 100;

dims = size(Xfull);
Z = reshape(PCs, [length(ind) dims(2:end)]);

yspan = max(abs(Z(:)));

figure
N = min(length(Xmargs)*ncompsPerMarg, 35);
for i=1:N
    subplot(floor(sqrt(N)),ceil(N/floor(sqrt(N))),i)
    
    cln = {i};
    for j=2:ndims(Z)
        cln{j} = ':';
    end
    
    plotFunction(Z(cln{:}), options.time, [-yspan yspan]*1.1, vars(i), i, options.timeEvents, [], 1)
    
    if ~isempty(options.marginalizationNames)
        xx = xlim;
        yy = ylim;
        text(xx(1)+(xx(2)-xx(1))*0.1, yy(2)-(yy(2)-yy(1))*0.1, ...
            [options.marginalizationNames{margs(i)} ' (' num2str(margVar(margs(i)),2) '%)'])
    end
end    

function [W,V,whichMarg] = dpca_pinv(Xfull, numC, varargin)

% dpca_pinv(X, numC, ...) performs PCA in each
% marginalization of X, selects numC components as encoding axes and takes
% its pseudoinverse as the decoding axes.

% dpca_pinv(..., 'PARAM1',val1, 'PARAM2',val2, ...) 
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
PCaxes = [];
whichMarg = [];

for m=1:length(Xmargs)
    [U,S,V] = svd(Xmargs{m});
    PCs = [PCs; S(1:10,1:10)*V(:,1:10)'];
    vars = [vars; diag(S(1:10,1:10)).^2];

    PCaxes = [PCaxes U(:,1:numC)];
    whichMarg = [whichMarg ones(1,numC)*m];
end
[~,ind] = sort(vars,'descend');
PCs = PCs(ind,:);
PCs = PCs(1:15,:);
vars = vars(ind) / totalVar * 100;

dims = size(Xfull);
Z = reshape(PCs, [15 dims(2:end)]);

yspan = max(abs(Z(:)));

figure
for i=1:15
    subplot(3,5,i)
    plotFunction(Z(i,:,:,:), [], [-yspan yspan]*1.1, vars(i), i, [], [], 1)
end    

PCaxes = PCaxes(:,ind(1:25));
V = PCaxes;
W = pinv(V)';
whichMarg = margNums(whichMarg(ind(1:25)));
%whichMarg = margNums(whichMarg);


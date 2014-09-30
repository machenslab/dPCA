function componentsSignif = dpca_signifComponents(accuracy, accuracyShuffle, whichMarg, varargin)

% componentsSignif = dpca_signifComponents(accuracy, accuracyShuffle, whichMarg)
% finds time periods of significant classification following
% cross-validation and shuffling procedures.
%
% [...] = dpca_signifComponents(..., 'PARAM1',val1, 'PARAM2',val2, ...) 
% specifies optional parameter name/value pairs:
%
% 'minChunk' - minimal required number of consecutive time points with
%              significant accuracy. Default: 10.

% default input parameters
options = struct('minChunk', 10);

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

componentsSignif = zeros(length(whichMarg), size(accuracy,3));

for marg = 1:size(accuracy,1)
    if isnan(accuracy(marg,1,1))
        % this is time marginalization => skip
        continue
    end
    
    comps = find(whichMarg == marg, 3);
    for num = 1:length(comps)
        componentsSignif(comps(num),:) = squeeze(accuracy(marg,num,:))' > ...
                                         squeeze(max(accuracyShuffle(marg,:,:), [], 3));    
    end
end

% remove short chunks of significance
for i=1:size(componentsSignif, 1)
    s = componentsSignif(i,:);
    ns = [0 find(s == 0) length(s)+1];
    tooShort = find(diff(ns) > 1 & diff(ns) < options.minChunk);
    for t = 1:length(tooShort)
        s(ns(tooShort(t))+1:ns(tooShort(t)+1)-1) = 0;
    end
    
    componentsSignif(i,:) = s;
end

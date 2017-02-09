function dpca_classificationPlot(accuracy, brier, accuracyShuffle, brierShuffle, decodingClasses, varargin)
% dpca_classificationPlot(accuracy, brier, accuracyShuffle, brierShuffle, decodingClasses)
% Plots the output of dpca_classificationAccuracy and dpca_classificationShuffled
% Only the accuracy input is required, other inputs may be []
%
% [...] = dpca_classificationPlot(..., 'PARAM1',val1, 'PARAM2',val2, ...) 
% specifies optional parameter name/value pairs:
%
%  'time'                   - time axis
%
%  'timeEvents'             - time-points that should be marked on each subplot
%
%  'whichMarg'              - which marginalization each component comes
%                             from. Is provided as an output of the dpca()
%                             function.
%
%  'marginalizationNames'   - names of each marginalization
%
% Note that the calculation of the accuracy in dpca_classificationAccuracy
% does its own dPCA internally, and it is possible that the order of
% components obtained therein is different to the order of components
% obtained externally when you calculated whichMarg. Therefore, it is not
% guaranteed that whichMarg component numbers will match exactly the
% component numbers in the plot.

% default input parameters
options = struct('time',                    [], ...   
                 'whichMarg',               [], ...
                 'marginalizationNames',	[],...
                 'timeEvents',              []);

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

% Get the time-axis values
if ~isempty(options.time) && length(options.time) >= size(accuracy, 3)
    acc_time = options.time(1:size(accuracy, 3));
else
    acc_time = 1:size(accuracy, 3);
end
if ~isempty(accuracyShuffle)
    if ~isempty(options.time) && length(options.time) >= size(accuracyShuffle, 2)
        sh_time = options.time(1:size(accuracyShuffle, 2));
    else
        sh_time = 1:size(accuracyShuffle, 2);
    end
end
if ~isempty(brier)
    if ~isempty(options.time) && length(options.time) >= size(brier, 3)
        br_time = options.time(1:size(brier, 3));
    else
        br_time = 1:size(brier, 3);
    end
end
if ~isempty(brierShuffle)
    if ~isempty(options.time) && length(options.time) >= size(brierShuffle, 2)
        brsh_time = options.time(1:size(brierShuffle, 2));
    else
        brsh_time = 1:size(brierShuffle, 2);
    end
end

numClasses = zeros(1, length(decodingClasses));
for i=1:length(decodingClasses)
    numClasses(i) = length(unique(decodingClasses{i}));
end

rows = 1:size(accuracy,1);
timeComp = find(isnan(accuracy(:,1,1)));
rows(timeComp) = NaN;
rows(rows>timeComp) = rows(rows>timeComp) - 1;

if isempty(options.marginalizationNames)
    options.marginalizationNames = arrayfun(@(x)...
        sprintf('Marg. #%s', x),...
        string(1:size(accuracy,1)),...
        'UniformOutput', false);
    options.marginalizationNames{timeComp} = 'Time';
end

figure
for i=setdiff(1:size(accuracy,1), timeComp)
    for j=1:size(accuracy,2)
        subplot(length(rows)-1,3,(rows(i)-1)*3+j)
        if ~isempty(options.whichMarg)
            comp_id = find(options.whichMarg==i, j);
        else
            comp_id = j;
        end
        title([options.marginalizationNames{i} ' (comp. ' num2str(comp_id(end)) ')'])

        hold on
        axis([min(acc_time) max(acc_time) 0 1])
        
        if i == max(setdiff(1:size(accuracy,1), timeComp))
            xlabel('Time (s)')
        end
        if j==1
            ylabel(sprintf('Classification\nAccuracy'))
        end
        
        if ~isempty(accuracyShuffle)
            axis([min(sh_time) max(sh_time) 0 1])
            hold on
            maxSh = max(accuracyShuffle(i,:,:),[],3);
            minSh = min(accuracyShuffle(i,:,:),[],3);
            h = patch([sh_time fliplr(sh_time)], [maxSh fliplr(minSh)], 'b');
            set(h, 'FaceAlpha', 0.5)
            set(h, 'EdgeColor', 'none')
        end
        
        if ~isempty(brierShuffle)
            maxSh = max(brierShuffle(i,:,:),[],3);
            minSh = min(brierShuffle(i,:,:),[],3);
            h = patch([brsh_time fliplr(brsh_time)], [maxSh fliplr(minSh)], 'r');
            set(h, 'FaceAlpha', 0.5)
            set(h, 'EdgeColor', 'none')
        end
        
        if ~isempty(decodingClasses)
            plot(xlim, 1/numClasses(i)*[1 1], 'k')
        end
        
        plot(acc_time, squeeze(accuracy(i,j,:)), 'b')  % Plot this after above to make sure it is drawn on top
        
        if ~isempty(brier)
            plot(br_time, squeeze(brier(i,j,:)), 'r')
        end
        
        if ~isempty(options.timeEvents)
            plot([options.timeEvents', options.timeEvents'], [0 1], 'Color', [0.6 0.6 0.6])
        end
    end
end
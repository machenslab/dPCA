function dpca_plot(Xfull, W, V, plotFunction, varargin)

% dpca_plot(X, W, V, plotFunction, ...) 
% produces a plot of the dPCA results. X is the data matrix, W and V
% are decoder and encoder matrices, plotFunction is a
% pointer to to the function that plots one component (see dpca_plot_default()
% for the template)

% dpca_plot(..., 'PARAM1',val1, 'PARAM2',val2, ...) 
% specifies optional parameter name/value pairs:
%
%  'whichMarg'              - which marginalization each component comes
%                             from. Is provided as an output of the dpca()
%                             function.
%
%  'time'                   - time axis
%
%  'timeEvents'             - time-points that should be marked on each subplot
%
%  'ylims'                  - array of y-axis spans for each
%                             marginalization or a single value to be used
%                             for each marginalization
%
%  'componentsSignif'       - time-periods of significant classification for each
%                             component. See dpca_signifComponents()
%
%  'timeMarginalization'    - if provided, it will be shown on top, and 
%                             irrespective of significance (because
%                             significant classification is not assessed for 
%                             time components)
%
%  'legendSubplot'          - number of the legend subplot
%
%  'marginalizationNames'   - names of each marginalization
%
%  'marginalizationColours' - colours for each marginalization
%
%  'explainedVar'           - structure returned by the dpca_explainedVariance
%
%  'numCompToShow'          - number of components to show on the explained
%                             variance plots (default = 15)
%
%  'X_extra'                - data array used for plotting that can be larger
%                             (i.e. have more conditions) than the one used
%                             for dpca computations

% default input parameters
options = struct('time',           [], ...   
                 'whichMarg',      [], ...
                 'timeEvents',     [], ...
                 'ylims',          [], ...
                 'componentsSignif', [], ...
                 'timeMarginalization', [], ...
                 'legendSubplot',  [], ...
                 'marginalizationNames', [], ...
                 'marginalizationColours', [], ...
                 'explainedVar',   [], ...
                 'numCompToShow',  15, ...
                 'X_extra',        []);

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

% can't show more than there is
numCompToShow = min(options.numCompToShow, size(W,2));

X = Xfull(:,:)';
Xcen = bsxfun(@minus, X, mean(X));
XfullCen = bsxfun(@minus, Xfull, mean(X)');
N = size(X, 1);
dataDim = size(Xfull);
Z = Xcen * W;
%!!
%Z = bsxfun(@times, Z, 1./std(Z, [], 1));
%!!

toDisplayMargNames = 0;

% if there are 4 or less marginalizations, split them into rows
if ~isempty(options.whichMarg) && ...
   length(unique(options.whichMarg)) <= 4 && length(unique(options.whichMarg)) > 1

    % time marginalization, if specified, goes on top
    if ~isempty(options.timeMarginalization)
        margRowSeq = [options.timeMarginalization setdiff(1:max(options.whichMarg), options.timeMarginalization)];
    else
        margRowSeq = 1:max(options.whichMarg);
    end
    
    componentsToPlot = [];
    subplots = [];
    for i=1:length(margRowSeq)
        if ~isempty(options.componentsSignif) && margRowSeq(i) ~= options.timeMarginalization
            % selecting only significant components
            minL = min(length(options.whichMarg), size(options.componentsSignif,1));
            moreComponents = find(options.whichMarg(1:minL) == margRowSeq(i) & ...
                sum(options.componentsSignif(1:minL,:), 2)'~=0, 3);
        else
            moreComponents = find(options.whichMarg == margRowSeq(i), 3);
        end
        componentsToPlot = [componentsToPlot moreComponents];
        subplots = [subplots (i-1)*4+2:(i-1)*4+2 + length(moreComponents) - 1];
    end
else
    % if there are more than 4 marginalizatons
    
    if isempty(options.whichMarg)
        % if there is no info about marginaliations
        componentsToPlot = 1:12;
    else
        % if there is info about marginaliations, select first 3 in each
        uni = unique(options.whichMarg);
        componentsToPlot = [];
        for u = 1:length(uni)
            componentsToPlot = [componentsToPlot find(options.whichMarg==uni(u), 2)];
        end
        componentsToPlot = sort(componentsToPlot);
        if length(componentsToPlot) > 12
            componentsToPlot = componentsToPlot(1:12);
        end
        
        toDisplayMargNames = 1;
    end
    subplots = [2 3 4 6 7 8 10 11 12 14 15 16];
    
    if numCompToShow < 12
        componentsToPlot = componentsToPlot(1:numCompToShow);
        subplots = subplots(1:numCompToShow);
    end
end
    
Zfull = reshape(Z(:,componentsToPlot)', [length(componentsToPlot) dataDim(2:end)]);

if ~isempty(options.X_extra)
    XF = options.X_extra(:,:)';
    XFcen = bsxfun(@minus, XF, mean(X));
    ZF = XFcen * W;
    %!!
    %ZF = bsxfun(@times, ZF, 1./std(ZF, [], 1));
    %!!
    dataDimFull = size(options.X_extra);
    Zfull = reshape(ZF(:,componentsToPlot)', [length(componentsToPlot) dataDimFull(2:end)]);
end

myFig = figure('Position', [0 0 1800 1000]);

% y-axis spans
if isempty(options.ylims)
    options.ylims = max(abs(Zfull(:))) * 1.1;
end
if length(options.ylims) == 1
    if ~isempty(options.whichMarg)
        options.ylims = repmat(options.ylims, [1 max(options.whichMarg)]);
    end
end

% plotting all components as subplots
for c = 1:length(componentsToPlot)
    cc = componentsToPlot(c);
    subplot(4, 4, subplots(c))
    
    if ~isempty(options.componentsSignif)
        signifTrace = options.componentsSignif(cc,:);
    else
        signifTrace = [];
    end
    
    if ~isempty(options.explainedVar)
        thisVar = options.explainedVar.componentVar(cc);
    else
        thisVar = [];
    end
    
    if ~isempty(options.whichMarg)
        thisYlim = options.ylims(options.whichMarg(cc));
        thisMarg = options.whichMarg(cc);
    else
        thisYlim = options.ylims;
        thisMarg = [];
    end
        
    dim = size(Xfull);
    cln = {c};
    for i=2:length(dim)
        cln{i} = ':';
    end

    %thisYlim = 5;
    % plot individual components using provided function
    plotFunction(Zfull(cln{:}), options.time, [-thisYlim thisYlim], ...
        thisVar, cc, options.timeEvents, ...
        signifTrace, thisMarg)
    
    if ismember(subplots(c), [2 6 10 14])
        if subplots(c) == 2 || subplots(c) == 14
            xlabel('Time (s)')
        else
            set(gca, 'XTickLabel', [])
        end
        ylabel('Normalized firing rate (Hz)')
    elseif ismember(subplots(c), [13 14 15 16])
        xlabel('Time (s)')
        set(gca, 'YTickLabel', [])
    else
        set(gca, 'XTickLabel', [])
        set(gca, 'YTickLabel', [])
    end
    
    if toDisplayMargNames && ~isempty(options.marginalizationNames)
        xx = xlim;
        yy = ylim;
        text(xx(1)+(xx(2)-xx(1))*0.1, yy(2)-(yy(2)-yy(1))*0.1, options.marginalizationNames(thisMarg))
    end
end 

% colours for marginalizations
if isempty(options.marginalizationColours)
    if ~isempty(options.explainedVar)
        L = length(options.explainedVar.totalMarginalizedVar);
        options.marginalizationColours = lines(L);
    elseif ~isempty(options.whichMarg)
        L = length(unique(options.whichMarg));
        options.marginalizationColours = lines(L);
    else
        options.marginalizationColours = [];
    end
end

% red-to-blue colormap
r = [5 48 97]/256;       %# end
w = [.95 .95 .95];       %# middle
b = [103 0 31]/256;      %# start
c1 = zeros(128,3);
c2 = zeros(128,3);
for i=1:3
    c1(:,i) = linspace(r(i), w(i), 128);
    c2(:,i) = linspace(w(i), b(i), 128);
end
redBlue256 = [c1;c2];

colormap([options.marginalizationColours; redBlue256])

% if there are four marginalizations or less, display labels
if ~isempty(options.whichMarg) && ...
   length(unique(options.whichMarg)) <= 4 && length(unique(options.whichMarg)) > 1 ...
   && ~isempty(options.marginalizationNames)
   
    offsetX = 0.31;
    yposs = [0.9 0.65 0.45 0.25];

    for m = intersect(1:4, unique(options.whichMarg(componentsToPlot)))        
        row = find(margRowSeq == m, 1);
        subplot(4,4,(row-1)*4+2)
        pos = get(gca, 'Position');
        
        annotation('rectangle', [offsetX-0.005 pos(2) 0.015 pos(4)], ...
            'EdgeColor', 'none', 'FaceColor', options.marginalizationColours(m,:));
        
        annotation('textarrow', offsetX*[1 1], yposs(row)*[1 1], ...
            'string', options.marginalizationNames{m}, ...
            'HeadStyle', 'none', 'LineStyle', 'none', ...
            'TextRotation', 90);
    end
end

% bar plot with projected variances
if ~isempty(options.explainedVar)
    axBar = subplot(4,4,9);
    hold on
    axis([0 numCompToShow+1 0 12.5])
    ylabel('Component variance (%)')
    b = bar(options.explainedVar.margVar(:,1:numCompToShow)' , 'stacked', 'BarWidth', 0.75);
    
    caxis([1 length(options.marginalizationColours)+256])
end

% cumulative explained variance
if ~isempty(options.explainedVar)
    axCum = subplot(4,4,5);
    hold on

%     % show signal variance if it's provided
%     if isfield(options.explainedVar, 'cumulativePCA_signal')
%         plot(1:numCompToShow, options.explainedVar.cumulativePCA_signal(1:numCompToShow), ...
%             '--k', 'LineWidth', 1)
%         plot(1:numCompToShow, options.explainedVar.cumulativeDPCA_signal(1:numCompToShow), ...
%             '--r', 'LineWidth', 1)
%         yy = [options.explainedVar.cumulativePCA_signal(1:numCompToShow) ...
%               options.explainedVar.cumulativeDPCA_signal(1:numCompToShow)];
%     end
    
    plot(1:numCompToShow, options.explainedVar.cumulativePCA(1:numCompToShow), ...
        '.-k', 'LineWidth', 1, 'MarkerSize', 15);
    plot(1:numCompToShow, options.explainedVar.cumulativeDPCA(1:numCompToShow), ...
        '.-r', 'LineWidth', 1, 'MarkerSize', 15);
    %yy = [options.explainedVar.cumulativePCA(1:numCompToShow) ...
    %    options.explainedVar.cumulativeDPCA(1:numCompToShow)];
    ylabel({'Explained variance (%)'})
        
    if isfield(options.explainedVar, 'totalVar_signal')
        plot([0 numCompToShow+1], options.explainedVar.totalVar_signal/options.explainedVar.totalVar*100*[1 1], 'k--')
    end
           
    %axis([0 numCompToShow+1 floor(min(yy-5)/10)*10 min(ceil(max(yy+5)/10)*10, 100)])
    axis([0 numCompToShow+1 0 100])
    xlabel('Component')
    legend({'PCA', 'dPCA'}, 'Location', 'SouthEast');
    legend boxoff
end

% angles and correlations between components
a = corr(Z(:,1:numCompToShow));
%a = a*0;
b = V(:,1:numCompToShow)'*V(:,1:numCompToShow);

% display(['Maximal correlation: ' num2str(max(abs(a(a<0.999))))])
% display(['Minimal angle: ' num2str(acosd(max(abs(b(b<0.999)))))])

[~, psp] = corr(V(:,1:numCompToShow), 'type', 'Kendall');
%[cpr, ppr] = corr(V(:,1:numCompToShow));
map = tril(a,-1)+triu(b);

axColormap = subplot(4,4,13);
L = length(options.marginalizationColours);
image(round(map*128)+128 + L)

xlabel('Component')
ylabel('Component')

cb = colorbar('location', 'southoutside');
set(cb, 'xlim', [L+1 L+256], 'XTick', [L+1:65:L+256 L+256], 'XTickLabel', -1:0.5:1)

hold on
[i,j] = ind2sub(size(triu(b,1)), ...
    find(abs(triu(b,1)) > 3.3/sqrt(size(Xfull,1)) & psp<0.001)); % & abs(csp)>0.02));
plot(j,i,'k*')


% adjust subplot positions
subplot(4,4,14)
pos = get(gca, 'Position');
bottom = pos(2);
if isempty(find(subplots==14, 1))
    delete(gca)
end

pos = get(cb, 'Position');
set(cb, 'Position', [pos(1)+pos(3)*2/3 bottom-pos(4)*3 pos(3)/3 pos(4)])

pos = get(axColormap, 'Position');
height = pos(3)*1800/1000;
set(axColormap, 'Position', [pos(1) bottom pos(3) height])
set(axColormap, 'Xtick', [1 5:5:numCompToShow])
set(axColormap, 'Ytick', [1 5:5:numCompToShow])
set(axColormap, 'XtickLabel', [1 5:5:numCompToShow])

if ~isempty(options.explainedVar)
    pos = get(axColormap, 'Position');
    top = pos(2)+pos(4);
    pos = get(axBar, 'Position');
    set(axBar, 'Position', [pos(1) top+0.03 pos(3) pos(4)])
    
    pos = get(axBar, 'Position');
    top = pos(2)+pos(4);
    pos = get(axCum, 'Position');
    set(axCum, 'Position', [pos(1) top+0.05 pos(3) pos(4)])
    
    set(axBar, 'Xlim', [0.5 numCompToShow+0.5])
    set(axBar, 'Xtick', [1 5:5:numCompToShow])
    set(axCum, 'Xlim', [0.5 numCompToShow+0.5])
    set(axCum, 'Xtick', [1 5:5:numCompToShow])
end

for pl = intersect(2:4:16, subplots)
    subplot(4,4,pl)
    pos = get(gca, 'Position');
    set(gca, 'Position', [pos(1)+0.02 pos(2) pos(3) pos(4)])
end

for pl = intersect(4:4:16, subplots)
    subplot(4,4,pl)
    pos = get(gca, 'Position');
    set(gca, 'Position', [pos(1)-0.02 pos(2) pos(3) pos(4)])
end

% legend
if ~isempty(options.legendSubplot)
    s = subplot(4,4,options.legendSubplot);
    delete(s)
    subplot(4,4,options.legendSubplot)

    if ~isempty(options.X_extra)
        plotFunction('legend', size(options.X_extra))
    else
        plotFunction('legend', size(Xfull))
    end
end

% pie chart
if ~isempty(options.explainedVar)
    axes('position', [0.205 0.47 0.1 0.1])
    
    if isfield(options.explainedVar, 'totalMarginalizedVar_signal')
        d = options.explainedVar.totalMarginalizedVar_signal / options.explainedVar.totalVar_signal * 100;
       
        % In some rare cases the *signal* explained variances can be
        % negative (usually around 0 though); this means that the
        % corresponding marginalization does not carry [almost] any signal.
        % In order to avoid confusing pie charts, we set those to zero and
        % rescale the others to sum to 100%.
        if ~isempty(find(d<0, 1))
            d(d<0) = 0;
            d = d/sum(d)*100;
        end
    else
        d = options.explainedVar.totalMarginalizedVar / options.explainedVar.totalVar * 100;
    end
    
    % Rounding such that the rounded values still sum to 100%. Using
    % "largest remainder method" of allocation
    roundedD = floor(d);
    while sum(roundedD) < 100
        [~, ind] = max(d-roundedD);
        roundedD(ind) = roundedD(ind) + 1;
    end
    
    if ~isempty(options.marginalizationNames)
        for i=1:length(d)
            margNamesPerc{i} = [options.marginalizationNames{i} ' ' num2str(roundedD(i)) '%'];
        end
    else
        for i=1:length(d)
            margNamesPerc{i} = [num2str(roundedD(i)) '%'];
        end
    end
    pie(d, ones(size(d)), margNamesPerc)
    caxis([1 length(options.marginalizationColours) + 256])
end

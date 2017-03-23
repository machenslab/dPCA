function [optimalLambda, optimalLambdas] = dpca_optimizeLambda(Xfull, ...
    Xtrial, numOfTrials, varargin)

% optimalLambda = dpca_optimizeLambda(X, Xtrial, numOfTrials, ...)
% computes optimal regularization parameter. X is the data array. Xtrial 
% is an array storing single trials. It has one extra dimension as compared 
% with X and stores individual single trial firing rates, as opposed to the 
% trial average. numOfTrials has one dimension fewer than X and for each 
% neuron and combination of parameters (without time) specifies the number 
% of available trials in X_trial. All entries have to be larger than 1.
%
% This code assumes that time parameter is stored in the last dimension of
% X. For datasets without time, some other cross-validation needs to be
% used.
%
% [optimalLambda, optimalLambdas] = dpca_optimizeLambda(...) additionally
% returns a list of optimal lambdas found separately for each
% marginalization
%
% [...] = dpca_optimizeLambda(..., 'PARAM1',val1, 'PARAM2',val2, ...) 
% specifies optional parameter name/value pairs:
%
% 'numComps'        - how many components to use overall or in each marginalization 
%                     (default: 25)
%
% 'lambdas'         - an array of lambdas to scan
%                     (default: 1e-07 * 1.5.^[0:25])
%
% 'numRep'          - how many cross-validation iterations to perform
%                     (default: 10)
%
% 'display'         - "yes" or "no". If yes, then a figure is displayed showing
%                     reconstruction errors.
%                     (default: yes)
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
% 'filename'        - if provided, reconstruction errors and optimal lambdas will
%                     be stored in this file
%
% 'method'          - three possible ways to compute the objective to be
%                     minimized:
%                        'naive'      - reconstruction error on the test
%                                       (deprecated! do not use)
%                        'training'   - use test data to reconstruct training
%                                       data (DEFAULT)
%                        'neuronwise' - reconstruction error on the test
%                                       data computed per neuron
%
% 'simultaneous'    - if the dataset is simultaneously recorded (true) or
%                     not (false - DEFAULT)
%
% 'noiseCovType'    - two possible ways to compute noise covariance matrix:
%                        'averaged'   - average over conditions
%                        'pooled'     - pooled over conditions (DEFAULT)
%                        'none'       - not using noise covariance at all

% default input parameters
options = struct('numComps',       25,                  ...   
                 'lambdas',        1e-07 * 1.5.^[0:25], ...
                 'numRep',         10,                  ...
                 'display',        'yes',               ...
                 'combinedParams', [],                  ...
                 'filename',       [],                  ...
                 'method',         'training',          ...
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

if min(numOfTrials(:)) <= 0
    error('dPCA:tooFewTrials0','Some neurons seem to have no trials in some condition(s).\nPlease ensure that min(numOfTrials) > 0.')
elseif min(numOfTrials(:)) == 1
    error('dPCA:tooFewTrials1','Cannot perform cross-validation when there are neurons \nhaving only one trial per some condition(s). \nPlease ensure that min(numOfTrials) > 1.')
end

tic
Xsum = bsxfun(@times, Xfull, numOfTrials);
% Xsum = nansum(Xtrial,5);      % the previous line is equivalent but faster

% Note (12 Jul 2016): the function was updating the noise cov for the training
% data without recomputing it. It's tricky to implement for simultaneous
% recordings, so I am dropping this for now, and switching to the easier
% but slower version. (The same goes for decoding functions.)
% X2sum = nansum(Xtrial.^2, ndims(Xtrial)) - bsxfun(@times, Xfull.^2, numOfTrials);

for rep = 1:options.numRep
    fprintf(['Iteration #' num2str(rep) ' out of ' num2str(options.numRep)])
    repTic = tic;
    
    [Xtest, XtrainFull] = dpca_getTestTrials(Xtrial, numOfTrials, ...
        'simultaneous', options.simultaneous);
    Xtrain = bsxfun(@times, Xsum - Xtest, 1./(numOfTrials-1));
    % This gives the same result as the previous line, but is much slower
    % Xtrain = nanmean(XtrainFull, ndims(XtrainFull));
    
    % Note (12 Jul 2016): the function was updating the noise cov for the training
    % data without recomputing it. It's tricky to implement for simultaneous
    % recordings, so I am dropping this for now, and switching to the easier
    % but slower version.
%     ssTrain = X2sum + bsxfun(@times, Xfull.^2, numOfTrials) ...
%         - Xtest.^2 - bsxfun(@times, Xtrain.^2, (numOfTrials-1));
%     SSnoiseSumOverT = sum(ssTrain, ndims(ssTrain));
%     CnoiseTrain = diag(sum(bsxfun(@times, SSnoiseSumOverT(:,:), 1./(numOfTrials(:,:)-1)),2));
    
    if ~strcmp(options.noiseCovType, 'none')
        CnoiseTrain = dpca_getNoiseCovariance(Xtrain, XtrainFull, numOfTrials-1, ...
                      'simultaneous', options.simultaneous, 'type', options.noiseCovType);
    else
    	CnoiseTrain = 0;
    end	
    
    XtestCen = bsxfun(@minus, Xtest, mean(Xtest(:,:),2));
    XtestMargs = dpca_marginalize(XtestCen, 'combinedParams', options.combinedParams, ...
                    'ifFlat', 'yes');
    for i=1:length(XtestMargs)
        margTestVar(i) = sum(XtestMargs{i}(:).^2);
    end
    
    XtrainCen = bsxfun(@minus, Xtrain, mean(Xtrain(:,:),2));
    XtrainMargs = dpca_marginalize(XtrainCen, 'combinedParams', options.combinedParams, ...
                    'ifFlat', 'yes');
    for i=1:length(XtrainMargs)
        margTrainVar(i) = sum(XtrainMargs{i}(:).^2);
    end
    
    if strcmp(options.method, 'naive') || strcmp(options.method, 'neuronwise')
        margVar_toNormalize = margTestVar;
    else
        margVar_toNormalize = margTrainVar;
    end

    for l = 1:length(options.lambdas)
        fprintf('.')
        
        [W,V,whichMarg] = dpca(Xtrain, options.numComps, ...
            'combinedParams', options.combinedParams, ...
            'lambda', options.lambdas(l), 'Cnoise', CnoiseTrain);
                        
        cumError = 0;
        for i=1:length(XtestMargs)
            recError = 0;
            
            if strcmp(options.method, 'naive')
                recError = sum(sum((XtestMargs{i} - V(:,whichMarg==i)*W(:,whichMarg==i)'*XtestCen(:,:)).^2));

            elseif strcmp(options.method, 'training')
                recError = sum(sum((XtrainMargs{i} - V(:,whichMarg==i)*W(:,whichMarg==i)'*XtestCen(:,:)).^2));
            
            elseif strcmp(options.method, 'neuronwise')
                % diagVW = diag(diag(V(:,whichMarg==i)*W(:,whichMarg==i)'));
                diagVW = diag(sum(V(:,whichMarg==i).*W(:,whichMarg==i), 2));
                
                recError = sum(sum((XtestMargs{i} - V(:,whichMarg==i)*W(:,whichMarg==i)'*XtestCen(:,:) ...
                    + diagVW*XtestCen(:,:)).^2));
                
                % FOR DEBUGGING: computes the same thing
                % for neur = 1:size(XtestCen,1)
                %   otherN = [1:(neur-1) (neur+1):size(XtestCen,1)];
                %   recError = recError + ...
                %       sum((XtestMargs{i}(neur,:) - V(neur,whichMarg==i)*W(otherN,whichMarg==i)'*XtestCen(otherN,:)).^2);
                % end
            end
            
            errorsMarg(i, l, rep) = recError/margVar_toNormalize(i);
            cumError = cumError + recError;
        end
        
        errors(l,rep) = cumError / sum(margVar_toNormalize);
    end
        
    repTime = toc(repTic);
    fprintf([' [' num2str(round(repTime)) ' s]'])
    fprintf('\n')
end

timeTaken = toc;

meanError = mean(errors,2);
[~, ind] = min(meanError);
optimalLambda = options.lambdas(ind);

meanErrorMarg = mean(errorsMarg(:, :,:), 3);
[~, indm] = min(meanErrorMarg, [], 2);
optimalLambdas = options.lambdas(indm);

if ~isempty(options.filename)
    lambdas = options.lambdas;
    numComps = options.numComps;
    save(options.filename, 'lambdas', 'errors', 'errorsMarg', 'optimalLambda', 'optimalLambdas', 'numComps', 'timeTaken')
end

if strcmp(options.display, 'yes')
    figure
    
    title('Relative cross-validation errors')
    xlabel('Regularization parameter, lambda')
    ylabel('Residual variance over total test variance')
    
    hold on
    hh = patch([log(options.lambdas) fliplr(log(options.lambdas))], ...
        [min(errors,[],2)' fliplr(max(errors,[],2)')], [0 0 0]);
    set(hh, 'FaceAlpha', 0.2)
    set(hh, 'EdgeColor', 'none')
    h1 = plot(log(options.lambdas), meanError, '.-k', 'LineWidth', 2);
    plot(log(options.lambdas(ind)), meanError(ind), '.k', 'MarkerSize', 30)

    colors = lines(size(meanErrorMarg,1));
    for i=1:size(meanErrorMarg,1)
        hh = patch([log(options.lambdas) fliplr(log(options.lambdas))], ...
            [squeeze(min(errorsMarg(i,:,:),[],3)) fliplr(squeeze(max(errorsMarg(i,:,:),[],3)))], ...
            colors(i,:));
        set(hh, 'FaceAlpha', 0.2)
        set(hh, 'EdgeColor', 'none')
    end
    hh = plot(log(options.lambdas), meanErrorMarg', '.-', 'LineWidth', 1);
    for i=1:size(meanErrorMarg,1)
        plot(log(options.lambdas(indm(i))), meanErrorMarg(i,indm(i)), '.k', 'MarkerSize', 20)
    end
    
    legendText = {};
    for i = 1:length(hh)
        legendText{i} = ['Marginalization #' num2str(i)];
    end
    legendText{end+1} = 'Overall';
    legend([hh; h1], legendText, 'Location', 'East')
    
    xticks = [1e-07:1e-07:1e-06 2e-06:1e-06:1e-05 2e-05:1e-05:1e-04 2e-04:1e-04:1e-03];
    xtickLabels = num2cell(xticks);
    for i=setdiff(1:length(xticks), [1 10 19 28])
        xtickLabels{i} = '';
    end
    set(gca,'XTick', log(xticks))
    set(gca,'XTickLabel', xtickLabels)
    
    plot(xlim, [1 1], 'k')
    axis([log(min(options.lambdas)) log(max(options.lambdas)) 0 1.2])
end

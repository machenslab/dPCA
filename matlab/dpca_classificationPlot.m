function dpca_classificationPlot(accuracy, brier, accuracyShuffle, brierShuffle, decodingClasses)

for i=1:length(decodingClasses)
    numClasses(i) = length(unique(decodingClasses{i}));
end

rows = 1:size(accuracy,1);
timeComp = find(isnan(accuracy(:,1,1)));
rows(timeComp) = NaN;
rows(rows>timeComp) = rows(rows>timeComp) - 1;

figure
for i=setdiff(1:size(accuracy,1), timeComp)
    for j=1:size(accuracy,2)
        subplot(length(rows)-1,3,(rows(i)-1)*3+j)
        title(['Marginalization #' num2str(i)])
        hold on
        axis([0 size(accuracy,3) 0 1])
        
        if ~isempty(accuracyShuffle)
            axis([1 size(accuracy,3) 0 1])
            hold on
            maxSh = max(accuracyShuffle(i,:,:),[],3);
            minSh = min(accuracyShuffle(i,:,:),[],3);
            time = 1:length(maxSh);
            h = patch([time fliplr(time)], [maxSh fliplr(minSh)], 'b');
            set(h, 'FaceAlpha', 0.5)
            set(h, 'EdgeColor', 'none')
        end
        
        if ~isempty(brierShuffle)
            maxSh = max(brierShuffle(i,:,:),[],3);
            minSh = min(brierShuffle(i,:,:),[],3);
            time = 1:length(maxSh);
            h = patch([time fliplr(time)], [maxSh fliplr(minSh)], 'r');
            set(h, 'FaceAlpha', 0.5)
            set(h, 'EdgeColor', 'none')
        end
        
        if ~isempty(decodingClasses)
            plot(xlim, 1/numClasses(i)*[1 1], 'k')
        end
        
        if ~isempty(accuracy)
            plot(squeeze(accuracy(i,j,:)))
        end
        
        if ~isempty(brier)
            plot(squeeze(brier(i,j,:)), 'r')
        end
    end
end
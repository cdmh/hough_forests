close all;

files = {'HOG_roc.mat','4DISM-train210_roc.mat','Art-ISM-train400_roc.mat','roc_mr.mat','roc2.mat'};

figure(1);
hold on;
set(gca,'FontSize',14);
axis([0 1 0 1]);
grid on;

colors = [0 0 0; 1 0 0; 0 1 0; 0 0 1; 1 1 0; 0 1 1; 1 0 1; 1 1 1];

for i = 1:length(files)
    load( files{i} ); 
    plot((1-pre),rec, 'Color', colors(mod(i-1,length(files))+1,:), 'LineWidth', 2 );
end

legend({'HOG+SVM*','4D-ISM*','Andriluka et al. CVPR08','HF - weaker supervision','Hough Forest'},'Location','SouthEast');
xlabel('1-Precision');
ylabel('Recall');

hold off;



























close all;

files = {'roc_sc2_a.mat','roc_sc2_b.mat','roc_mr.mat'};
figure(1);
hold on;
set(gca,'FontSize',14);
axis([0 0.5 0.5 1]);
grid on;

colors = [0 0 0; 1 0 0; 0 1 0; 0 0 1; 0 1 1; 1 0 1; 1 1 1];
ledisp = cell(1,length(files));

for i = 1:length(files)
    load( files{i} ); 
    plot((1-pre),rec, 'Color', colors(mod(i-1,length(files))+1,:), 'LineWidth', 2 );
end

legend({'Hough Forest','Hough Forest (Recenter)','Hough Forest (Recenter,4D)'},'Location','SouthEast');
xlabel('1-Precision');
ylabel('Recall');

hold off;





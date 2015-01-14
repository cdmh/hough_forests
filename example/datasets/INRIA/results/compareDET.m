close all;

color = [.9 .1 .1;
         .1 .9 .1;
         .1 .1 .9;
         .9 .9 .1;
         .9 .1 .9;
         .1 .9 .9];
     
figure(1);
set(gca,'XScale','log');
set(gca,'FontSize',14);
hold on;

plotHOG([0 0 0])
plotDET('histsmooth.txt','schistsmooth.txt',color(1,:));
 
axis([10^(-6) 1 0.65 1]);
legend({'HOG+SVM','Hough Forest'},'Location','SouthEast');
xlabel('FPPW');
ylabel('Recall');
hold off;


    
   
 
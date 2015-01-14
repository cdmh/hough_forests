function plotHOG(color)

file = fopen('OLTbinaries/HOG/WindowTest_Positive/histogram.txt','r');
p = fscanf(file,'%d ',[1 4001]);
fclose(file);

file = fopen('OLTbinaries/HOG/WindowTest_Negative/histogram.txt','r');
n = fscanf(file,'%d ',[1 4001]);
fclose(file);

p=cumsum(fliplr(p./sum(p)));
n=cumsum(fliplr(n./sum(n)));

semilogx(n,p,'Linewidth',2,'Color',color);



 
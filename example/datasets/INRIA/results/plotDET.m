function plotDET(posfile, negfile, color)

file = fopen(posfile,'r');
p = fscanf(file,'%d ',[1 256]);
fclose(file);

file = fopen(negfile,'r');
n = fscanf(file,'%d ',[1 256]);
fclose(file);

p=cumsum(fliplr(p./sum(p)));
n=cumsum(fliplr(n./sum(n)));

semilogx(n,p,'Linewidth',2,'Color',color);

 
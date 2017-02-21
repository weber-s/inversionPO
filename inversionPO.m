clear all

file="PODTTµg-Mass.csv"
factorFile="Mass.csv";
POFile="POAA.csv";
src={'Indus', 'Sels', 'Bioprimaire', 'Dust', ...
	'Véhiculaire','Biosecondaire', 'Sulfate0rich', 'Nitrate0rich', ...
	'BB'};

factors=dlmread(factorFile);
PO=dlmread(POFile);

nb_obs=size(factors)(1);
nb_factors=size(factors)(2);

idok=~isnan(PO);
POInv=PO(idok);
factorsInv=factors(idok,:);

% inversion
r=factorsInv\POInv

cell2csv(file, src)
csvwrite(file,r',"-append")

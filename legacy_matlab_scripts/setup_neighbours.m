% a function stolen from PSMT's MC simulations that efficienctly calculates
% the neighbours to all sites, given a cutoff radius rcut and lattice with
% dimensions n*a1 and n*a2 for lattice parameters a1 and a2

function [neighbours,distance]=setup_neighbours(a1,a2,n,sites,rcut)

rounderr=1e-3;

sitesX=sites(:,1);
sitesY=sites(:,2);

[X1,X2]=meshgrid(sitesX,sitesX);
[Y1,Y2]=meshgrid(sitesY,sitesY);
distanceCurrent_s=(X1-X2).^2+(Y1-Y2).^2;

trials=[-1 0 1];
for k=1:numel(trials)
    for kk=1:numel(trials)
        X2wrap=X2+trials(k)*a1(1)*n+trials(kk)*a2(1)*n; 
        Y2wrap=Y2+trials(k)*a1(2)*n+trials(kk)*a2(2)*n; 
        distance_s=(X1-X2wrap).^2+(Y1-Y2wrap).^2; 
        distanceCurrent_s=min(distance_s,distanceCurrent_s); 
    end
end

distance=sqrt(distanceCurrent_s);

d=distance(1,:); % vector of distances
numneigh=sum(d<rcut)-1; % number of neighbours

sitenos=ones(size(1:n^2))'*(1:n^2); sitenos=sitenos'; % matrix of all possible neighbours, to be truncated
d=distance(:); sitenos=sitenos(:); % manipulating to compare to rcut
sitenos=sitenos(d<rcut); d=d(d<rcut); % comparing to rcut
sitenos(d<rounderr)=[]; d(d<rounderr)=[]; % removing sites included due to rounding errors

neighbours=reshape(sitenos,[numneigh,n^2]); % reshaping the array
neighbours=neighbours'; % juggle the shape one more time so that every row represents a site
distance=reshape(d,[numneigh,n^2]); % return matrix of same size as neighbours matrix, containing distances
distance=distance'; % juggling the shape again

end
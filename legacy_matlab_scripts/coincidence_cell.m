% outputs: rsnuc, s11, s12, s21, s22
% rsnuc is my guess as to the position of the first sulphur atom when the
% sqrt(39) phase nucleates at a defect site on the Ni(111) surface
% it's the position vector of the sulphur atom in a sqrt(39) cell closest
% to the unreconstructed fcc sites of the Ni(111) surface on which the
% sulphur atoms move
% s11, s12 tile one of the two possible structures sqrt(39) phase
% s21, s22 tile the other possible orientation of the sqrt(39) phase

% there are 3 x 2 = 6 possible orientations of the sqrt(39) phase
% factor of three from three-fold symmetry of the Ni(111) surface, factor
% of two from the broken symmetry when the sqrt(39) phase is formed
% this explains why the diffraction scans in e.g. my thesis have reflection
% symmetry about the <10> and <11> directions
% this script only calculates the two orientations related by reflection
% symmetry - the rotations are calculated in the main script

% the vectors describing the s superstructure (s11, s21, s12 and s22) are
% needed to determine how an island grows

% the script accounts for coincidence of the S superstructure with the
% second Ni(111) layer (and not the first!) - be careful

function [rsnuc, s11, s12, s21, s22] = coincidence_cell(a1, a2, plotflag)

% needs to be large enough - size of lattice to test points
n = 10;

% generating a hexagonal lattice
r = zeros((2 * n + 1)^2, 2);
index = 0;
for i = -n:n
    for j = -n:n
        index = index + 1;
        temp = i * a1 + j * a2;
        r(index, :) = temp;
    end
end

% generating the top and fcc sites for this lattice
rtop = r + repmat((a1 + a2) / 3, size(r, 1), 1);
rfcc = r + repmat(2 * (a1 + a2) / 3, size(r, 1), 1);

% the vectors decribing the s/ni coincidence cell
% came from staring at the diagrams of the sqrt(39) phase in the 
% literature for way too long
c1 = 7 * a1 - 5 * a2;
c2 = 2 * a1 + 5 * a2;

% the vectors describing the unit cell of sulphur
% as above, from diagrams of the sqrt(39) phase in the literature
s1 = (3 * c1 + c2) / 10;
s2 = (3 * c2 - c1) / 10;

% generating the positions of all sulphur atoms in one coincidence cell
rs = zeros(13, 2);
index = 0;
for i = 0:2
    for j = 0:2
        index = index + 1;
        rs(index, :) = s1 + i * s1 + j * s2;
    end
end
rs(11, :) = c1;
rs(12, :) = c2;
rs(13, :) = c1 + c2;

% finding the sulphur atom closest to an fcc site (on which lattice the
% individual sulphur atoms on the unreconstructed surface move)
% this calculation is probably a bit overkill, as it probably makes no
% difference to the simulation of the physics

% calculate all distances in a vectorized manner
distances = zeros(size(rs, 1), size(rfcc, 1));
for i = 1:size(rs, 1)
    distances(i, :) = sqrt(sum((rfcc - rs(i, :)).^2, 2));
end
[~, mindex] = min(distances(:));
[inds, indrupper] = ind2sub(size(distances), mindex);


% center coincidence cell
rscentered = rs - rfcc(indrupper, :);
rsnuc = rscentered(inds, :);
rfcccent = rfcc(indrupper, :);
rstest = rfcccent + rsnuc;

% plot if plotflag is set
if plotflag == 1
    figure; hold on
    plot(r(:, 1), r(:, 2), 'bo')
    plot(rtop(:, 1), rtop(:, 2), 'co')
    plot(rfcc(:, 1), rfcc(:, 2), 'k.')
    plot(rs(:, 1), rs(:, 2), 'rx')
    plot([rfcccent(1) rstest(1)], [rfcccent(2) rstest(2)], 'y')
    axis equal
end

% generating the modified lattice of nickel atoms that sit just below the
% sulphur, based on the information available in the literature - just for
% plotting purposes
nsquare = 5;
rmodtop = zeros((2 * n + 1)^2, 2);
index = 0;
for i = -nsquare:nsquare
    for j = -nsquare:nsquare
        index = index + 1;
        rmodtop(index, :) = i * s1 / 2 + j * s2 / 2;
    end
end

% shift coincidence cell points
rsshifted = rs - r(indrupper, :);
rsshiftedcentre = rsshifted(5, :);
rmodtop = rmodtop + rsshiftedcentre - (s1 + s2) / 4;

% plot if plotflag is set
if plotflag == 1
    figure; hold on
    plot(r(:, 1), r(:, 2), 'bo')
    plot(rmodtop(:, 1), rmodtop(:, 2), 'k.')
    plot(rsshifted(:, 1), rsshifted(:, 2), 'rx')
    axis equal
end

% collecting together all unit vectors
s11 = s1;
s12 = s2;
s21 = (3 * c1 - c2) / 10;
s22 = (c1 + 3 * c2) / 10;

% reflect coincidence cell points
Rreflect = @(theta) [cos(2 * theta) sin(2 * theta); sin(2 * theta) -cos(2 * theta)];
point = (c2 - c1) / 2;
ang = atan(point(2) / point(1));

% centre points again
rszeroed = rs - repmat(rs(5, :), size(rs, 1), 1);
rszeroedr = zeros(size(rszeroed));
for i = 1:size(rszeroedr, 1)
    rszeroedr(i, :) = (Rreflect(ang) * rszeroed(i, :)')';
end

% plot if plotflag is set
if plotflag == 1
    figure; hold on
    plot(rszeroed(:, 1), rszeroed(:, 2), 'rx')
    plot(rszeroedr(:, 1), rszeroedr(:, 2), 'bx')
    plot([0 point(1)], [0 point(2)], 'k.')
    plot([0 s11(1)], [0 s11(2)], 'y')
    plot([0 s12(1)], [0 s12(2)], 'y')
    plot([0 s21(1)], [0 s21(2)], 'g')
    plot([0 s22(1)], [0 s22(2)], 'g')
    axis equal
end

end

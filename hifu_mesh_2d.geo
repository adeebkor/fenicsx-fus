Point(1) = {0.015166852264521177, 0.05, 0.000000, 1.0};
Point(2) = {0.015166852264521177, -0.05, 0.000000, 1.0};
Point(3) = {0.12, 0.05, 0, 1.0};
Point(4) = {0.12, -0.05, 0, 1.0};
Point(5) = {0.09, 0.000000, 0.000000, 1.0};

Circle(1) = {1, 5, 2};
Line(2) = {2, 4};
Line(3) = {4, 3};
Line(4) = {3, 1};

Line Loop(5) = {1, 2, 3, 4};
Surface(6) = {5};

density = 81;

Transfinite Line {1, 3} = density;
Transfinite Line {2, 4} = density;

Physical Line(1) = {1};
Physical Line(2) = {2, 3, 4};
Physical Surface(1) = {6};

Transfinite Surface "*";
Recombine Surface "*";

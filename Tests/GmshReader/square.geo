Mesh.MshFileVersion = 2.2;

L = 10.0;

Point(1) = {-L, -L, 0, 1.0};
Point(2) = { L, -L, 0, 1.0};
Point(3) = { L,  L, 0, 1.0};
Point(4) = {-L,  L, 0, 1.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Transfinite Curve {4, 3, 2, 1} = 5 Using Progression 1;
Transfinite Surface {1};
Recombine Surface {1};

Physical Surface("Interior", 1) = {1};
Physical Curve("Periodic", 2) = {1, 2, 3, 4};

//+ Options controlling mesh generation
Mesh.ElementOrder = 1; //+ Set desired element order beetween 3 and 8 (1,2 not supported in SOD)
Mesh 2;                //+ Volumetric mesh

//+ Generate the periodicity between surface pairs
Periodic Curve {2} = {4} Translate {2*L, 0, 0};
Periodic Curve {3} = {1} Translate {0, 2*L, 0};
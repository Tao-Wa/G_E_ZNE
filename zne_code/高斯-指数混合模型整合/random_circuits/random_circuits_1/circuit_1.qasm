OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
rz(pi) q[0];
rz(2.470620973951443) q[1];
rz(1.084523370342321) q[2];
rz(-pi) q[3];
x q[3];

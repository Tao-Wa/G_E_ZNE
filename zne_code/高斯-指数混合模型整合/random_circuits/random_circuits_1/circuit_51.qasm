OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
rz(-pi/4) q[0];
x q[2];
rz(pi/2) q[2];
sx q[3];

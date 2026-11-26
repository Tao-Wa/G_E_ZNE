OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
x q[0];
x q[1];
rz(-pi) q[2];
x q[2];
rz(-pi/4) q[3];

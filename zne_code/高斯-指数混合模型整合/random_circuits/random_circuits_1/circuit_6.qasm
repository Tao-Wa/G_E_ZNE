OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
sx q[0];
rz(-pi) q[1];
x q[1];
rz(4.500539581559064) q[2];

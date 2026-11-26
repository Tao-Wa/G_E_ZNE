OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
rz(-pi/2) q[1];
cx q[0],q[2];
rz(0.41771849421293344) q[2];
cx q[0],q[2];
x q[3];

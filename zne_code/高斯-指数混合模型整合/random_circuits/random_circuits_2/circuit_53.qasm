OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
rz(1.2234260826665453) q[2];
cx q[0],q[2];
rz(-1.2234260826665453) q[2];
cx q[0],q[2];
cx q[1],q[3];

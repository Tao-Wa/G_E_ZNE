OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
rz(1.3426412554756963) q[0];
cx q[1],q[0];
rz(-1.3426412554756963) q[0];
cx q[1],q[0];
rz(4.088984985103426) q[2];

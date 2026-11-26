OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
cx q[0],q[1];
cx q[1],q[0];
rz(-0.5547176507912279) q[3];
sx q[3];
rz(2.8762518470026652) q[3];

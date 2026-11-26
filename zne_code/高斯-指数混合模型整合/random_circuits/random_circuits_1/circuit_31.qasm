OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[0];
rz(5.036807734048276) q[1];
rz(pi/4) q[2];

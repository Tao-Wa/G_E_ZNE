OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
rz(-pi/2) q[0];
rz(1.5361832337264325) q[1];
rz(-pi) q[2];
x q[2];
rz(pi) q[3];

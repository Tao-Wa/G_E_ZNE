OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(2.218744742024208) q[1];
rz(5.2108960674576705) q[2];

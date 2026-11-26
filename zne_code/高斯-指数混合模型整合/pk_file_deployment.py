"""Unpack the generated .pk circuit collections into .qasm / .qpy files.

Reads ``{random_gen|trotter_gen|grover_gen}/step_*.pk`` and writes
``circuit_<i>.qasm`` and ``circuit_<i>.qpy`` into the matching
``<family>_circuits_<n>`` directory, together with ``export_summary.txt``.

The step/entry mapping mirrors the original ``__main__`` block exactly:
random uses ``step_{2*i}.pk`` -> entry ``i+1`` (i = 0..18), ising_trotter uses
``step_{i}.pk`` -> entry ``i+1`` (i = 0..17), grover uses ``step_{i+1}.pk`` ->
entry ``i+1`` (i = 0..4).
"""

import pickle
import os, json
from qiskit import QuantumCircuit
from qiskit.circuit import qpy_serialization
import qiskit

def get_active_qubits_dynamics(circuit):
    operations = circuit.data
    active_qubits = set()

    for op in operations:
        if isinstance(op[0], qiskit.circuit.Barrier) or isinstance(op[0], qiskit.circuit.Delay):
            continue
        for qubit in op[1]:
            active_qubits.add(qubit.index)

    return active_qubits

def remove_idle_qubits(circuit, layout):
    # Get active qubits, which excludes 'Delay' and 'Barrier' operations
    active_qubits = get_active_qubits_dynamics(circuit)

    # Create a new list ordered_qubits, preserving the order in layout
    ordered_qubits = [qubit for qubit in layout if qubit in active_qubits]

    # Keep the original ClassicalRegisters intact
    new_qc = QuantumCircuit(len(ordered_qubits))
    for cr in circuit.cregs:
        new_qc.add_register(cr)

    # Go through each instruction and replace the qubit indices with the indices of ordered qubits
    for instr, qargs, cargs in circuit.data:
        ordered_qarg_indices = [ordered_qubits.index(qubit.index) for qubit in qargs if qubit.index in ordered_qubits]
        # Since we're keeping the classical bits as they were, there's no need to change the indices
        if ordered_qarg_indices:  # If list is not empty, means qubits are involved in operation
            new_qc.append(instr, qargs=ordered_qarg_indices, cargs=cargs)

    return new_qc

def export_first_40_circuits(pk_filename, output_dir="exported_circuits", max_circuits=40, cir_type='random'):
    """
    Export the first N quantum circuits from a .pk file.

    Args:
        pk_filename: .pk file name (without path)
        output_dir: Output directory
        max_circuits: Maximum number of circuits to export
    """

    # Build the full file path
    if cir_type == 'random':
        pk_file_path = os.path.join("random_gen", pk_filename)
    elif cir_type == 'ising_trotter':
        pk_file_path = os.path.join("trotter_gen", pk_filename)
    elif cir_type == 'grover':
        pk_file_path = os.path.join("grover_gen", pk_filename)
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing file: {pk_file_path}")
    print(f"Export directory: {output_dir}")
    print(f"Max export count: {max_circuits}")

    try:
        # 1. Load .pk file
        with open(pk_file_path, 'rb') as f:
            data = pickle.load(f)
        print("✅ PK file loaded successfully")

    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return []

    # 2. Collect all quantum circuits
    all_circuits = []

    def collect_circuits(obj, current_path=""):
        """Recursively collect all quantum circuits."""
        if isinstance(obj, dict) and 'qasm_string' in obj:
            qasm_str = obj['qasm_string']
            circuit = QuantumCircuit.from_qasm_str(qasm_str)
            all_circuits.append((current_path, circuit))
        elif isinstance(obj, QuantumCircuit):
            all_circuits.append((current_path, obj))
        elif isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{current_path}.{key}" if current_path else key
                print(key)
                collect_circuits(value, new_path)
            print(2)
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                new_path = f"{current_path}[{i}]"
                collect_circuits(item, new_path)

    collect_circuits(data)
    print(f"🔍 Found {len(all_circuits)} quantum circuits in the file")

    # 3. Limit the number of exports
    circuits_to_export = all_circuits[:max_circuits]
    print(f"📤 Preparing to export first {len(circuits_to_export)} circuits")

    # 4. Export circuits
    exported_files = []

    for i, (path, circuit) in enumerate(circuits_to_export):
        circuit_name = f"circuit_{i}"
        layout = list(range(circuit.num_qubits))
        # circuit = remove_idle_qubits(circuit, layout)
        try:
            # Export as QASM format
            qasm_filename = os.path.join(output_dir, f"{circuit_name}.qasm")
            with open(qasm_filename, 'w') as f:
                f.write(circuit.qasm())

            # Export as QPY format (Qiskit native binary format)
            qpy_filename = os.path.join(output_dir, f"{circuit_name}.qpy")
            with open(qpy_filename, 'wb') as f:
                qpy_serialization.dump(circuit, f)

            # Record export information
            circuit_info = {
                'index': i,
                'original_path': path,
                'name': circuit_name,
                'num_qubits': circuit.num_qubits,
                'num_clbits': circuit.num_clbits,
                'depth': circuit.depth(),
                'size': circuit.size(),
                'operations': circuit.count_ops(),
                'qasm_file': qasm_filename,
                'qpy_file': qpy_filename
            }
            exported_files.append(circuit_info)

            print(f"✅ [{i + 1:2d}/{len(circuits_to_export)}] {circuit_name}: "
                  f"{circuit.num_qubits} qubits, {circuit.depth()} depth, "
                  f"{circuit.size()} operations")

        except Exception as e:
            print(f"❌ Error exporting circuit {circuit_name}: {e}")

    # 5. Generate export summary
    generate_export_summary(exported_files, output_dir)

    print(f"\n🎉 Export completed! Successfully exported {len(exported_files)} quantum circuits to '{output_dir}' directory")
    return exported_files

def generate_export_summary(exported_files, output_dir):
    """Generate export summary file."""
    summary_file = os.path.join(output_dir, "export_summary.txt")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Quantum Circuit Export Summary\n")
        f.write("=" * 50 + "\n\n")
        # f.write(f"Export time: {import datetime; datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total exported: {len(exported_files)}\n\n")

        f.write("Circuit Details:\n")
        f.write("-" * 80 + "\n")
        for circuit_info in exported_files:
            f.write(f"Circuit {circuit_info['index']}: {circuit_info['name']}\n")
            f.write(f"  Original path: {circuit_info['original_path']}\n")
            f.write(f"  Qubits: {circuit_info['num_qubits']}, Classical bits: {circuit_info['num_clbits']}\n")
            f.write(f"  Depth: {circuit_info['depth']}, Operations: {circuit_info['size']}\n")
            f.write(f"  Operation counts: {circuit_info['operations']}\n")
            f.write(f"  Files: {circuit_info['qasm_file']}\n")
            f.write(f"         {circuit_info['qpy_file']}\n")
            f.write("\n")

    print(f"📄 Export summary saved to: {summary_file}")

def visualize_some_circuits(exported_files, num_to_visualize=5):
    """Visualize the first few circuits (optional feature)."""
    print(f"\n🎨 Visualizing first {num_to_visualize} circuits:")

    for i, circuit_info in enumerate(exported_files[:num_to_visualize]):
        try:
            # Reload circuit for visualization
            with open(circuit_info['qpy_file'], 'rb') as f:
                circuit = qpy_serialization.load(f)[0]

            print(f"\nCircuit {circuit_info['index']}: {circuit_info['name']}")
            print(circuit.draw(fold=-1))  # fold=-1 means no folding

        except Exception as e:
            print(f"❌ Error visualizing circuit {circuit_info['name']}: {e}")

# Usage example
if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)

    cir_type = config["cir_type"]
    for i in range(0, 20):
        if cir_type == 'random':
            if i > 18:
                break
            pk_filename = f"step_{2 * i}.pk"
            circuit_dir = os.path.join("random_circuits", f"random_circuits_{i+1}")
        elif cir_type == 'ising_trotter':
            if i > 17:
                break
            pk_filename = f"step_{i}.pk"
            circuit_dir = os.path.join("ising_trotter_circuits", f"ising_trotter_circuits_{i+1}")
        elif cir_type == 'grover':
            if i > 4:
                break
            pk_filename = f"step_{i+1}.pk"
            circuit_dir = os.path.join("grover_circuits", f"grover_circuits_{i + 1}")

        # Export first 40 circuits
        exported_circuits = export_first_40_circuits(
            pk_filename=pk_filename,
            output_dir=circuit_dir,
            max_circuits=200,
            cir_type=cir_type
        )

        # Optional: visualize the first 5 circuits
        if exported_circuits:
            visualize_some_circuits(exported_circuits, num_to_visualize=5)

            # Print some statistics
            total_qubits = sum(circuit['num_qubits'] for circuit in exported_circuits)
            avg_qubits = total_qubits / len(exported_circuits)
            print(f"\n📊 Statistics:")
            print(f"  Average number of qubits: {avg_qubits:.1f}")
            print(f"  Total operations: {sum(circuit['size'] for circuit in exported_circuits)}")
"""Extract and visualise the FakeQuito / FakeLima calibration snapshots.

Writes CSV tables (qubit parameters, two-qubit gate errors), the native gate-map
figure, a custom coupling map annotated with gate errors, and the Qiskit error
map.

Reads nothing but the built-in Qiskit fake backends.  Note that the Qiskit error
map (``plot_error_map``) needs the Graphviz *binaries*; without them the script
stops after the CSVs and the first three figures, which is what the archived
output shows.
"""

# -*- coding: utf-8 -*-
"""
提取 FakeQuito 和 FakeLima 的设备信息并保存到 Device_properties 文件夹
包含：表格 CSV、原生架构图、带错误率标注的自定义拓扑图、错误率热图
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from qiskit.providers.fake_provider import FakeQuito, FakeLima
from qiskit.visualization import plot_gate_map, plot_error_map

# ---------------------------- 设置输出目录 ---------------------------- #
output_dir = "Device_properties"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------- 加载后端 ---------------------------- #
backend_quito = FakeQuito()
backend_lima = FakeLima()

props_quito = backend_quito.properties()
props_lima = backend_lima.properties()
config_quito = backend_quito.configuration()
config_lima = backend_lima.configuration()


def get_backend_info(backend, props, config):
    """提取单个后端的详细参数，返回 DataFrame 和连接信息"""
    qubits_info = []
    num_qubits = config.n_qubits

    for q in range(num_qubits):
        freq = props.qubit_property(q, 'frequency')[0]  # Hz
        f10_GHz = freq / 1e9

        anharm = props.qubit_property(q, 'anharmonicity')[0]  # Hz
        eta_MHz = anharm / 1e6

        t1 = props.qubit_property(q, 'T1')[0]  # s
        t1_us = t1 * 1e6

        t2 = props.qubit_property(q, 'T2')[0]  # s
        t2_us = t2 * 1e6

        prob_meas0_prep1 = props.qubit_property(q, 'prob_meas0_prep1')[0]
        prob_meas1_prep0 = props.qubit_property(q, 'prob_meas1_prep0')[0]
        F0 = 1 - prob_meas1_prep0
        F1 = 1 - prob_meas0_prep1

        readout_freq = np.nan

        qubits_info.append({
            'Qubit index': q,
            'Qubit frequency, f10 (GHz)': f10_GHz,
            'Readout frequency, f1 (GHz)': readout_freq,
            'Anharmonicity, η (MHz)': eta_MHz,
            'Relaxation time, T1 (μs)': t1_us,
            'Coherence time, T2 (μs)': t2_us,
            'Readout fidelity of state |0⟩, F0': F0,
            'Readout fidelity of state |1⟩, F1': F1
        })

    df_qubits = pd.DataFrame(qubits_info)

    # 提取两比特门错误率
    gate_errors = []
    for gate in props.gates:
        if len(gate.qubits) == 2:
            gate_name = gate.gate
            q0, q1 = gate.qubits
            try:
                error = props.gate_property(gate_name, (q0, q1), 'gate_error')[0]
            except:
                fidelity = props.gate_property(gate_name, (q0, q1), 'gate_fidelity')[0]
                error = 1 - fidelity
            gate_errors.append({
                'Gate': gate_name,
                'Qubits': f"{q0}-{q1}",
                'Error rate': error
            })

    df_gates = pd.DataFrame(gate_errors)
    coupling_map = config.coupling_map

    return df_qubits, df_gates, coupling_map


import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

def plot_coupling_with_errors(backend_name, coupling_map, df_gates, output_path, spacing):
    """
    绘制带边错误率标注的芯片拓扑图，所有连线均为水平或垂直，且不遗漏任何边。
    """
    G = nx.Graph()
    G.add_edges_from(coupling_map)

    # 节点列表（按编号排序）
    nodes = sorted(G.nodes())

    # ---------- 自动网格布局 ----------
    pos = {}
    # 选择度数最小的节点作为起点，通常位于角落
    start = min(nodes, key=lambda n: G.degree(n))
    pos[start] = (0, 0)
    visited = {start}
    queue = deque([start])

    # 方向向量：右、左、上、下
    # 方向向量：右、左、上、下，乘以 spacing
    directions = [(spacing, 0), (-spacing, 0), (0, spacing), (0, -spacing)]

    while queue:
        node = queue.popleft()
        x, y = pos[node]
        for neighbor in G.neighbors(node):
            if neighbor in visited:
                continue
            placed = False
            for dx, dy in directions:
                nx_pos = x + dx
                ny_pos = y + dy
                if (nx_pos, ny_pos) not in pos.values():
                    pos[neighbor] = (nx_pos, ny_pos)
                    visited.add(neighbor)
                    queue.append(neighbor)
                    placed = True
                    break
            if not placed:
                offset = spacing
                while (x + offset, y) in pos.values():
                    offset += spacing
                pos[neighbor] = (x + offset, y)
                visited.add(neighbor)
                queue.append(neighbor)

    # ---------- 构建边错误率字典 ----------
    edge_error = {}
    for _, row in df_gates.iterrows():
        q0, q1 = map(int, row['Qubits'].split('-'))
        edge = tuple(sorted((q0, q1)))
        if edge not in edge_error:
            edge_error[edge] = row['Error rate']

    # ---------- 绘制 ----------
    # ---------- 绘制（添加坐标轴限制） ----------
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1600, node_shape='o')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=4, alpha=0.5)
        x_text = (x0 + x1) / 2
        y_text = (y0 + y1) / 2
        error_val = edge_error.get(tuple(sorted(edge)), np.nan)
        if not np.isnan(error_val):
            plt.text(x_text, y_text, f"{error_val:.4f}", fontsize=16,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
                     ha='center', va='center')

    # ---- 关键：手动缩小坐标轴范围，使连线变短 ----
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    margin = 0.3  # 可根据需要调整，值越小边距越少，连线看起来越短
    plt.xlim(min(xs) - margin, max(xs) + margin)
    plt.ylim(min(ys) - margin, max(ys) + margin)

    plt.title(f"{backend_name} Coupling Map with Gate Errors", fontsize=14)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------- 获取信息 ---------------------------- #
df_q_quito, df_g_quito, cmap_quito = get_backend_info(backend_quito, props_quito, config_quito)
df_q_lima, df_g_lima, cmap_lima = get_backend_info(backend_lima, props_lima, config_lima)

# ---------------------------- 保存表格为 CSV ---------------------------- #
df_q_quito.to_csv(os.path.join(output_dir, "FakeQuito_qubits.csv"), index=False)
df_g_quito.to_csv(os.path.join(output_dir, "FakeQuito_gates.csv"), index=False)
df_q_lima.to_csv(os.path.join(output_dir, "FakeLima_qubits.csv"), index=False)
df_g_lima.to_csv(os.path.join(output_dir, "FakeLima_gates.csv"), index=False)

# 打印预览
print("=" * 80)
print("FakeQuito 比特参数表（前5行）")
print("=" * 80)
print(df_q_quito.round(4).head().to_string(index=False))

print("\n" + "=" * 80)
print("FakeQuito 两比特门错误率")
print("=" * 80)
print(df_g_quito.round(6).to_string(index=False))

print("\n" + "=" * 80)
print("FakeLima 比特参数表（前5行）")
print("=" * 80)
print(df_q_lima.round(4).head().to_string(index=False))

print("\n" + "=" * 80)
print("FakeLima 两比特门错误率")
print("=" * 80)
print(df_g_lima.round(6).to_string(index=False))

# ---------------------------- 绘制并保存原生架构图 ---------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_gate_map(backend_quito, ax=axes[0], plot_directed=False)
axes[0].set_title("FakeQuito Qubit Layout")
plot_gate_map(backend_lima, ax=axes[1], plot_directed=False)
axes[1].set_title("FakeLima Qubit Layout")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fake_backends_layout.png"), dpi=150, bbox_inches='tight')
plt.close()

# ---------------------------- 绘制带错误率标注的自定义拓扑图 ---------------------------- #
plot_coupling_with_errors("FakeQuito", cmap_quito, df_g_quito,
                          os.path.join(output_dir, "FakeQuito_layout_with_errors.png"), spacing=0.3)
plot_coupling_with_errors("FakeLima", cmap_lima, df_g_lima,
                          os.path.join(output_dir, "FakeLima_layout_with_errors.png"), spacing=0.3)

# ---------------------------- 绘制并保存错误率热图 ---------------------------- #
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
plot_error_map(backend_quito, ax=axes2[0])
axes2[0].set_title("FakeQuito Error Map")
plot_error_map(backend_lima, ax=axes2[1])
axes2[1].set_title("FakeLima Error Map")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fake_backends_error_map.png"), dpi=150, bbox_inches='tight')
plt.close()

print("\n所有表格和图片已保存到文件夹:", os.path.abspath(output_dir))
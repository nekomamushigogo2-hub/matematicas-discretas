"""
Serpientes y Escaleras — Cadenas de Markov
Proyecto 20 · Matemáticas Discretas y Pensamiento Lógico · 2026-1

Ejecutar con:
    pip install streamlit numpy plotly
    streamlit run serpientes_escaleras_markov.py
"""

import streamlit as st
import numpy as np
import random
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Serpientes y Escaleras — Markov",
    page_icon="🎲",
    layout="wide",
)

# ─────────────────────────────────────────────
# ESTILOS CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.main-header { margin-bottom: 1.5rem; }
.main-header h1 { font-size: 22px; font-weight: 600; margin-bottom: 2px; }
.main-header p { font-size: 13px; color: #888; }

.card {
    background: #f8f9fa;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 10px;
}
.card-title {
    font-size: 11px;
    font-weight: 600;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
}
.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 5px;
    font-size: 13px;
}
.metric-label { color: #555; }
.metric-value { font-weight: 600; color: #185FA5; }

.board-grid {
    display: grid;
    grid-template-columns: repeat(10, 1fr);
    gap: 3px;
    max-width: 520px;
}
.cell {
    aspect-ratio: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border-radius: 5px;
    border: 1px solid #dde2e8;
    font-size: 9px;
    font-weight: 600;
    cursor: default;
    position: relative;
    user-select: none;
    transition: border-color 0.15s;
}
.cell-num { font-size: 8px; color: #aaa; line-height: 1; }
.cell-icon { font-size: 12px; line-height: 1; }
.cell.normal { background: #fff; }
.cell.snake-head { background: #FAECE7; border-color: #D85A30; color: #993C1D; }
.cell.snake-tail { background: #FFF3EF; }
.cell.ladder-bottom { background: #EAF3DE; border-color: #639922; color: #3B6D11; }
.cell.ladder-top { background: #F3F9EA; }
.cell.win-cell { background: #EAF3DE; border-color: #639922; }
.cell.current { border: 2px solid #378ADD !important; box-shadow: 0 0 0 3px rgba(55,138,221,0.2); }

.game-log {
    height: 110px;
    overflow-y: auto;
    font-size: 12px;
    line-height: 1.9;
    padding: 8px 10px;
    background: #f4f6f8;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    font-family: 'IBM Plex Mono', monospace;
}
.event { color: #444; }
.snake-event { color: #993C1D; font-weight: 500; }
.ladder-event { color: #3B6D11; font-weight: 500; }
.win-event { color: #185FA5; font-weight: 600; }

.legend-row { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: #666; margin-bottom: 10px; }
.legend-item { display: flex; align-items: center; gap: 5px; }
.legend-dot { width: 12px; height: 12px; border-radius: 3px; border: 1px solid #ccc; }

.top-cell-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.rank-badge {
    width: 24px; height: 24px;
    background: #185FA5; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 600; color: #fff; flex-shrink: 0;
}
.prob-bar-bg { flex: 1; height: 6px; border-radius: 3px; background: #e5e7eb; }
.prob-bar-fill { height: 100%; border-radius: 3px; background: #378ADD; }

.answer-box {
    background: #f4f6f8;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 12px;
    border-left: 3px solid #378ADD;
}
.answer-box h3 { font-size: 13px; font-weight: 600; margin-bottom: 6px; color: #1a1a1a; }
.answer-box p { font-size: 13px; color: #555; line-height: 1.65; }
.hl { color: #185FA5; font-weight: 600; }

.big-metric {
    text-align: center;
    background: #f4f6f8;
    border-radius: 10px;
    padding: 1rem 0.5rem;
}
.big-metric .label { font-size: 11px; color: #888; margin-bottom: 4px; }
.big-metric .value { font-size: 24px; font-weight: 600; color: #185FA5; }

.dice-display { font-size: 40px; line-height: 1; }

.prob-grid {
    display: grid;
    grid-template-columns: repeat(10, 1fr);
    gap: 2px;
    margin-bottom: 10px;
}
.prob-cell {
    aspect-ratio: 1;
    border-radius: 3px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 7.5px;
    font-weight: 600;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LÓGICA DE MARKOV
# ─────────────────────────────────────────────
SERPIENTES = {16:6, 47:26, 49:11, 56:53, 62:19,
              64:60, 87:24, 93:73, 95:75, 99:78}
ESCALERAS  = {1:38, 4:14, 9:31, 20:38, 28:84,
              40:59, 51:67, 63:81, 71:91}

def get_next(pos, d):
    sig = pos + d
    if sig > 100: return pos
    if sig in SERPIENTES: return SERPIENTES[sig]
    if sig in ESCALERAS:  return ESCALERAS[sig]
    return sig

@st.cache_data
def compute_win_probs():
    prob = np.zeros(101)
    prob[100] = 1.0
    for i in range(99, -1, -1):
        for d in range(1, 7):
            prob[i] += prob[get_next(i, d)] / 6.0
    return prob

@st.cache_data
def compute_expected_turns():
    A = np.eye(100)
    b = np.ones(100)
    for i in range(100):
        for d in range(1, 7):
            j = get_next(i, d)
            if j < 100:
                A[i][j] -= 1.0 / 6
    return np.linalg.solve(A, b)

WIN_PROBS = compute_win_probs()
EXP_TURNS = compute_expected_turns()

# ─────────────────────────────────────────────
# ESTADO DE SESIÓN
# ─────────────────────────────────────────────
defaults = dict(pos=0, game_over=False, turns=0,
                snake_hits=0, ladder_hits=0,
                log=["Haz clic en 'Tirar dado' para comenzar."],
                last_roll=None)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# HELPERS VISUALES
# ─────────────────────────────────────────────
def cell_class(n):
    if n in SERPIENTES:                          return "snake-head"
    if n in SERPIENTES.values():                 return "snake-tail"
    if n in ESCALERAS:                           return "ladder-bottom"
    if n in ESCALERAS.values():                  return "ladder-top"
    if n == 100:                                 return "win-cell"
    return "normal"

def cell_icon(n):
    if n in SERPIENTES: return "🐍"
    if n in ESCALERAS:  return "🪜"
    if n == 100:        return "🏆"
    return ""

def prob_color(p):
    stops = ["#E6F1FB","#B5D4F4","#85B7EB","#378ADD","#185FA5","#0C447C"]
    idx = min(int(p * len(stops)), len(stops)-1)
    return stops[idx]

def build_board_html(current_pos):
    html = '<div class="board-grid">'
    for row in range(9, -1, -1):
        for col in range(10):
            if row % 2 == 0:
                num = row * 10 + (10 - col)
            else:
                num = row * 10 + col + 1
            cls = cell_class(num)
            if num == current_pos:
                cls += " current"
            icon = cell_icon(num)
            html += f"""<div class="cell {cls}" title="{num}">
                <span class="cell-num">{num}</span>
                <span class="cell-icon">{icon}</span>
            </div>"""
    html += "</div>"
    return html

def build_prob_grid_html(selected=None):
    html = '<div class="prob-grid">'
    for row in range(9, -1, -1):
        for col in range(10):
            if row % 2 == 0:
                num = row * 10 + (10 - col)
            else:
                num = row * 10 + col + 1
            p = WIN_PROBS[num] if num <= 100 else 0
            bg = prob_color(p)
            txt_color = "#E6F1FB" if p > 0.5 else "#0C447C"
            border = "2px solid #185FA5" if num == selected else "none"
            html += f'<div class="prob-cell" style="background:{bg};color:{txt_color};border:{border}" title="Casilla {num}: {p*100:.1f}%">{num}</div>'
    html += "</div>"
    return html

def build_log_html(log_lines):
    html = '<div class="game-log">'
    for line in log_lines:
        if "🐍" in line:
            css = "snake-event"
        elif "🪜" in line:
            css = "ladder-event"
        elif "🏆" in line:
            css = "win-event"
        else:
            css = "event"
        html += f'<div class="{css}">{line}</div>'
    html += "</div>"
    return html

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🎲 Serpientes y Escaleras — Cadenas de Markov</h1>
  <p>Proyecto 20 · Matemáticas Discretas y Pensamiento Lógico · 2026-1</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PESTAÑAS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎮 Tablero", "📊 Probabilidades", "🔄 Simulación", "📈 Análisis", "❓ Preguntas"
])

# ══════════════════════════════════════════════
# TAB 1 — TABLERO
# ══════════════════════════════════════════════
with tab1:
    faces = ["⚀","⚁","⚂","⚃","⚄","⚅"]

    col_dice, col_info, col_btns = st.columns([1, 3, 2])

    with col_dice:
        roll_face = faces[st.session_state.last_roll - 1] if st.session_state.last_roll else "🎲"
        st.markdown(f'<div class="dice-display">{roll_face}</div>', unsafe_allow_html=True)

    with col_info:
        pos_label = "Inicio" if st.session_state.pos == 0 else f"Casilla {st.session_state.pos}"
        win_pct = WIN_PROBS[st.session_state.pos] * 100
        st.markdown(f"**Posición:** {pos_label}")
        st.markdown(f"**Probabilidad de ganar:** :blue[{win_pct:.1f}%]")

    with col_btns:
        c1, c2 = st.columns(2)
        with c1:
            tirar = st.button("🎲 Tirar dado", use_container_width=True, type="primary",
                              disabled=st.session_state.game_over)
        with c2:
            reiniciar = st.button("↺ Reiniciar", use_container_width=True)

    if tirar and not st.session_state.game_over:
        roll = random.randint(1, 6)
        st.session_state.last_roll = roll
        prev = st.session_state.pos
        next_pos = prev + roll
        if next_pos > 100:
            next_pos = prev
        st.session_state.turns += 1
        msg = f"Turno {st.session_state.turns}: tiré {roll}, fui de {prev} a {next_pos}"
        if next_pos in SERPIENTES:
            st.session_state.snake_hits += 1
            dest = SERPIENTES[next_pos]
            msg += f" — 🐍 serpiente! bajo a {dest}"
            next_pos = dest
        elif next_pos in ESCALERAS:
            st.session_state.ladder_hits += 1
            dest = ESCALERAS[next_pos]
            msg += f" — 🪜 escalera! subo a {dest}"
            next_pos = dest
        st.session_state.pos = next_pos
        st.session_state.log.append(msg)
        if next_pos >= 100:
            st.session_state.game_over = True
            st.session_state.log.append(f"🏆 ¡Ganaste en {st.session_state.turns} turnos!")

    if reiniciar:
        for k, v in defaults.items():
            st.session_state[k] = v if not isinstance(v, list) else list(v)
        st.rerun()

    st.markdown(build_log_html(st.session_state.log), unsafe_allow_html=True)
    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    col_board, col_side = st.columns([3, 1.4])

    with col_board:
        st.markdown("""
        <div class="legend-row">
          <div class="legend-item"><div class="legend-dot" style="background:#FAECE7;border-color:#D85A30"></div>Cabeza serpiente</div>
          <div class="legend-item"><div class="legend-dot" style="background:#EAF3DE;border-color:#639922"></div>Base escalera</div>
          <div class="legend-item"><div class="legend-dot" style="background:#E6F1FB;border-color:#378ADD"></div>Posición actual</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(build_board_html(st.session_state.pos), unsafe_allow_html=True)

    with col_side:
        st.markdown('<div class="card"><div class="card-title">Serpientes</div>', unsafe_allow_html=True)
        for k, v in sorted(SERPIENTES.items()):
            st.markdown(f'<div class="metric-row"><span class="metric-label">{k} → {v}</span><span style="color:#D85A30;font-size:11px;font-weight:600">-{k-v}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">Escaleras</div>', unsafe_allow_html=True)
        for k, v in sorted(ESCALERAS.items()):
            st.markdown(f'<div class="metric-row"><span class="metric-label">{k} → {v}</span><span style="color:#3B6D11;font-size:11px;font-weight:600">+{v-k}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
          <div class="card-title">Estado del juego</div>
          <div class="metric-row"><span class="metric-label">Turnos jugados</span><span class="metric-value">{st.session_state.turns}</span></div>
          <div class="metric-row"><span class="metric-label">Serpientes</span><span style="color:#D85A30;font-weight:600">{st.session_state.snake_hits}</span></div>
          <div class="metric-row"><span class="metric-label">Escaleras</span><span style="color:#3B6D11;font-weight:600">{st.session_state.ladder_hits}</span></div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — PROBABILIDADES
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p style="font-size:13px;color:#888;margin-bottom:10px">Mapa de calor: probabilidad de ganar desde cada casilla. Color más oscuro = mayor probabilidad.</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="legend-row">
      <span style="font-size:12px;color:#888">Baja probabilidad</span>
      <div style="display:flex;gap:3px;align-items:center">
        <div style="width:16px;height:16px;border-radius:2px;background:#E6F1FB"></div>
        <div style="width:16px;height:16px;border-radius:2px;background:#85B7EB"></div>
        <div style="width:16px;height:16px;border-radius:2px;background:#378ADD"></div>
        <div style="width:16px;height:16px;border-radius:2px;background:#185FA5"></div>
        <div style="width:16px;height:16px;border-radius:2px;background:#0C447C"></div>
      </div>
      <span style="font-size:12px;color:#888">Alta probabilidad</span>
    </div>
    """, unsafe_allow_html=True)

    selected_cell = st.selectbox("Selecciona una casilla para ver detalles:",
                                  options=list(range(1, 101)), index=0, key="prob_select")

    col_grid, col_detail = st.columns([2, 1])

    with col_grid:
        st.markdown(build_prob_grid_html(selected=selected_cell), unsafe_allow_html=True)
        st.markdown('<p style="font-size:11px;color:#aaa;margin-top:4px">Selecciona una casilla en el menú para ver detalles.</p>', unsafe_allow_html=True)

    with col_detail:
        p = WIN_PROBS[selected_cell]
        extra = ""
        if selected_cell in SERPIENTES:
            extra = f'<div style="color:#993C1D;font-size:12px;margin-top:6px">🐍 Serpiente a {SERPIENTES[selected_cell]}</div>'
        elif selected_cell in ESCALERAS:
            extra = f'<div style="color:#3B6D11;font-size:12px;margin-top:6px">🪜 Escalera a {ESCALERAS[selected_cell]}</div>'
        bar_pct = int(p * 100)
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Casilla {selected_cell}</div>
          <div class="metric-row">
            <span class="metric-label">Prob. de ganar</span>
            <span class="metric-value">{p*100:.2f}%</span>
          </div>
          <div style="height:6px;border-radius:3px;background:#e5e7eb;margin-top:4px">
            <div style="width:{bar_pct}%;height:100%;border-radius:3px;background:#378ADD"></div>
          </div>
          {extra}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="font-size:13px;font-weight:600;margin-bottom:10px;margin-top:8px">Top 5 casillas</div>', unsafe_allow_html=True)
        top5 = sorted(range(1, 100), key=lambda i: WIN_PROBS[i], reverse=True)[:5]
        for rank, cell in enumerate(top5, 1):
            p5 = WIN_PROBS[cell]
            icon = "🪜" if cell in ESCALERAS else ""
            bar = int(p5 * 100)
            st.markdown(f"""
            <div class="top-cell-row">
              <div class="rank-badge">{rank}</div>
              <div style="flex:1">
                <div style="font-size:12px;font-weight:600">Casilla {cell} {icon}</div>
                <div style="height:5px;border-radius:3px;background:#e5e7eb;margin-top:3px">
                  <div style="width:{bar}%;height:100%;border-radius:3px;background:#378ADD"></div>
                </div>
              </div>
              <div style="font-size:13px;font-weight:600;color:#185FA5">{p5*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 — SIMULACIÓN
# ══════════════════════════════════════════════
with tab3:
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        n_sims = st.slider("Número de partidas a simular", 100, 10000, 1000, step=100)
    with col_sc2:
        start_pos = st.slider("Posición inicial", 0, 95, 0, step=1)

    if st.button("▶ Ejecutar simulación", type="primary"):
        turn_counts = []
        for _ in range(n_sims):
            pos, t = start_pos, 0
            while pos < 100 and t < 2000:
                roll = random.randint(1, 6)
                pos = get_next(pos, roll)
                t += 1
            turn_counts.append(t)

        avg = np.mean(turn_counts)
        mn  = int(np.min(turn_counts))
        mx  = int(np.max(turn_counts))
        std = np.std(turn_counts)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="big-metric"><div class="label">Partidas</div><div class="value">{n_sims:,}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="big-metric"><div class="label">Promedio turnos</div><div class="value">{avg:.1f}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="big-metric"><div class="label">Mín. turnos</div><div class="value">{mn}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="big-metric"><div class="label">Máx. turnos</div><div class="value">{mx}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:16px;font-size:13px;font-weight:600;margin-bottom:4px'>Distribución de número de turnos para terminar</div>", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=turn_counts,
            nbinsx=30,
            marker_color="#378ADD",
            marker_line_color="#185FA5",
            marker_line_width=0.5,
            name="Partidas",
        ))
        fig.add_vline(x=avg, line_dash="dash", line_color="#D85A30",
                      annotation_text=f"Promedio: {avg:.1f}", annotation_position="top right")
        fig.update_layout(
            height=240,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Turnos",
            yaxis_title="Frecuencia",
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(gridcolor="#f0f0f0"),
            yaxis=dict(gridcolor="#f0f0f0"),
            font=dict(family="IBM Plex Sans", size=11),
            bargap=0.05,
        )
        st.plotly_chart(fig, use_container_width=True)

        expected_analytical = EXP_TURNS[start_pos]
        diff_pct = abs(avg - expected_analytical) / expected_analytical * 100
        st.markdown(f"""
        <div class="card" style="margin-top:8px">
          <div class="card-title">Comparación con modelo de Markov</div>
          <div class="metric-row"><span class="metric-label">Valor analítico (Markov)</span><span class="metric-value">{expected_analytical:.2f} turnos</span></div>
          <div class="metric-row"><span class="metric-label">Simulación Monte Carlo</span><span class="metric-value">{avg:.2f} turnos</span></div>
          <div class="metric-row"><span class="metric-label">Diferencia relativa</span><span class="metric-value">{diff_pct:.2f}%</span></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#aaa;font-size:13px;margin-top:10px">Configura los parámetros y ejecuta la simulación.</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4 — ANÁLISIS
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div style="font-size:13px;font-weight:600;margin-bottom:8px">Probabilidad de ganar por posición inicial</div>', unsafe_allow_html=True)

    labels = list(range(1, 101))
    probs  = [WIN_PROBS[i] * 100 for i in labels]
    colors = ["#D85A30" if i in SERPIENTES else "#639922" if i in ESCALERAS else "#B5D4F4" for i in labels]

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=labels, y=probs,
        marker_color=colors,
        marker_line_width=0,
        name="% prob. ganar",
    ))
    fig1.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Casilla",
        yaxis_title="Probabilidad (%)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#f0f0f0", dtick=10),
        yaxis=dict(gridcolor="#f0f0f0"),
        font=dict(family="IBM Plex Sans", size=11),
        showlegend=False,
        bargap=0.1,
    )
    fig1.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
        marker=dict(size=10, color="#D85A30", symbol="square"), name="Serpiente"))
    fig1.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
        marker=dict(size=10, color="#639922", symbol="square"), name="Escalera"))
    fig1.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
        marker=dict(size=10, color="#B5D4F4", symbol="square"), name="Normal"))
    fig1.update_layout(showlegend=True, legend=dict(orientation="h", y=1.15, x=0))
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown('<div style="font-size:13px;font-weight:600;margin-bottom:8px;margin-top:8px">Impacto de serpientes y escaleras</div>', unsafe_allow_html=True)

    impact_items = []
    for k, v in SERPIENTES.items():
        delta = (WIN_PROBS[v] - WIN_PROBS[k]) * 100
        impact_items.append({"label": f"S: {k}→{v}", "delta": delta, "type": "snake"})
    for k, v in ESCALERAS.items():
        delta = (WIN_PROBS[v] - WIN_PROBS[k]) * 100
        impact_items.append({"label": f"E: {k}→{v}", "delta": delta, "type": "ladder"})
    impact_items.sort(key=lambda x: x["delta"], reverse=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        y=[x["label"] for x in impact_items],
        x=[x["delta"] for x in impact_items],
        orientation="h",
        marker_color=["#639922" if x["type"]=="ladder" else "#D85A30" for x in impact_items],
        marker_line_width=0,
    ))
    fig2.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Cambio en % prob. ganar",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#aaa"),
        yaxis=dict(gridcolor="#f0f0f0"),
        font=dict(family="IBM Plex Sans", size=11),
        showlegend=False,
        bargap=0.25,
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📋 Tabla detallada de turnos esperados por casilla"):
        st.markdown('<div style="font-size:12px;color:#888;margin-bottom:6px">Turnos esperados E[i] para ganar desde cada casilla según el modelo de Markov</div>', unsafe_allow_html=True)
        col_e1, col_e2, col_e3, col_e4, col_e5 = st.columns(5)
        cols_exp = [col_e1, col_e2, col_e3, col_e4, col_e5]
        chunk = 20
        for idx, col in enumerate(cols_exp):
            with col:
                for i in range(idx*chunk, (idx+1)*chunk):
                    tipo = "🐍" if i in SERPIENTES else "🪜" if i in ESCALERAS else ""
                    col.markdown(f"<span style='font-size:11px;font-family:IBM Plex Mono,monospace'>"
                                 f"**{i}** {tipo}: {EXP_TURNS[i]:.1f}</span>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 5 — PREGUNTAS
# ══════════════════════════════════════════════
with tab5:
    p = WIN_PROBS

    top_by_prob = sorted(range(1, 100), key=lambda i: WIN_PROBS[i], reverse=True)
    best_cell   = top_by_prob[0]
    best_prob   = WIN_PROBS[best_cell]

    snake_avg  = np.mean([WIN_PROBS[k] for k in SERPIENTES]) * 100
    ladder_avg = np.mean([WIN_PROBS[k] for k in ESCALERAS]) * 100

    answers = [
        {
            "q": "1. ¿Qué casillas tienen la mayor probabilidad de permitir una victoria?",
            "a": f'La casilla con mayor probabilidad de ganar es la <span class="hl">{best_cell}</span> con <span class="hl">{best_prob*100:.1f}%</span>. Las casillas base de escaleras tienen probabilidades muy altas porque garantizan un avance inmediato. En particular, la escalera 28→84 es la más poderosa: coloca al jugador a solo 16 casillas de la meta.'
        },
        {
            "q": "2. ¿Cómo influyen las serpientes y las escaleras en las probabilidades de ganar?",
            "a": f'Las serpientes reducen la probabilidad de ganar: estar en una cabeza de serpiente da en promedio <span class="hl">{snake_avg:.1f}%</span>. Las escaleras la aumentan: caer en una base de escalera da promedio <span class="hl">{ladder_avg:.1f}%</span>. La diferencia es de <span class="hl">{ladder_avg-snake_avg:.1f} puntos porcentuales</span> en favor de las escaleras.'
        },
        {
            "q": "3. ¿Qué estrategia puede mejorar las probabilidades de éxito en el juego?",
            "a": f'Como el dado es aleatorio, no hay estrategia de movimiento directa. Sin embargo, el modelo de Markov revela que las posiciones entre <span class="hl">60 y 95</span> (evitando serpientes en 62, 64, 87, 93, 95, 99) son las más favorables. En variantes con cartas de decisión, se puede usar el vector E[i] para evaluar cuándo vale la pena usar un comodín.'
        },
        {
            "q": "4. ¿Cómo cambia la probabilidad de victoria en función de la posición inicial?",
            "a": f'La probabilidad crece de forma <span class="hl">no lineal</span> con la posición. Desde casilla 0: <span class="hl">{p[0]*100:.1f}%</span>. Casilla 50: <span class="hl">{p[50]*100:.1f}%</span>. Casilla 80: <span class="hl">{p[80]*100:.1f}%</span>. Las caídas drásticas se observan en serpientes de alta posición (87, 93, 95, 99). La curva NO es monótona: las escaleras crean picos y las serpientes crean valles abruptos.'
        },
        {
            "q": "5. ¿Es posible diseñar una estrategia basada en las probabilidades de las casillas?",
            "a": f'Sí, usando la cadena de Markov. El vector de probabilidades de absorción calcula el valor exacto de cada estado. El número esperado de turnos desde el inicio es <span class="hl">{EXP_TURNS[0]:.1f}</span>. La simulación Monte Carlo lo valida con errores típicos por debajo del <span class="hl">2%</span> al usar 1000+ partidas. Se pueden identificar cuellos de botella (serpiente 99→78) y posiciones objetivo (escalera 28→84) para optimizar decisiones en variantes del juego con elementos estratégicos.'
        },
    ]

    for ans in answers:
        st.markdown(f"""
        <div class="answer-box">
          <h3>{ans['q']}</h3>
          <p>{ans['a']}</p>
        </div>
        """, unsafe_allow_html=True)

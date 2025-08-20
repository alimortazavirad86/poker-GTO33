import streamlit as st
import math
import plotly.graph_objects as go

# ================== Poker GTO Helper — Unified (single render) ==================
st.set_page_config(page_title="Poker GTO Helper", page_icon="♠️", layout="wide")
st.title("♠️ Poker GTO Decision Helper — Unified")
st.caption("Plotly oval table • single render per rerun • Strict GTO vs Exploit + ICM toggle")

# ---------------- Sidebar: mode & table size ----------------
mode = st.sidebar.radio("Mode", ["Strict GTO", "Exploit + ICM"], index=0)
strict_mode = (mode == "Strict GTO")
seat_count = st.sidebar.select_slider("Seats", options=[6,8,9], value=9)
st.sidebar.caption("Tip: 9-handed is the default.")

# ---------------- Example inputs (replace with your UI if needed) ----------------
# You can wire these to controls exactly like your larger app does.
hero_seat = st.number_input("Hero seat index", min_value=0, max_value=seat_count-1, value=0, step=1)
btn_seat  = st.number_input("Button seat index", min_value=0, max_value=seat_count-1, value=seat_count-1, step=1)

# Hero cards (compact selectors to keep this file short)
colh1, colh2 = st.columns(2)
with colh1:
    r1 = st.selectbox("Hero Card 1 Rank", list("23456789TJQKA"), index=12)
    s1 = st.selectbox("Hero Card 1 Suit", ["c","d","h","s"], index=0)
with colh2:
    r2 = st.selectbox("Hero Card 2 Rank", list("23456789TJQKA"), index=11)
    s2 = st.selectbox("Hero Card 2 Suit", ["c","d","h","s"], index=2)

hero_hand = f"{r1}{s1}{r2}{s2}".upper()

# Minimal board (optional)
bf1, bf2, bf3 = st.columns(3)
with bf1:
    flop1_r = st.selectbox("Flop 1 Rank", ["—"]+list("23456789TJQKA"), index=0, key="f1r")
    flop1_s = st.selectbox("Flop 1 Suit", ["—","c","d","h","s"], index=0, key="f1s")
with bf2:
    flop2_r = st.selectbox("Flop 2 Rank", ["—"]+list("23456789TJQKA"), index=0, key="f2r")
    flop2_s = st.selectbox("Flop 2 Suit", ["—","c","d","h","s"], index=0, key="f2s")
with bf3:
    flop3_r = st.selectbox("Flop 3 Rank", ["—"]+list("23456789TJQKA"), index=0, key="f3r")
    flop3_s = st.selectbox("Flop 3 Suit", ["—","c","d","h","s"], index=0, key="f3s")

def _card_if(r, s):
    if r == "—" or s == "—":
        return ""
    return f"{r}{s}".upper()

board_cards = "".join([
    _card_if(flop1_r, flop1_s),
    _card_if(flop2_r, flop2_s),
    _card_if(flop3_r, flop3_s),
])

# ---------------- Simple (placeholder) decision logic ----------------
# Replace this with your real gto_preflop_mix / gto_postflop_mix.
def dummy_gto(hand: str, strict: bool = True):
    # A very tiny stub so the page runs; plug your engine here.
    base = {"RAISE":0.50, "CALL":0.30, "FOLD":0.20} if strict else {"RAISE":0.45, "CALL":0.35, "FOLD":0.20}
    if hand[:2] in ("AA","KK","QQ","AK","AQ"):
        base = {"RAISE":0.85, "CALL":0.10, "FOLD":0.05}
    best = max(base, key=base.get)
    return {"best": best, "best_pct": base[best], "all": base}

result = dummy_gto(hero_hand, strict_mode)

# ---------------- Decision box ----------------
st.subheader("Decision — Preflop")
if result:
    pct = int(round(result.get('best_pct',0.0)*100))
    st.success(f"{result.get('best','')} — {pct}%")
    allp = result.get("all", {})
    if allp:
        st.markdown("**Action mix**")
        for lbl, p in sorted(allp.items(), key=lambda kv: kv[1], reverse=True):
            lab = "BET/RAISE" if lbl == "RAISE" else lbl
            st.write(f"• {lab}: **{int(round(p*100))}%**")

# ---------------- Single-render Plotly table ----------------
SUIT_SYM = {"s":"♠","h":"♥","d":"♦","c":"♣"}
SUIT_COLOR = {"s":"#1f2a44","c":"#1f2a44","h":"#c62828","d":"#c62828"}

def draw_table(seat_count:int, btn_seat:int, hero_seat:int,
               total_pot:float, board_codes, hero_codes,
               width:int=900, height:int=540):
    fig = go.Figure()
    fig.update_layout(
        width=width, height=height,
        plot_bgcolor="#0b1225", paper_bgcolor="#0b1225",
        xaxis=dict(visible=False, range=[0,100]),
        yaxis=dict(visible=False, range=[0,100]),
        margin=dict(l=20, r=20, t=20, b=20)
    )

    # Table oval
    fig.add_shape(
        type="circle", xref="x", yref="y",
        x0=10, y0=10, x1=90, y1=90,
        line=dict(color="#1f2937", width=6),
        fillcolor="#0f172a"
    )

    # Pot
    fig.add_annotation(
        x=50, y=48, text=f"<b>POT: {total_pot:.2f} BB</b>",
        showarrow=False, font=dict(color="#ffd166", size=18)
    )

    # Board cards (top-center)
    if board_codes:
        start_x = 50 - 6*len(board_codes)
        for i, c in enumerate(board_codes):
            if len(c) != 2:  # guard
                continue
            r, s = c[0], c[1].lower()
            fig.add_shape(
                type="rect", x0=start_x+12*i-4, y0=58, x1=start_x+12*i+4, y1=72,
                line=dict(color="#cbd5e1"), fillcolor="#f8fafc"
            )
            fig.add_annotation(
                x=start_x+12*i, y=65, text=f"<b>{r}{SUIT_SYM.get(s,'')}</b>",
                showarrow=False, font=dict(color=SUIT_COLOR.get(s,"#e5e7eb"), size=16)
            )

    # Seat badges around ellipse
    cx, cy = 50, 50
    rx, ry = 34, 30
    for i in range(seat_count):
        angle = -math.pi/2 + (2*math.pi * i / seat_count)
        x = cx + rx * math.cos(angle)
        y = cy + ry * math.sin(angle)
        is_btn = (i == btn_seat)
        is_hero = (i == hero_seat)
        badge = []
        if is_btn:  badge.append("BTN")
        if is_hero: badge.append("YOU")
        label = f"Seat {i}" + (f" ({', '.join(badge)})" if badge else "")

        # seat box
        fig.add_shape(
            type="rect", x0=x-7, y0=y-4, x1=x+7, y1=y+4,
            line=dict(color="#374151"), fillcolor="#111827", opacity=1.0
        )
        fig.add_annotation(
            x=x, y=y, text=f"<b>{label}</b>",
            showarrow=False, font=dict(color="#e5e7eb", size=11)
        )

    # Hero hand displayed outside table near hero seat
    if len(hero_codes) == 2:
        i = hero_seat
        angle = -math.pi/2 + (2*math.pi * i / seat_count)
        hx = cx + (rx+7) * math.cos(angle)
        hy = cy + (ry+10) * math.sin(angle)
        for k, c in enumerate(hero_codes):
            if len(c) != 2:
                continue
            r, s = c[0], c[1].lower()
            fig.add_shape(
                type="rect", x0=hx-8+9*k, y0=hy-6, x1=hx-1+9*k, y1=hy+6,
                line=dict(color="#cbd5e1"), fillcolor="#f8fafc"
            )
            fig.add_annotation(
                x=hx-4+9*k, y=hy, text=f"<b>{r}{SUIT_SYM.get(s,'')}</b>",
                showarrow=False, font=dict(color=SUIT_COLOR.get(s,"#e5e7eb"), size=16)
            )

    return fig

# Prepare codes
board_codes = [board_cards[i:i+2] for i in range(0, len(board_cards), 2) if board_cards]
hero_codes = [hero_hand[0:2], hero_hand[2:4]] if len(hero_hand) == 4 else []
total_pot = 1.5  # SB + BB by default. Replace with your simulation’s pot if needed.

st.subheader("Live Table — Preflop")
# SINGLE RENDER: use a placeholder and render the figure once
table_placeholder = st.empty()
table_fig = draw_table(
    seat_count=seat_count,
    btn_seat=btn_seat,
    hero_seat=hero_seat,
    total_pot=total_pot,
    board_codes=board_codes,
    hero_codes=hero_codes,
    width=900, height=540
)
table_placeholder.plotly_chart(table_fig, use_container_width=False)

st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("Strict GTO disables exploit & ICM; Exploit + ICM enables them. Chart is rendered exactly once per rerun.")

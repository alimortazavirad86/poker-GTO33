import streamlit as st
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Tuple, List
import math
from collections import Counter
import plotly.graph_objects as go

# ================== Poker GTO Helper — Build v9e (Plotly table + full action mix) ==================
st.set_page_config(page_title="Poker GTO Helper", page_icon="♠️", layout="wide")
st.title("♠️ Poker GTO Decision Helper")
st.caption("Build: v9e • Plotly-rendered table • Auto-call allocation • Decision box shows all action percentages • Decision-point toggle (handles 3-bets behind you)")

# -------- Core cards --------
Position = Literal["UTG", "MP", "CO", "BTN", "SB", "BB"]
RANK_TO_VAL = {"A":14,"K":13,"Q":12,"J":11,"T":10,"9":9,"8":8,"7":7,"6":6,"5":5,"4":4,"3":3,"2":2}
VAL_TO_RANK = {v:k for k,v in RANK_TO_VAL.items()}
RANK_ORDER = "23456789TJQKA"
RANK_TO_INT = {r:i for i,r in enumerate(RANK_ORDER, start=2)}
SUITS = ["c","d","h","s"]
SUIT_SYM = {"s":"♠","h":"♥","d":"♦","c":"♣"}
SUIT_COLOR = {"s":"#1f2a44","c":"#1f2a44","h":"#c62828","d":"#c62828"}
PERSONALITIES = ["unknown","amateur-conservative","amateur-loose","professional-conservative","professional-aggressive"]

def _clamp(x: float, a: float=0.0, b: float=1.0) -> float: return max(a, min(b, x))
def _sigmoid(x: float) -> float: return 1/(1+math.exp(-x))
def _pct_round(x: float) -> int: return int(round(_clamp(x,0,1)*100/5)*5)

def _normalize_probs(d: Dict[str, float]) -> Dict[str, float]:
    tot = sum(max(0.0, v) for v in d.values())
    if tot <= 1e-12:
        return {k: 0.0 for k in d}
    return {k: max(0.0, v)/tot for k, v in d.items()}

def normalize_combo(hand: str) -> Tuple[str,str,str,bool]:
    h = hand.strip().upper()
    if len(h)!=4: raise ValueError("Hand must be like 'AhKh'")
    r1,s1,r2,s2 = h[0],h[1],h[2],h[3]
    suited = (s1==s2)
    v1,v2 = RANK_TO_VAL[r1], RANK_TO_VAL[r2]
    if v1==v2: return r1+r2, r1, r2, suited
    hi_v, lo_v = max(v1,v2), min(v1,v2)
    hi, lo = VAL_TO_RANK[hi_v], VAL_TO_RANK[lo_v]
    return f"{hi}{lo}{'s' if suited else 'o'}", hi, lo, suited

def gap_size(hi: str, lo: str) -> int: return max(0, RANK_TO_VAL[hi] - RANK_TO_VAL[lo] - 1)

def chen_score(hi: str, lo: str, suited: bool) -> float:
    base_map = {"A":10,"K":8,"Q":7,"J":6,"T":5,"9":4.5,"8":4,"7":3.5,"6":3,"5":2.5,"4":2,"3":1.5,"2":1}
    base = base_map[hi]
    if hi==lo: return max(5.0, base*2)
    score = base + (2 if suited else 0)
    g = gap_size(hi,lo)
    if g==1: score -= 1
    elif g==2: score -= 2
    elif g==3: score -= 4
    elif g>=4: score -= 5
    if RANK_TO_VAL[hi]<=12 and g==0: score += 1
    return max(0.0, score)

def parse_card(card: str) -> Tuple[str,str]:
    c = card.strip().upper()
    if len(c)!=2 or c[0] not in RANK_TO_INT or c[1] not in "CDHS": raise ValueError(f"Bad card: {card}")
    return c[0], c[1]

def parse_cards(cards: str) -> list:
    s = (cards or "").strip().upper()
    if len(s)%2!=0: raise ValueError("Cards string must be even length")
    arr = [s[i:i+2] for i in range(0,len(s),2)]
    for c in arr: parse_card(c)
    return arr

def board_texture(board: list) -> Dict[str,float]:
    suits = [parse_card(c)[1] for c in board]
    ranks = [RANK_TO_INT[parse_card(c)[0]] for c in board]
    suit_counts = Counter(suits)
    paired = len(board)!=len(set(ranks))
    flushy = max(suit_counts.values()) if suit_counts else 0
    gaps = sorted(set(ranks)); straightiness = 0
    if len(gaps)>=3:
        for a in range(min(gaps), max(gaps)-1):
            straightiness = max(straightiness, sum((a+i) in gaps for i in range(3)))
    wet = 0 + (0.5 if flushy>=3 else 0) + (0.5 if straightiness>=3 else 0) + (0.25 if paired else 0)
    return {"paired": float(paired), "flushy": float(flushy), "straightiness": float(straightiness), "wetness": float(wet)}

def made_hand_strength(hand: list, board: list) -> Tuple[str,int]:
    ranks=[]; suits=[]
    for c in hand+board:
        r,s = parse_card(c); ranks.append(RANK_TO_INT[r]); suits.append(s)
    suit_counts = Counter(suits); flush = max(suit_counts.values())>=5 if suit_counts else False
    uniq = sorted(set(ranks))
    if {14,2,3,4,5}.issubset(set(uniq)): straight=True
    else: straight = any(all((x+i) in uniq for i in range(5)) for x in range(min(uniq), max(uniq)-3)) if uniq else False
    cnt = Counter(ranks); counts = sorted(cnt.values(), reverse=True) if cnt else []
    if 4 in counts: return ("quads",8)
    if 3 in counts and 2 in counts: return ("full_house",7)
    if flush and straight: return ("straight_flush-ish",7)
    if flush: return ("flush",6)
    if straight: return ("straight",5)
    if 3 in counts: return ("trips",4)
    if counts.count(2)>=2: return ("two_pair",3)
    if 2 in counts: return ("pair",2)
    return ("high_card",1)

def top_pair_overpair_underpair(hand: list, board: list) -> str:
    if len(hand)<2 or not board: return "no_pair"
    hr1,_ = parse_card(hand[0]); hr2,_ = parse_card(hand[1])
    branks = sorted([RANK_TO_INT[parse_card(c)[0]] for c in board], reverse=True)
    r1 = RANK_TO_INT[hr1]; r2 = RANK_TO_INT[hr2]
    if r1==r2:
        if not branks: return "no_pair"
        return "overpair" if r1>branks[0] else ("top_pair" if r1==branks[0] else "underpair")
    top = branks[0] if branks else 0
    if r1==top or r2==top: return "top_pair"
    if r1 in branks or r2 in branks: return "mid_pair"
    return "no_pair"

def draw_outs(hand: list, board: list) -> Dict[str,int]:
    suits = [parse_card(c)[1] for c in hand+board]; suit_counts = Counter(suits)
    flush_draw = (max(suit_counts.values())==4) if suit_counts else False
    ranks = sorted(set([RANK_TO_INT[parse_card(c)[0]] for c in hand+board]))
    open_ender=0; gutshot=0
    if ranks:
        for a in range(min(ranks), max(ranks)-2):
            window=[a+i for i in range(4)]; got=sum(1 for v in window if v in ranks)
            if got==4: open_ender=1
            elif got==3: gutshot=1
    outs = 0 + (9 if flush_draw else 0) + (8 if open_ender else (4 if gutshot else 0))
    return {"flush_draw":int(flush_draw),"open_ender":int(open_ender),"gutshot":int(gutshot),"outs":outs}

def equity_from_outs(outs:int, cards_to_come:int) -> float:
    return min(0.95, (outs * (4 if cards_to_come==2 else 2)) / 100)

_POS_TIGHTNESS = {"UTG":1.00,"MP":0.92,"CO":0.84,"BTN":0.75,"SB":0.80,"BB":0.90}
def bubble_factor(players_left:int, total_players:int) -> float:
    if total_players<=0 or players_left<=0: return 1.0
    pct = players_left/max(1,total_players)
    if pct<=0.05: return 1.5
    if pct<=0.15: return 1.3
    return 1.0

# ---------- FIXED: Unopened-pot open frequency (pocket-pair boost + late-position floors) ----------
def _preflop_open_freq(hi:str, lo:str, suited:bool, position:Position, stack_bb:float, bubble_mult:float) -> float:
    """
    Baseline open frequency in unopened pots.
    Tweaks:
      • Pocket-pair boost (esp. 66–99 and TT+)
      • Late-position bonus (CO/BTN/SB)
      • Floors so pairs (like 88) don't get folded too often from CO/BTN/SB
    """
    s = chen_score(hi,lo,suited)

    # Pocket-pair boost
    pair = (hi == lo)
    if pair:
        rv = RANK_TO_VAL[hi]
        if rv >= 10:      # TT+
            s += 2.0
        elif rv >= 8:     # 88–99
            s += 1.5
        elif rv >= 6:     # 66–77
            s += 1.0
        else:             # 22–55
            s += 0.5

    # Late-position bonus
    s += {"CO": 1.0, "BTN": 1.6, "SB": 0.6}.get(position, 0.0)

    # Threshold by position/stack/bubble
    base_t = 12.0 * _POS_TIGHTNESS.get(position,1.0)
    if stack_bb<=12: base_t += 1.0
    if stack_bb<=8:  base_t += 1.0
    base_t *= bubble_mult

    f = _clamp(_sigmoid((s - base_t)/2.2))

    # Floors for pairs in late positions
    if pair:
        rv = RANK_TO_VAL[hi]
        if position == "BTN":
            f = max(f, 0.55)            # any pair at least 55% open on BTN
            if rv >= 8:                 # 88+
                f = max(f, 0.70)
        elif position == "CO":
            f = max(f, 0.42)
            if rv >= 8:
                f = max(f, 0.58)
        elif position == "SB":
            f = max(f, 0.38)
    return f
# -----------------------------------------------------------------------------------------------

def _preflop_three_bet_split(hi:str, lo:str, suited:bool, position:Position, vs_open_size_bb:float, stack_bb:float) -> Tuple[float,float,float]:
    s = chen_score(hi,lo,suited)
    t_val = 14.0 - (stack_bb-20)/40
    t_blf_lo, t_blf_hi = 10.0, 12.5
    val = _clamp((s - t_val)/4.0); blf = 0.0
    if position in ("CO","BTN","SB"):
        if suited and (hi in "AKQJ" or (hi=="A" and RANK_TO_VAL[lo]<=6)):
            blf = _clamp((s - t_blf_lo)/(t_blf_hi - t_blf_lo)) * 0.6
        else:
            blf = _clamp((s - t_blf_lo)/(t_blf_hi - t_blf_lo)) * 0.3
    cont = _clamp(val+blf); call = _clamp(1.0 - cont)
    total = val+blf+call
    return (val/total, blf/total, call/total) if total>0 else (0.0,0.0,0.0)

def _hand_strength_score(made_tier:int, outs_eq:float, tou:Optional[str]) -> float:
    base = {1:0.15,2:0.45,3:0.62,4:0.70,5:0.78,6:0.85,7:0.92,8:0.98}.get(made_tier,0.3)
    if tou in ("overpair","top_pair"): base += 0.05
    if tou=="underpair": base -= 0.05
    return _clamp(max(base, outs_eq))
def _range_advantage(prev_aggressor:str, texture:Dict[str,float]) -> float:
    adv = 0.0
    if prev_aggressor=="hero": adv += 0.25
    adv += (0.6 - texture.get("wetness",0.4))
    adv -= 0.1 * texture.get("paired",0.0)
    return max(-0.3, min(0.5, adv))

@dataclass
class PostflopContext:
    street: Literal["flop","turn","river"]
    hero_hand: str
    board_cards: str
    pot_bb: float
    eff_stack_bb: float
    facing_bet_bb: float = 0.0
    position: Literal["IP","OOP"] = "IP"
    n_players: int = 2
    prev_aggressor: Literal["hero","villain","none"] = "none"
    exploit_adv: float = 0.0

# ================== Strategy (now returns full action mix) ==================
def gto_preflop_mix(hand:str, position:Position, stack_bb:float, players_left:int, total_players:int,
                    facing_open:bool=False, open_size_bb:float=2.5, facing_shove:bool=False, shove_size_bb:float=0.0,
                    pot_before_hero_bb:float=0.0, exploit_adj:float=0.0) -> Dict[str,str]:
    combo, hi, lo, suited = normalize_combo(hand)
    bub = bubble_factor(players_left, total_players)

    # Facing all-in
    if facing_shove and shove_size_bb>0:
        pot_before = pot_before_hero_bb if pot_before_hero_bb>0 else 1.5
        req = shove_size_bb / max(1e-9, (pot_before + shove_size_bb)) * bub
        s = chen_score(hi,lo,suited)
        diff = (s/20.0) - req + exploit_adj*0.05
        p_call = _clamp(0.5 + diff*2.0)
        allp = _normalize_probs({"CALL": p_call, "RAISE": 0.0, "CHECK": 0.0, "FOLD": 1.0 - p_call})
        best, best_p = max(allp.items(), key=lambda kv: kv[1])
        return {
            "primary": best, "best": best, "best_pct": best_p,
            "mix": f"CALL {_pct_round(p_call)}% / FOLD {_pct_round(1-p_call)}%",
            "all": allp,
            "explain": f"Pot-odds ~{req*100:.1f}%; Chen≈{s:.1f}. Exploit {exploit_adj:+.2f}"
        }

    # Facing an open
    if facing_open:
        val, blf, call = _preflop_three_bet_split(hi,lo,suited,position,open_size_bb,stack_bb)
        # Exploit adjustments
        val = _clamp(val + max(0, exploit_adj)*0.10)
        blf = _clamp(blf + max(0, exploit_adj)*0.05)
        call = _clamp(call + (-min(0, exploit_adj))*0.07)
        jam_share = _clamp((20 - stack_bb)/15.0) if stack_bb<20 else 0.0

        parts = []
        def add(lbl,p):
            if p>0.02: parts.append((lbl,p))
        add("3B-VALUE", val*(1-jam_share)); add("JAM-VALUE", val*jam_share)
        add("3B-BLUFF", blf*(1-jam_share)); add("JAM-BLUFF", blf*jam_share)
        add("CALL", call)
        fold = _clamp(1.0 - sum(p for _,p in parts)); add("FOLD", fold)

        # Aggregate to CALL / RAISE / CHECK / FOLD
        agg = {"CALL":0.0,"RAISE":0.0,"CHECK":0.0,"FOLD":0.0}
        for lbl,p in parts:
            if lbl=="CALL": agg["CALL"] += p
            elif lbl=="FOLD": agg["FOLD"] += p
            else: agg["RAISE"] += p
        allp = _normalize_probs(agg)

        if parts:
            primary, primary_p = max(parts, key=lambda x:x[1])
            mix = " / ".join([f"{l} {_pct_round(p)}%" for l,p in parts])
        else:
            primary, primary_p, mix = "FOLD", 1.0, "FOLD 100%"

        # Choose best among aggregated buckets for display as well
        best_bucket, best_bucket_p = max(allp.items(), key=lambda kv: kv[1])
        return {
            "primary": primary, "best": best_bucket, "best_pct": best_bucket_p,
            "mix": mix, "all": allp,
            "explain": f"Stack {stack_bb:.0f}bb; exploit {exploit_adj:+.2f}"
        }

    # Unopened pot (hero first to act or folded to hero)
    f_open = _preflop_open_freq(hi,lo,suited,position,stack_bb,bub)
    f_open = _clamp(f_open + exploit_adj*0.06)
    p_open = f_open; p_fold = 1 - p_open
    small = _clamp(0.8 - (20 - stack_bb)/50.0) if stack_bb<40 else 0.85
    large = _clamp(1.0 - small)
    parts=[]
    if p_open>0.02:
        parts.append((f"OPEN {int(2.0 if position not in ('CO','BTN','SB') else 2.2)}bb", p_open*small))
        parts.append((("JAM" if stack_bb<=14 else "OPEN 2.8bb"), p_open*large))
    if p_fold>0.02: parts.append(("FOLD", p_fold))

    if parts:
        primary, primary_p = max(parts, key=lambda x:x[1]); mix = " / ".join([f"{l} {_pct_round(p)}%" for l,p in parts])
    else:
        primary, primary_p, mix = "FOLD", 1.0, "FOLD 100%"

    # Aggregate to buckets
    agg = {"CALL":0.0,"RAISE":0.0,"CHECK":0.0,"FOLD":0.0}
    for lbl,p in parts:
        if lbl.startswith("OPEN") or "JAM" in lbl: agg["RAISE"] += p
        elif lbl=="FOLD": agg["FOLD"] += p
    allp = _normalize_probs(agg)

    best_bucket, best_bucket_p = max(allp.items(), key=lambda kv: kv[1])
    return {"primary":primary,"best":best_bucket,"best_pct":best_bucket_p,"mix":mix,"all":allp,
            "explain":f"Chen; bubble x{bub:.2f}; exploit {exploit_adj:+.2f}"}

def gto_postflop_mix(ctx: 'PostflopContext') -> Dict[str,str]:
    hand = parse_cards(ctx.hero_hand); board = parse_cards(ctx.board_cards)
    made, tier = made_hand_strength(hand, board); texture = board_texture(board)
    outs_info = draw_outs(hand, board); cards_to_come = 2 if ctx.street=='flop' else (1 if ctx.street=='turn' else 0)
    draw_eq = equity_from_outs(outs_info['outs'], cards_to_come) if cards_to_come else 0.0
    tou = top_pair_overpair_underpair(hand, board)
    hs = _hand_strength_score(tier, draw_eq, tou)
    spr = max(0.01, ctx.eff_stack_bb / max(0.01, ctx.pot_bb))
    adv = _range_advantage(ctx.prev_aggressor, texture) + ctx.exploit_adv*0.10
    adv -= 0.1 * max(0, ctx.n_players - 2)  # multiway penalty

    sizes = []
    if ctx.street=='flop': sizes = [(0.33, 0.5+adv), (0.66, 0.3-adv/2)]
    elif ctx.street=='turn': sizes = [(0.5, 0.45+adv/2), (0.66, 0.35-adv/4), (1.0, 0.2)]
    else: sizes = [(0.66, 0.4), (1.0, 0.35+adv/3)]
    total_prior = sum(max(0.0,p) for _,p in sizes); sizes = [(b, max(0.0,p)/max(1e-9,total_prior)) for b,p in sizes]

    # Facing a bet
    if ctx.facing_bet_bb>0:
        bet = ctx.facing_bet_bb; pot = ctx.pot_bb
        mdf = _clamp(pot/(pot+bet)) * (1.0 - max(0, ctx.n_players-2)*0.15)
        mdf = _clamp(mdf + (-ctx.exploit_adv)*0.05)
        p_continue = _clamp((hs - 0.35)/0.45) * (1.0 - 0.5*max(0, ctx.n_players-2)*0.15)
        p_raise = _clamp((hs-0.70)*2.5) if spr>2 else 0.0
        p_raise *= (1.0 - max(0, ctx.n_players-2)*0.15)
        p_raise = _clamp(p_raise + ctx.exploit_adv*0.08)
        p_call = _clamp(p_continue - p_raise)
        if p_continue < mdf:
            p_call = _clamp(p_call + (mdf - p_continue))
        p_fold = _clamp(1.0 - _clamp(p_call + p_raise))

        allp = _normalize_probs({"CALL": p_call, "RAISE": p_raise, "CHECK": 0.0, "FOLD": p_fold})
        best, best_p = max(allp.items(), key=lambda kv: kv[1])

        return {"primary":best,"best":best,"best_pct":best_p,
                "mix":f"CALL {_pct_round(p_call)}% / RAISE {_pct_round(p_raise)}% / FOLD {_pct_round(p_fold)}%",
                "all": allp,
                "explain":f"{ctx.street.upper()} vs {bet:.1f} into {pot:.1f} | N={ctx.n_players} | hs~{hs*100:.0f}% | adv~{adv:+.2f} | SPR~{spr:.1f} | exploit {ctx.exploit_adv:+.2f}"}

    # No bet faced: choose bet sizes vs check
    p_bet_total = _clamp(0.15 + adv + (hs-0.45)) * (1.0 - 0.6*max(0, ctx.n_players-2)*0.15)
    p_check = _clamp(1.0 - p_bet_total)
    bet_parts=[]; weight_sum=0.0
    for b,p in sizes:
        w = p * (0.8 + 0.6*(hs-0.5)) * (1.0 + (0.3 if b>=0.66 else 0.0))
        w *= (1.0 - 0.5*max(0, ctx.n_players-2)*0.15)
        w *= (1.0 + ctx.exploit_adv*0.10)
        bet_parts.append((b,w)); weight_sum += w
    mix_parts=[]; p_primary=p_check; primary="CHECK"; primary_p=p_check
    if weight_sum>0 and p_bet_total>0.02:
        for b,w in bet_parts:
            frac=(w/weight_sum)*p_bet_total
            if frac>p_primary: p_primary=frac; primary=f"BET {int(b*100)}%"; primary_p=frac
            mix_parts.append((f"BET {int(b*100)}%", frac))
    mix_parts.append(("CHECK", p_check))
    mix_str = " / ".join([f"{lbl} {_pct_round(p)}%" for lbl,p in mix_parts])

    # Aggregate to buckets for display
    agg = {"CALL":0.0,"RAISE":0.0,"CHECK":p_check,"FOLD":0.0}
    for lbl,p in mix_parts:
        if lbl.startswith("BET"): agg["RAISE"] += p
    allp = _normalize_probs(agg)
    best_bucket, best_bucket_p = max(allp.items(), key=lambda kv: kv[1])

    return {"primary":primary,"best":best_bucket,"best_pct":best_bucket_p,"mix":mix_str,"all":allp,
            "explain":f"{ctx.street.upper()} | N={ctx.n_players} | hs~{hs*100:.0f}% | adv~{adv:+.2f} | SPR~{spr:.1f} | exploit {ctx.exploit_adv:+.2f}"}

def poker_decision_gto(*args, **kwargs): return gto_preflop_mix(*args, **kwargs)
def postflop_decision_gto(*args, **kwargs):
    ctx = PostflopContext(*args, **kwargs); return gto_postflop_mix(ctx)

# =============== Plotly Table Renderer ===============
def plotly_table(seat_count:int, btn_seat:int, hero_seat:int, pos_by_seat:Dict[int,str],
                 active_seats:List[int], contribs:Dict[int,float], total_pot:float,
                 board_codes:List[str], hero_codes:List[str], width:int=900, height:int=520):
    # Base figure
    fig = go.Figure()
    fig.update_layout(width=width, height=height, plot_bgcolor="#0b1225", paper_bgcolor="#0b1225",
                      xaxis=dict(visible=False, range=[0,100]), yaxis=dict(visible=False, range=[0,100]),
                      margin=dict(l=20,r=20,t=20,b=20))
    # Table oval
    oval = dict(type="circle", xref="x", yref="y",
                x0=10, y0=10, x1=90, y1=90,
                line=dict(color="#1f2937", width=6),
                fillcolor="#0f172a")
    fig.add_shape(oval)
    # Pot label
    fig.add_annotation(x=50, y=48, text=f"<b>POT: {total_pot:.2f} BB</b>", showarrow=False, font=dict(color="#ffd166", size=18))
    # Board cards
    if board_codes:
        start_x = 50 - 6*len(board_codes)
        for i,c in enumerate(board_codes):
            fig.add_shape(type="rect", x0=start_x+12*i-4, y0=58, x1=start_x+12*i+4, y1=72, line=dict(color="#cbd5e1"), fillcolor="#f8fafc")
            r,s = c[0], c[1].lower(); fig.add_annotation(x=start_x+12*i, y=65, text=f"<b>{r}{SUIT_SYM[s]}</b>", showarrow=False, font=dict(color=SUIT_COLOR[s], size=16))
    # seat positions ellipse
    import math as _m
    cx, cy = 50, 50; rx, ry = 34, 30
    for i in range(seat_count):
        angle = -_m.pi/2 + (2*_m.pi * i / seat_count)
        x = cx + rx * _m.cos(angle); y = cy + ry * _m.sin(angle)
        role = pos_by_seat.get(i,""); is_btn = (i==btn_seat); is_active = (i in active_seats)
        badge = "BTN" if is_btn else role
        opa = 1.0 if is_active else 0.35
        fig.add_shape(type="rect", x0=x-7, y0=y-4, x1=x+7, y1=y+4, line=dict(color="#374151"), fillcolor="#111827", opacity=opa)
        fig.add_annotation(x=x, y=y, text=f"<b>Seat {i}</b><br><span style='color:#9ca3af'>{badge}</span>", showarrow=False, font=dict(color="#e5e7eb", size=11), opacity=opa)
        put = float(contribs.get(i,0.0))
        if put>0:
            fig.add_annotation(x=x, y=y-6.5, text=f"<b>{put:.2f}</b>", showarrow=False, font=dict(color="#ffffff", size=11), bgcolor="#ef4444")
    # Hero hand outside ellipse
    if len(hero_codes)==2:
        i = hero_seat
        angle = -_m.pi/2 + (2*_m.pi * i / seat_count)
        hx = cx + (rx+7) * _m.cos(angle); hy = cy + (ry+10) * _m.sin(angle)
        for k,c in enumerate(hero_codes):
            fig.add_shape(type="rect", x0=hx-8+9*k, y0=hy-6, x1=hx-1+9*k, y1=hy+6, line=dict(color="#cbd5e1"), fillcolor="#f8fafc")
            r,s = c[0], c[1].lower(); fig.add_annotation(x=hx-4+9*k, y=hy, text=f"<b>{r}{SUIT_SYM[s]}</b>", showarrow=False, font=dict(color=SUIT_COLOR[s], size=16))
    st.plotly_chart(fig, use_container_width=False)

# ================== Session State ==================
if "step" not in st.session_state: st.session_state.step = 0
if "actions" not in st.session_state: st.session_state.actions = {0:{},1:{},2:{},3:{}}
step = st.session_state.step
street_map = {0:"preflop",1:"flop",2:"turn",3:"river"}
street = street_map[step]; street_title = street.capitalize()

# ================== Sidebar ==================
SEATS = st.sidebar.select_slider("Seats", options=[6,8,9], value=9)

with st.sidebar:
    st.subheader("Tournament Context")
    stack_bb = st.number_input("Effective Stack (BB)", min_value=1, value=25, key="stack_bb")
    players_left = st.number_input("Players Left", min_value=2, value=120, key="players_left")
    total_players = st.number_input("Total Players", min_value=2, value=1000, key="total_players")
    st.markdown("---"); st.subheader("Street Progress")
    st.progress((step+1)/4); st.caption("1) Preflop → 2) Flop → 3) Turn → 4) River")
    if st.button("Reset current street actions"):
        st.session_state.actions[step] = {}

# ================== Table Setup & Players ==================
st.subheader("Table Setup & Player Selection")
cols = st.columns(SEATS)
seat_active=[]; seat_persona=[]
hero_seat = st.radio("Who are you? (choose your seat)", options=list(range(SEATS)), index=0, horizontal=True, key="hero_seat", format_func=lambda i: f"Seat {i}")
btn_disabled = step>0
btn_seat = st.radio("Dealer Button Seat (set on Step 1: Preflop)", options=list(range(SEATS)), index=SEATS-1, horizontal=True, key="btn_seat", disabled=btn_disabled, format_func=lambda i: f"Seat {i}")
if btn_disabled: st.caption("Button selection is locked after Preflop. Go Back to Step 1 to change it.")
for i in range(SEATS):
    with cols[i]:
        st.markdown(f"**Seat {i}{' (YOU)' if i==hero_seat else ''}**")
        active = st.checkbox("Active in pot", value=(i==hero_seat or i in ((btn_seat+1)%SEATS, (btn_seat+2)%SEATS)), key=f"active_{i}")
        default_other = PERSONALITIES.index("professional-conservative")
        default_hero = PERSONALITIES.index("professional-aggressive")
        persona = st.selectbox("Personality", PERSONALITIES, index=(default_hero if i==hero_seat else default_other), key=f"p_{i}")
        seat_active.append(active); seat_persona.append(persona)

# ================== Cards Picker ==================
st.subheader("Cards Picker")
cp1, cp2, cp3 = st.columns([1,1,2])
def _pick(prefix:str) -> Optional[str]:
    r = st.selectbox(f"{prefix} Rank", list(RANK_ORDER), key=f"{prefix}_r")
    s = st.selectbox(f"{prefix} Suit", SUITS, key=f"{prefix}_s")
    return f"{r}{s}"
with cp1:
    st.markdown("**Your Hand**")
    h1 = _pick("Hero Card 1"); h2 = _pick("Hero Card 2"); hero_hand = (h1+h2).upper()
with cp2:
    st.markdown("**Board (by street)**")
    b1 = _pick("Flop 1") if step>=1 else ""
    b2 = _pick("Flop 2") if step>=1 else ""
    b3 = _pick("Flop 3") if step>=1 else ""
    b4 = _pick("Turn") if step>=2 else ""
    b5 = _pick("River") if step>=3 else ""
    board_cards = (b1+b2+b3+b4+b5).upper()
with cp3:
    st.write("Selected cards preview appears on the table.")

# ================== Positions from BTN ==================
labels_by_table = {9:["BTN","SB","BB","UTG","UTG+1","MP","LJ","HJ","CO"],
                   8:["BTN","SB","BB","UTG","UTG+1","MP","HJ","CO"],
                   6:["BTN","SB","BB","UTG","HJ","CO"]}
labels = labels_by_table.get(SEATS, labels_by_table[9])
pos_by_seat: Dict[int,str] = {}
seat_sequence = [(btn_seat + i) % SEATS for i in range(SEATS)]
for idx,s in enumerate(seat_sequence): pos_by_seat[s] = labels[idx] if idx < len(labels) else f"P{idx}"
active_indices = [i for i,a in enumerate(seat_active) if a]
active_order = [s for s in seat_sequence if s in active_indices]
hero_pos_label = pos_by_seat.get(hero_seat, "CO")
players_in_pot = max(2, len(active_indices))

# Personalities -> exploit
weights=[]
for i in range(SEATS):
    if not seat_active[i] or i==hero_seat: continue
    p = seat_persona[i]
    w = {"professional-aggressive":0.8,"amateur-loose":0.4,"professional-conservative":-0.4,"amateur-conservative":-0.2}.get(p,0.0)
    weights.append(w)
exploit_adv = float(max(-1.0, min(1.0, (sum(weights)/len(weights)) if weights else 0.0)))

# ================== Betting Order ==================
def betting_order(street_step:int, btn:int, seats:int, only_active=True) -> List[int]:
    if street_step==0: start = (btn + 3) % seats
    else: start = (btn + 1) % seats
    order = [(start + k) % seats for k in range(seats)]
    if only_active: order = [s for s in order if seat_active[s]]
    return order
order_now = betting_order(step, btn_seat, SEATS, only_active=True)

# ================== Decision point (first act vs. facing action behind) ==================
decision_point = st.radio(
    "Decision point",
    ["First action", "After action behind (return to hero)"],
    horizontal=True,
    key="decision_point"
)

# ================== Action Windows (per seat) ==================
st.subheader(f"Actions — {street_title}")
ACTIONS = ["No action","Fold","Check/Call","Bet/Raise"]
if "actions" not in st.session_state: st.session_state.actions = {0:{},1:{},2:{},3:{}}
cur_actions: Dict[int, Dict[str, float]] = st.session_state.actions.get(step, {})
grid_cols = st.columns( min(6, max(3, len(order_now))) )
for idx, seat in enumerate(order_now):
    with grid_cols[idx % len(grid_cols)]:
        st.markdown(f"**Seat {seat} — {pos_by_seat.get(seat,'')} {'(YOU)' if seat==hero_seat else ''}**")
        act = st.selectbox("Action", ACTIONS, index=ACTIONS.index(cur_actions.get(seat,{}).get("type","No action")), key=f"act_{step}_{seat}")
        size = st.number_input("Size (BB)", min_value=0.0, step=0.5, value=float(cur_actions.get(seat,{}).get("size",0.0)), key=f"size_{step}_{seat}")
        cur_actions[seat] = {"type": act, "size": float(size)}
st.session_state.actions[step] = cur_actions

# ================== Betting simulation ==================
def simulate_street(street_step:int, btn:int, seats:int, actions:Dict[int,Dict[str,float]], hero:int) -> Tuple[Dict[int,float], float, float, Optional[int]]:
    contribs = {i:0.0 for i in range(seats)}
    if street_step==0:
        sb = (btn + 1) % seats; bb = (btn + 2) % seats
        contribs[sb] += 0.5; contribs[bb] += 1.0
        current_bet = 1.0
    else:
        current_bet = 0.0
    pot_total = sum(contribs.values())
    last_agg = None
    order = betting_order(street_step, btn, seats, only_active=True)
    for s in order:
        a = actions.get(s, {"type":"No action","size":0.0})
        t = a.get("type","No action"); size = float(a.get("size",0.0))
        prev = contribs.get(s,0.0)
        if t=="Fold":
            pass
        elif t=="Check/Call":
            # AUTO-CALL: contribute exactly what is needed to match the current bet
            need = max(0.0, current_bet - prev)
            contribs[s] = prev + need
        elif t=="Bet/Raise":
            # 'size' is the TARGET bet (not increment)
            target = max(current_bet, 0.0)
            if street_step==0 and prev<1.0: target = max(1.0, target)  # behind blinds ensure ≥ BB
            target = max(target, size)
            need = max(0.0, target - prev)
            contribs[s] = prev + need
            if target > current_bet + 1e-9:
                current_bet = target; last_agg = s
        pot_total

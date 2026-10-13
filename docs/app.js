/* के लाउने? (K Laune): avatar engine, translations and app logic */
/* ===================== K Laune avatar + outfit engine (v3) ===================== */
const SKINS = ["#f1c9a5","#dfae84","#c68b59","#a86f45","#8a5530","#5f3a22"];
const HAIR_BLACK = "#1f1a17", HAIR_GREY = "#cfc9c1";
const INK = "#22264b";
const C = {
  tee:"#f2a93b", graphic:"#2f2f3a", polo:"#4e9a8e", shirt:"#dfe8f3", shirtF:"#f4e1e6", blouse:"#f4e1e6",
  sweater:"#9c3d54", hoodie:"#5b6b8c", hoodieF:"#c98aa4", jeans:"#3f5f8f", joggers:"#3b3f4f", chinos:"#a99263",
  trousers:"#3b3f55", trousersF:"#4a4560", shorts:"#5a6b7d", skirt:"#2f6f73", dress:"#2f6f73", sundress:"#e8b04a",
  jacket:"#34508a", bomber:"#5d6b3f", denim:"#4f78a8", blazer:"#2e3550", overcoat:"#7a6250", longcoat:"#a7764f",
  puffer:"#c2453d", cardigan:"#8a6d3b", istakot:"#2e3a55", shawl:"#6d5a8c",
  kurta:"#b8466a", kurtaSur:"#f3d9e0", kurtaM:"#e9e4d8", pyjama:"#f4f1ea", daura:"#efe4cb", dauraSur:"#e6dbc1",
  sari:"#c43d2f", sariBorder:"#e3b23c", sariBlouse:"#e3b23c",
  shoe:"#3a2f2a", sneaker:"#f4f4f6", sole:"#c9cdd6", tights:"#2b2a35", strap:"#7a5230",
  scarf:"#f2a93b", beanie:"#e9e2d0", raincoat:"#c9d84a", helmet:"#eef2f7", cap:"#4e9a8e",
  leather:"#2b2b30", flannel:"#b5413a", linen:"#e8dfcc", kurti:"#2f8f83",
  patasiBlouse:"#b3262e", patasi:"#1e1b22", patasiBorder:"#c22f35", patuka:"#f3efe6",
  bakhu:"#4a2f2a", bakhuBlouse:"#e9dfc8", cholo:"#d9577a", gunyu:"#3f8f5a", gunyuBorder:"#e3b23c"
};

/* ---------- Wardrobes: style → body → warmth tier 0 (very cold) … 5 (hot) ---------- */
const W_ = (label, top, o = {}) => ({label, top, ...o});
const WARDROBE = {
  street: {
    m:[W_("Puffer over hoodie","hoodie",{outer:"puffer",legs:"jeans",shoes:"sneaker"}), W_("Bomber jacket","tee",{outer:"bomber",legs:"jeans",shoes:"sneaker"}),
       W_("Hoodie","hoodie",{legs:"joggers",shoes:"sneaker"}), W_("Denim jacket","tee",{outer:"denim",legs:"jeans",shoes:"sneaker"}),
       W_("Graphic tee","graphic",{legs:"jeans",shoes:"sneaker"}), W_("Tee & shorts","graphic",{legs:"shorts",shoes:"sneaker",light:true})],
    f:[W_("Puffer over hoodie","hoodie",{outer:"puffer",legs:"jeans",shoes:"sneaker"}), W_("Bomber jacket","tee",{outer:"bomber",legs:"jeans",shoes:"sneaker"}),
       W_("Hoodie","hoodie",{legs:"jeans",shoes:"sneaker"}), W_("Denim jacket","tee",{outer:"denim",legs:"jeans",shoes:"sneaker"}),
       W_("Graphic tee","graphic",{legs:"jeans",shoes:"sneaker"}), W_("Tee & skirt","tee",{legs:"skirt",shoes:"sneaker",light:true})]
  },
  smart: {
    m:[W_("Overcoat","shirt",{outer:"overcoat",legs:"trousers",shoes:"shoe"}), W_("Shirt & blazer","shirt",{outer:"blazer",legs:"trousers",shoes:"shoe"}),
       W_("Sweater over shirt","sweater",{collar:true,legs:"chinos",shoes:"shoe"}), W_("Shirt","shirt",{legs:"chinos",shoes:"shoe"}),
       W_("Shirt, sleeves rolled","shirt",{rolled:true,legs:"chinos",shoes:"shoe"}), W_("Polo shirt","polo",{legs:"chinos",shoes:"shoe",light:true})],
    f:[W_("Long coat","blouse",{outer:"longcoat",legs:"trousers",shoes:"boot"}), W_("Blazer","blouse",{outer:"blazer",legs:"trousers",shoes:"shoe"}),
       W_("Sweater","sweater",{legs:"trousers",shoes:"shoe"}), W_("Blouse","blouse",{legs:"trousers",shoes:"shoe"}),
       W_("Short-sleeve blouse","blouse",{short:true,legs:"trousers",shoes:"shoe"}), W_("Light blouse","blouse",{short:true,legs:"trousers",shoes:"sandal",light:true})]
  },
  dress: {
    f:[W_("Dress, coat & tights","dress",{outer:"longcoat",legs:"tights",shoes:"boot"}), W_("Dress & cardigan","dress",{outer:"cardigan",legs:"tights",shoes:"boot"}),
       W_("Long-sleeve dress","dress",{legs:"tights",shoes:"shoe"}), W_("Long-sleeve dress","dress",{legs:"bare",shoes:"shoe"}),
       W_("Dress","dress",{short:true,legs:"bare",shoes:"shoe"}), W_("Sundress","sundress",{short:true,legs:"bare",shoes:"sandal",light:true})]
  },
  kurta: {
    m:[W_("Puffer over kurta","kurtaM",{outer:"puffer",legs:"pyjama",shoes:"shoe"}), W_("Kurta + coat","kurtaM",{outer:"jacket",legs:"pyjama",shoes:"shoe"}),
       W_("Kurta + istakot","kurtaM",{outer:"istakot",legs:"pyjama",shoes:"shoe"}), W_("Kurta-pyjama","kurtaM",{legs:"pyjama",shoes:"shoe"}),
       W_("Kurta-pyjama","kurtaM",{legs:"pyjama",shoes:"sandal"}), W_("Cotton kurta","kurtaM",{legs:"pyjama",shoes:"sandal",light:true})],
    f:[W_("Puffer over kurta","kurta",{outer:"puffer",legs:"suruwal",shoes:"shoe"}), W_("Kurta + shawl","kurta",{outer:"shawl",legs:"suruwal",shoes:"shoe"}),
       W_("Kurta + cardigan","kurta",{outer:"cardigan",legs:"suruwal",shoes:"shoe"}), W_("Kurta-suruwal","kurta",{legs:"suruwal",shoes:"shoe"}),
       W_("Cotton kurta","kurta",{short:true,legs:"suruwal",shoes:"sandal"}), W_("Light cotton kurta","kurta",{short:true,legs:"suruwal",shoes:"sandal",light:true})]
  },
  daura: {
    m:[W_("Puffer over daura","daura",{outer:"puffer",legs:"dauraSur",shoes:"shoe",topi:true}), W_("Daura + coat","daura",{outer:"jacket",legs:"dauraSur",shoes:"shoe",topi:true}),
       W_("Daura + istakot","daura",{outer:"istakot",legs:"dauraSur",shoes:"shoe",topi:true}), W_("Daura-suruwal","daura",{legs:"dauraSur",shoes:"shoe",topi:true}),
       W_("Daura-suruwal","daura",{legs:"dauraSur",shoes:"shoe",topi:true}), W_("Light daura","daura",{legs:"dauraSur",shoes:"sandal",topi:true,light:true})]
  },
  sari: {
    f:[W_("Puffer over sari","sari",{outer:"puffer",legs:"sari",shoes:"shoe"}), W_("Sari + shawl","sari",{outer:"shawl",legs:"sari",shoes:"shoe"}),
       W_("Sari + cardigan","sari",{outer:"cardigan",legs:"sari",shoes:"shoe"}), W_("Sari","sari",{legs:"sari",shoes:"shoe"}),
       W_("Sari","sari",{legs:"sari",shoes:"sandal"}), W_("Light cotton sari","sari",{legs:"sari",shoes:"sandal",light:true})]
  }
};
const STYLE_NAMES = {street:"Streetwear", smart:"Smart casual", dress:"Dresses", kurta:"Kurta", daura:"Daura-suruwal", sari:"Sari"};
const STYLE_SUB = {street:"tees, hoodies, jeans", smart:"shirts, blazers, trousers", dress:"dresses, cardigans", kurta:"kurta-suruwal / pyjama", daura:"with dhaka topi", sari:"with shawl in winter"};
const stylesFor = sex => Object.keys(WARDROBE).filter(k => WARDROBE[k][sex === "f" ? "f" : "m"]);
const defaultStyle = (age, sex) => age === "kid" || age === "young" ? "street" : age === "adult" ? "smart"
                                 : sex === "f" ? "sari" : sex === "m" ? "daura" : "kurta";
const HAIRS = {m:[["short","Short"],["buzz","Buzz cut"],["curly","Curly"],["tied","Tied back"]],
               f:[["long","Long"],["bun","Bun"],["braid","Braid"],["ponytail","Ponytail"],["bob","Bob"]],
               n:[["short","Short"],["buzz","Buzz cut"],["curly","Curly"],["long","Long"],["bun","Bun"],["bob","Bob"]]};
const TIER_MAX = [8,13,17,21,30,99];
const tierOf = t => TIER_MAX.findIndex(x => t < x);
const wardrobe = (P, tier) => WARDROBE[P.style][P.sex === "f" ? "f" : "m"][tier];

/* ---------- Body geometry ---------- */
const GEO = {
  m:{arms:[[34,76,27,136],[86,76,93,136]], hands:[[26,141],[94,141]], sl:36, sr:84,
     torso:(y, f = 0) => `M37 68 Q60 62 83 68 L${86 + f} ${y} Q60 ${y + 5} ${34 - f} ${y} Z`,
     legs:"M40 128 H80 L78 202 H63 L60 150 L57 202 H42 Z", legX:[49,71]},
  f:{arms:[[38,77,32,134],[82,77,88,134]], hands:[[31,139],[89,139]], sl:39, sr:81,
     torso:(y, f = 0) => y <= 112
       ? `M40 69 Q60 64 80 69 C79 88 77 100 77 ${y} Q60 ${y + 4} 43 ${y} C43 100 41 88 40 69 Z`
       : `M40 69 Q60 64 80 69 C79 88 76 100 77 110 C78 118 ${81 + f} ${y - 14} ${81 + f} ${y} Q60 ${y + 5} ${39 - f} ${y} C${39 - f} ${y - 14} 42 118 43 110 C44 100 41 88 40 69 Z`,
     legs:"M43 126 H77 L75 202 H63 L60 150 L57 202 H45 Z", legX:[50,70]}
};

/* ---------- Avatar ---------- */
function avatar(o, P){
  const W = o.outfit || wardrobe(P, o.tier), G = GEO[P.sex === "f" ? "f" : "m"], fem = P.sex === "f";
  const skin = SKINS[P.skin], hairCol = P.age === "elder" ? HAIR_GREY : HAIR_BLACK, cold = o.tier === 0;
  const tunicLen = W.top === "kurta" || W.top === "kurtaM" || W.top === "daura" ? (fem ? 172 : 168) : W.top === "kurti" ? 152 : W.top === "bakhuM" ? 170 : 0;
  const long = !(W.short || ["tee","graphic","polo","sundress"].includes(W.top));
  const modern = o.outfit ? o.outfit.family === "modern" : ["street"].includes(P.style);
  const backpack = modern && !["smart"].includes(P.style) && (P.age === "young" || P.age === "kid") && !o.raincoat;
  let s = `<ellipse cx="60" cy="210" rx="34" ry="5" fill="var(--floor)"/>`;

  // hair behind the body
  if (!o.raincoat) s += hairBack(P.hair, hairCol, fem);
  // hood behind the head
  if (W.top === "hoodie" && W.outer !== "puffer") s += `<path d="M42 64 Q60 46 78 64 L74 72 Q60 64 46 72 Z" fill="${fem ? C.hoodieF : C.hoodie}"/>`;

  // ---- legs + shoes
  s += legs(W, G, skin, fem);
  s += shoes(W.shoes, G, skin);

  // ---- arms (skin) + hands
  s += G.arms.map(([x1,y1,x2,y2]) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${skin}" stroke-width="${fem ? 9 : 10}" stroke-linecap="round"/>`).join("");
  s += G.hands.map(([x,y]) => `<circle cx="${x}" cy="${y}" r="${fem ? 5.5 : 6}" fill="${o.gloves ? "#2b2f45" : skin}"/>`).join("");

  // ---- main top
  const topCol = {tee:C.tee, graphic:C.graphic, polo:C.polo, shirt:fem ? C.shirtF : C.shirt, blouse:C.blouse, sweater:C.sweater,
    hoodie:fem ? C.hoodieF : C.hoodie, kurta:C.kurta, kurtaM:C.kurtaM, daura:C.daura, sari:C.sariBlouse, dress:C.dress, sundress:C.sundress,
    flannel:C.flannel, linen:C.linen, kurti:C.kurti, patasi:C.patasiBlouse, bakhu:C.bakhuBlouse, bakhuM:C.bakhu, cholo:C.cholo}[W.top];
  if (W.top === "dress" || W.top === "sundress") {
    s += `<path d="${fem ? "M43 108 H77 L90 184 Q60 190 30 184 Z" : "M38 108 H82 L92 184 Q60 190 28 184 Z"}" fill="${topCol}"/>`;
    s += `<path d="${G.torso(112)}" fill="${topCol}"/><path d="M43 110 Q60 114 77 110" stroke="#000" stroke-opacity=".15" stroke-width="2" fill="none"/>`;
    if (W.top === "sundress") s += `<path d="M47 70 Q60 80 73 70" stroke="${skin}" stroke-width="7" fill="none"/>`;
  } else if (W.top === "bakhu") {          // Sherpa chuba: sleeveless robe over a blouse, striped apron
    s += `<path d="${G.torso(112)}" fill="${C.bakhuBlouse}"/>`;
    s += `<path d="M${G.sl + 4} 70 L${G.sr - 4} 70 L${G.sr + 8} 204 H${G.sl - 8} Z" fill="${C.bakhu}"/>`;
    s += [0,1,2,3,4,5,6,7,8].map(i => `<rect x="49" y="${122 + i * 8.6}" width="22" height="8.6" fill="${["#c43d2f","#e3b23c","#3f8f5a","#3d6fb6","#e9dfc8","#c43d2f","#e3b23c","#3f8f5a","#3d6fb6"][i]}"/>`).join("");
    s += `<rect x="${G.sl}" y="114" width="${G.sr - G.sl}" height="7" rx="2" fill="#c43d2f"/>`;
  } else if (W.top === "patasi" || W.top === "cholo") {   // haku patasi / gunyu-cholo: blouse, patuka at the waist
    s += `<path d="${G.torso(122)}" fill="${topCol}"/>`;
    if (W.top === "cholo") s += `<path d="M44 68 Q60 92 76 110 M76 68 Q60 92 44 110" stroke="#000" stroke-opacity=".18" stroke-width="1.6" fill="none"/>`;
    s += `<rect x="${G.sl + 1}" y="116" width="${G.sr - G.sl - 2}" height="12" rx="3" fill="${C.patuka}"/>`;
  } else if (W.top === "sari") {
    s += `<path d="${G.torso(110)}" fill="${C.sariBlouse}"/><path d="M43 108 Q60 112 77 108 L78 126 H42 Z" fill="${skin}"/>`;
  } else if (tunicLen) {
    s += `<path d="${G.torso(tunicLen, fem ? 6 : 4)}" fill="${topCol}"/>`;
    s += `<path d="M${fem ? 35 : 33} ${tunicLen - 26} L${fem ? 33 : 31} ${tunicLen} M${fem ? 85 : 87} ${tunicLen - 26} L${fem ? 87 : 89} ${tunicLen}" stroke="#000" stroke-opacity=".15" stroke-width="1.5"/>`;
    if (W.top === "daura") s += `<path d="M44 68 Q66 92 80 126" stroke="#000" stroke-opacity=".22" stroke-width="1.6" fill="none"/><circle cx="70" cy="100" r="1.8" fill="#8a6d3b"/><circle cx="77" cy="114" r="1.8" fill="#8a6d3b"/>`;
    else s += `<path d="M54 64 L60 80 L66 64" stroke="${W.top === "kurta" ? skin : "#c9c2b0"}" stroke-width="3" fill="none"/>`;
  } else {
    s += `<path d="${G.torso(130)}" fill="${topCol}"/>`;
    if (W.top === "flannel") s += [78,94,110].map(y => `<path d="M${G.sl} ${y} H${G.sr}" stroke="#4a1f1c" stroke-opacity=".55" stroke-width="3"/>`).join("") +
      [46,56,64,74].map(x => `<path d="M${x} 66 V130" stroke="#4a1f1c" stroke-opacity=".4" stroke-width="2.5"/>`).join("");
    if (W.top === "graphic") s += `<circle cx="60" cy="96" r="9" fill="${C.tee}"/><path d="M55 96 L59 100 L66 92" stroke="${INK}" stroke-width="2" fill="none"/>`;
    if (W.top === "tee" || W.top === "graphic") s += `<path d="M52 64 Q60 70 68 64" stroke="${skin}" stroke-width="4" fill="none"/>`;
    if (W.top === "sweater") s += `<path d="M${fem ? 40 : 35} 125 Q60 130 ${fem ? 80 : 85} 125" stroke="#000" stroke-opacity=".18" stroke-width="5" fill="none"/>`;
    if (W.top === "hoodie") s += `<path d="M48 108 H72 L74 124 H46 Z" fill="#000" opacity=".12"/><path d="M55 66 V82 M65 66 V82" stroke="#fff" stroke-width="1.5"/>`;
    if (["shirt","polo","flannel","linen"].includes(W.top) || W.collar) s += `<path d="M51 63 L60 72 L55 76 Z M69 63 L60 72 L65 76 Z" fill="${W.collar ? C.shirt : "#fff"}" stroke="#000" stroke-opacity=".15"/>`;
    if (["shirt","flannel","linen"].includes(W.top) && !W.collar) s += `<path d="M60 72 V128" stroke="#000" stroke-opacity=".15" stroke-width="1.2"/>`;
    if (W.top === "blouse") s += `<path d="M50 65 Q60 76 70 65" stroke="#fff" stroke-width="3" fill="none"/>`;
  }
  // ---- sleeves
  const outerSleeve = {jacket:C.jacket, bomber:C.bomber, denim:C.denim, leather:C.leather, blazer:C.blazer, overcoat:C.overcoat, longcoat:C.longcoat,
                       puffer:C.puffer, cardigan:C.cardigan}[W.outer];
  const sleeveCol = outerSleeve || topCol, sw = W.outer === "puffer" ? 17 : outerSleeve ? 15 : (fem ? 11 : 13);
  if (W.top !== "sundress") s += G.arms.map(([x1,y1,x2,y2]) => {
    if (long && !W.rolled || outerSleeve) return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2 - 3}" stroke="${sleeveCol}" stroke-width="${sw}" stroke-linecap="round"/>` +
      (W.outer === "bomber" ? `<line x1="${x2 + (x1 - x2) * .08}" y1="${y2 - 8}" x2="${x2}" y2="${y2 - 3}" stroke="#4a5532" stroke-width="${sw}" stroke-linecap="round"/>` : "");
    const k = W.rolled ? .55 : .3;
    return `<line x1="${x1}" y1="${y1}" x2="${x1 + (x2 - x1) * k}" y2="${y1 + (y2 - y1) * k}" stroke="${topCol}" stroke-width="${fem ? 12 : 14}" stroke-linecap="round"/>`;
  }).join("");
  // sari pallu
  if (W.top === "cholo" && W.outer !== "puffer") s += `<path d="M80 126 L86 122 L46 64 L38 69 Z" fill="${C.gunyu}"/><path d="M82 124 L86 122 L46 64 L43 66 Z" fill="${C.gunyuBorder}"/>`;
  if (W.top === "sari" && W.outer !== "puffer") s += `<path d="M80 128 L86 124 L46 64 L38 69 Z" fill="${C.sari}"/><path d="M82 126 L86 124 L46 64 L43 66 Z" fill="${C.sariBorder}"/>`;

  if (W.top === "bakhuM") s += `<path d="M44 68 Q64 90 78 128" stroke="#000" stroke-opacity=".25" stroke-width="1.6" fill="none"/><rect x="${G.sl - 1}" y="118" width="${G.sr - G.sl + 2}" height="8" rx="3" fill="#c43d2f"/>`;
  // ---- outer layers
  const len = tunicLen ? 150 : W.top === "sari" || W.top === "dress" ? 150 : 136;
  const panels = (col, l, gap = 3) => `<path d="M${G.sl} 67 L${60 - gap} 65 L${60 - gap - 1} ${l} L${G.sl - 3} ${l - 2} Z" fill="${col}"/><path d="M${G.sr} 67 L${60 + gap} 65 L${60 + gap + 1} ${l} L${G.sr + 3} ${l - 2} Z" fill="${col}"/>`;
  if (W.outer === "jacket") s += panels(C.jacket, len) + `<path d="M49 64 L57 65 L52 80 Z M71 64 L63 65 L68 80 Z" fill="#26407a"/>`;
  if (W.outer === "leather") s += panels(C.leather, 132) + `<path d="M48 63 L58 67 L53 78 Z M72 63 L62 67 L67 78 Z" fill="#1c1c20"/><path d="M57 66 L56 132" stroke="#9aa0ad" stroke-width="1.4"/>`;
  if (W.outer === "denim") s += panels(C.denim, 134) + `<path d="M${G.sl + 4} 92 H55 M65 92 H${G.sr - 4}" stroke="#fff" stroke-opacity=".5" stroke-width="1.2" stroke-dasharray="2 2"/><path d="M48 63 L57 66 L53 76 Z M72 63 L63 66 L67 76 Z" fill="#3d6492"/>`;
  if (W.outer === "bomber") s += `<path d="${G.torso(130)}" fill="${C.bomber}"/><path d="M${fem ? 42 : 36} 126 Q60 131 ${fem ? 78 : 84} 126" stroke="#4a5532" stroke-width="6" fill="none"/><path d="M60 66 V130" stroke="#c9b98f" stroke-width="1.5"/><path d="M50 64 Q60 70 70 64" stroke="#4a5532" stroke-width="5" fill="none"/>`;
  if (W.outer === "blazer") s += panels(C.blazer, len, 5) + `<path d="M${G.sl + 6} 66 L55 66 L57 92 Z M${G.sr - 6} 66 L65 66 L63 92 Z" fill="#232a40"/><circle cx="56" cy="110" r="1.6" fill="#cdb68f"/>`;
  if (W.outer === "overcoat" || W.outer === "longcoat") {
    const col = C[W.outer];
    s += panels(col, fem ? 176 : 170, 2) + `<path d="M${G.sl + 5} 66 L56 66 L57 90 Z M${G.sr - 5} 66 L64 66 L63 90 Z" fill="#000" opacity=".15"/>`;
    if (W.outer === "longcoat") s += `<rect x="${G.sl - 2}" y="112" width="${G.sr - G.sl + 4}" height="6" rx="2" fill="#7d5636"/>`;
  }
  if (W.outer === "cardigan") s += panels(C.cardigan, len, 6);
  if (W.outer === "istakot") s += `<path d="M${G.sl + 3} 68 L56 66 L57 128 L${G.sl + 2} 126 Z" fill="${C.istakot}"/><path d="M${G.sr - 3} 68 L64 66 L63 128 L${G.sr - 2} 126 Z" fill="${C.istakot}"/><circle cx="61" cy="90" r="1.4" fill="#cdb68f"/><circle cx="61" cy="104" r="1.4" fill="#cdb68f"/>`;
  if (W.outer === "shawl") s += `<path d="M${G.sl - 3} 70 Q60 58 ${G.sr + 3} 70 L${G.sr + 6} 98 Q60 88 ${G.sl - 6} 98 Z" fill="${C.shawl}"/><path d="M${G.sl - 6} 96 L${G.sl} 150 L${G.sl + 10} 148 L${G.sl + 8} 92 Z M${G.sr + 6} 96 L${G.sr} 150 L${G.sr - 10} 148 L${G.sr - 8} 92 Z" fill="${C.shawl}"/>`;
  if (W.outer === "puffer") {
    const pl = fem && (W.top === "dress" || W.top === "sari" || tunicLen) ? 160 : len;
    s += `<path d="${G.torso(pl, 3)}" fill="${C.puffer}"/>` + [84,100,116,132,148].filter(y => y < pl - 4).map(y => `<path d="M${G.sl - 2} ${y} Q60 ${y + 5} ${G.sr + 2} ${y}" stroke="#000" stroke-opacity=".16" stroke-width="2" fill="none"/>`).join("");
  }
  if (cold || W.scarf) s += `<path d="M44 64 Q60 74 76 64 L77 70 Q60 80 43 70 Z" fill="${C.scarf}"/><rect x="62" y="70" width="8" height="22" rx="3" fill="${C.scarf}"/>`;
  if (backpack) s += `<path d="M${G.sl + 6} 66 L${G.sl + 9} 112 M${G.sr - 6} 66 L${G.sr - 9} 112" stroke="${C.strap}" stroke-width="4" stroke-linecap="round"/>`;
  if (o.raincoat) s += `<path d="M28 62 Q60 50 92 62 L104 160 Q60 168 16 160 Z" fill="${C.raincoat}" opacity=".93"/><path d="M60 58 V164" stroke="#000" stroke-opacity=".15" stroke-width="1.5"/>`;

  // ---- neck + head
  s += `<rect x="${fem ? 55 : 54}" y="54" width="${fem ? 10 : 12}" height="12" rx="4" fill="${skin}"/><circle cx="60" cy="38" r="${P.age === "kid" ? 21 : 19}" fill="${skin}"/>`;
  const hatted = cold || W.topi || o.cap || o.raincoat;
  if (o.raincoat) s += `<path d="M38 44 Q36 12 60 12 Q84 12 82 44 Q80 22 60 21 Q40 22 38 44 Z" fill="${C.raincoat}"/>`;
  else if (!hatted) s += hairFront(P.hair, hairCol);
  else if (["long","braid","bob","ponytail"].includes(P.hair)) s += `<path d="M41 36 Q41 28 46 26 L44 42 Z M79 36 Q79 28 74 26 L76 42 Z" fill="${hairCol}"/>`;
  if (P.hair === "braid" && !o.raincoat) s += [56,66,76,86,96].map((y,i) => `<ellipse cx="${74 + i * .6}" cy="${y}" rx="4.4" ry="5.8" fill="${hairCol}"/>`).join("");
  if (!o.raincoat) {
    if (cold) s += `<path d="M40 36 Q40 14 60 14 Q80 14 80 36 Z" fill="${C.beanie}"/><rect x="39" y="31" width="42" height="7" rx="3.5" fill="#d5ccb5"/><circle cx="60" cy="12" r="4" fill="${C.beanie}"/>`;
    else if (W.topi) s += `<path d="M42 28 L45 13 Q60 9 75 13 L78 28 Z" fill="#8b2b2b"/><path d="M44 18 H76 M43 23 H77" stroke="#f2a93b" stroke-width="2" stroke-dasharray="3 2"/>`;
    else if (o.cap) s += `<path d="M41 31 Q42 16 60 16 Q78 16 79 31 Z" fill="${C.cap}"/><path d="M58 30 Q76 27 90 33 Q76 35 58 33 Z" fill="#3c7f74"/>`;
  }
  // ---- face
  if (o.sunglasses) s += `<rect x="46" y="36" width="11" height="6" rx="3" fill="${INK}"/><rect x="63" y="36" width="11" height="6" rx="3" fill="${INK}"/><line x1="57" y1="38" x2="63" y2="38" stroke="${INK}" stroke-width="1.5"/>`;
  else {
    s += `<circle cx="53" cy="39" r="2" fill="${INK}"/><circle cx="67" cy="39" r="2" fill="${INK}"/>`;
    if (fem) s += `<path d="M50 36 L48 34.5 M70 36 L72 34.5" stroke="${INK}" stroke-width="1.2" stroke-linecap="round"/>`;
    if (P.age === "elder") s += `<circle cx="53" cy="39" r="5.5" stroke="${INK}" stroke-width="1.3" fill="none"/><circle cx="67" cy="39" r="5.5" stroke="${INK}" stroke-width="1.3" fill="none"/><line x1="58.5" y1="39" x2="61.5" y2="39" stroke="${INK}" stroke-width="1.3"/>`;
  }
  if (P.facial === "beard") s += `<path d="M42 40 Q44 58 60 59 Q76 58 78 40 Q74 50 66 50 Q60 46 54 50 Q46 50 42 40 Z" fill="${hairCol}"/>`;
  if (P.facial === "mustache" || P.facial === "beard") s += `<path d="M53 46 Q60 42 67 46 Q60 45 53 46 Z" fill="${hairCol}" stroke="${hairCol}" stroke-width="2"/>`;
  s += `<path d="M54 48 Q60 52 66 48" stroke="${INK}" stroke-width="1.8" fill="none" stroke-linecap="round"/>`;
  if (fem) s += `<path d="M56 49 Q60 51 64 49" stroke="#b5505a" stroke-width="1.6" fill="none" stroke-linecap="round" opacity=".7"/>`;

  // ---- accessories
  if (o.helmet) s += `<path d="M6 160 A13 13 0 0 1 32 160 Z" fill="${C.helmet}" stroke="${INK}" stroke-width="1.3"/><path d="M10 156 H28" stroke="#5d7fa8" stroke-width="3"/>`;
  else if (o.bottle) s += `<rect x="${G.hands[0][0] - 7}" y="134" width="10" height="20" rx="3" fill="#7cc4e0" stroke="${INK}" stroke-width="1.2"/>`;
  const hx = G.hands[1][0], hy = G.hands[1][1];
  if (o.umbrella === "closed") s += `<path d="M${hx + 4} ${hy - 1} L${hx + 18} ${hy + 55}" stroke="${INK}" stroke-width="3" stroke-linecap="round"/><path d="M${hx + 1} ${hy - 5} Q${hx - 2} ${hy - 13} ${hx + 4} ${hy - 14}" stroke="${INK}" stroke-width="3" fill="none" stroke-linecap="round"/><path d="M${hx + 6} ${hy + 5} L${hx + 14} ${hy + 9} L${hx + 20} ${hy + 51} L${hx + 16} ${hy + 55} Z" fill="${C.jacket}"/>`;
  if (o.umbrella === "open") s += `<line x1="${hx}" y1="${hy}" x2="${hx}" y2="-2" stroke="${INK}" stroke-width="3"/><path d="M${hx - 40} 2 Q${hx} -34 ${hx + 40} 2 Q${hx + 30} -4 ${hx + 20} 2 Q${hx + 10} -4 ${hx} 2 Q${hx - 10} -4 ${hx - 20} 2 Q${hx - 30} -4 ${hx - 40} 2 Z" fill="${C.jacket}"/>`;
  if (o.rain) s += [[8,20],[22,60],[114,70],[4,100],[120,120],[30,186]].map(([x,y]) => `<line x1="${x}" y1="${y}" x2="${x - 3}" y2="${y + 9}" stroke="${C.jacket}" stroke-width="2" stroke-linecap="round" opacity=".7"/>`).join("");

  const scale = {kid:.76, young:.93, adult:1, elder:.97}[P.age];
  const body = scale === 1 ? s : `<g transform="translate(60 210) scale(${scale}) translate(-60 -210)">${s}</g>`;
  const vb = o.umbrella === "open" ? "-12 -40 152 262" : "-12 -14 146 236";
  return `<svg viewBox="${vb}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${((o.time || "") + ": " + W.label).replace(/&/g, "and")}">${body}</svg>`;
}

function legs(W, G, skin, fem){
  const [lx, rx] = G.legX;
  const bare = (y0, col = skin) => `<line x1="${lx}" y1="${y0}" x2="${lx}" y2="200" stroke="${col}" stroke-width="9" stroke-linecap="round"/><line x1="${rx}" y1="${y0}" x2="${rx}" y2="200" stroke="${col}" stroke-width="9" stroke-linecap="round"/>`;
  switch (W.legs) {
    case "jeans": return `<path d="${G.legs}" fill="${C.jeans}"/><path d="M${lx + 2} 160 V200 M${rx - 2} 160 V200" stroke="#fff" stroke-opacity=".15" stroke-width="1"/>`;
    case "joggers": return `<path d="${G.legs}" fill="${C.joggers}"/><path d="M${lx - 6} 196 H${lx + 7} M${rx - 7} 196 H${rx + 6}" stroke="#2a2d3a" stroke-width="5"/>`;
    case "chinos": return `<path d="${G.legs}" fill="${C.chinos}"/>`;
    case "trousers": return `<path d="${G.legs}" fill="${fem ? C.trousersF : C.trousers}"/>`;
    case "shorts": return bare(160) + `<path d="M40 126 H80 L79 164 H62 L60 146 L58 164 H41 Z" fill="${C.shorts}"/>`;
    case "skirt": return bare(176) + `<path d="M43 118 H77 L86 180 Q60 185 34 180 Z" fill="${C.skirt}"/>`;
    case "tights": return bare(170, C.tights);
    case "bare": return bare(176);
    case "pyjama": return `<path d="M41 128 H79 L79 202 H63 L60 150 L57 202 H41 Z" fill="${C.pyjama}"/>`;
    case "suruwal": return `<path d="M43 128 H77 L74 202 H64 L60 150 L56 202 H46 Z" fill="${C.kurtaSur}"/>`;
    case "dauraSur": return `<path d="M42 128 H78 L75 202 H64 L60 150 L56 202 H45 Z" fill="${C.dauraSur}"/>`;
    case "patasi": return `<path d="M38 124 H82 L88 204 H32 Z" fill="${C.patasi}"/><path d="M33 194 H87 L88 204 H32 Z M66 124 L70 204 H75 L71 124 Z" fill="${C.patasiBorder}"/>`;
    case "gunyu": return `<path d="M38 124 H82 L88 204 H32 Z" fill="${C.gunyu}"/><path d="M33 196 H87 L88 204 H32 Z" fill="${C.gunyuBorder}"/><path d="M55 130 L52 196 M62 130 L62 196 M69 130 L72 196" stroke="#000" stroke-opacity=".12" stroke-width="1.5"/>`;
    case "none": return "";
    case "sari": return `<path d="M38 124 H82 L88 204 H32 Z" fill="${C.sari}"/><path d="M33 196 H87 L88 204 H32 Z" fill="${C.sariBorder}"/><path d="M55 130 L52 196 M62 130 L62 196 M69 130 L72 196" stroke="#000" stroke-opacity=".12" stroke-width="1.5"/>`;
  }
  return `<path d="${G.legs}" fill="${C.trousers}"/>`;
}
function shoes(type, G, skin){
  const [lx, rx] = G.legX;
  const pair = f => f(lx - 1) + f(rx + 1);
  switch (type) {
    case "sneaker": return pair(x => `<ellipse cx="${x}" cy="203" rx="10.5" ry="5.5" fill="${C.sneaker}"/><path d="M${x - 9} 206 H${x + 9}" stroke="${C.sole}" stroke-width="2"/><path d="M${x - 3} 200 L${x + 4} 203" stroke="${C.cap}" stroke-width="2"/>`);
    case "sandal": return pair(x => `<ellipse cx="${x}" cy="204" rx="9" ry="4" fill="${skin}"/><path d="M${x - 9} 207 H${x + 9}" stroke="#7a5230" stroke-width="2.5"/><path d="M${x - 5} 201 Q${x} 205 ${x + 5} 201" stroke="#7a5230" stroke-width="2" fill="none"/>`);
    case "boot": return pair(x => `<path d="M${x - 5} 184 H${x + 5} L${x + 6} 200 Q${x + 11} 202 ${x + 10} 207 H${x - 8} Q${x - 9} 201 ${x - 5} 200 Z" fill="#3a2a24"/>`);
    default: return pair(x => `<ellipse cx="${x}" cy="204" rx="10" ry="5" fill="${C.shoe}"/>`);
  }
}
function hairBack(style, col, fem){
  switch (style) {
    case "long": return `<path d="M41 30 Q38 70 40 98 H80 Q82 70 79 30 Z" fill="${col}"/>`;
    case "bob": return `<path d="M40 32 Q37 52 40 60 H80 Q83 52 80 32 Z" fill="${col}"/>`;
    case "ponytail": return `<path d="M76 26 Q92 34 86 74 Q84 82 80 76 Q86 50 74 34 Z" fill="${col}"/>`;
    case "tied": return `<circle cx="78" cy="26" r="6" fill="${col}"/>`;
    case "bun": return `<circle cx="60" cy="17" r="8" fill="${col}"/>`;
  }
  return "";
}
function hairFront(style, col){
  switch (style) {
    case "buzz": return `<path d="M42 34 Q42 19 60 19 Q78 19 78 34 Q72 25 60 25 Q48 25 42 34 Z" fill="${col}"/>`;
    case "curly": return `<path d="M40 36 Q36 26 44 22 Q44 14 52 16 Q56 10 63 14 Q70 11 74 18 Q82 18 80 28 Q84 32 80 37 Q74 27 60 27 Q46 27 40 36 Z" fill="${col}"/>`;
    case "bun": case "long": case "braid": case "ponytail": case "tied":
      return `<path d="M41 40 Q38 17 60 17 Q82 17 79 40 Q76 26 60 26 Q44 26 41 40 Z" fill="${col}"/>`;
    case "bob": return `<path d="M40 42 Q37 17 60 17 Q83 17 80 42 Q78 30 66 25 Q52 28 46 26 Q42 32 40 42 Z" fill="${col}"/>`;
    default: return `<path d="M41 36 Q40 17 60 17 Q80 17 79 34 Q70 25 54 27 Q46 28 41 36 Z" fill="${col}"/>`;
  }
}

/* ---------- Personalised looks for one forecast day ---------- */
function looks(d, P){
  const x = d.extras || {}, span = d.max_temp_c - d.min_temp_c;
  const pref = [-2, 0, 2][P.cold];
  // 8 am / 7 pm temperatures from the weather models' daily curve, scaled to our forecast low/high
  const morningT = d.min_temp_c + (x.morning_frac ?? 0.25) * span;
  const eveningT = d.min_temp_c + (x.evening_frac ?? 0.45) * span;
  // "feels like": humidity, wind and sun (from the models), the ride's wind chill, personal preference
  const feelM = morningT + (x.feels_min_adj ?? 0) - (P.ride ? 4 : 0) + pref;
  const feelA = d.max_temp_c + (x.feels_max_adj ?? 0) - (P.ride ? 2 : 0) + pref;
  const feelE = eveningT + (x.feels_min_adj ?? 0) - (P.ride ? 4 : 0) + pref;
  const m = tierOf(feelM), a = tierOf(feelA), e = tierOf(feelE), heavy = d.rain_prob >= 0.7;
  const sunny = !d.umbrella && (x.uv_max ?? 0) >= 6;
  const rainA = d.umbrella ? (P.ride ? {raincoat:true} : {umbrella: heavy ? "open" : "closed"}) : {};
  const rainM = d.umbrella ? (P.ride ? {raincoat:true} : {umbrella:"closed"}) : {};
  const aft = {tier:a, t:d.max_temp_c, feel:feelA, time:"Afternoon", rain:heavy && d.umbrella, helmet:P.ride,
    cap: P.style === "street" && (feelA >= 26 || sunny) && !d.umbrella,
    sunglasses: (feelA >= 30 || (x.uv_max ?? 0) >= 8) && !d.umbrella, bottle: a === 5, ...rainA};
  const mor = {tier:m, t:morningT, feel:feelM, time:"Morning", gloves:P.ride && feelM < 12, helmet:P.ride, ...rainM};
  const L = wardrobe(P, m).label === wardrobe(P, a).label ? [{...aft, time:"All day", tRange:[morningT, d.max_temp_c]}] : [mor, aft];
  // evening: only worth mentioning when it calls for something warmer than the afternoon
  L.evening = e < a ? {tier:e, t:eveningT, feel:feelE} : null;
  return L;
}
const labelOf = (o, P) => (o.outfit || wardrobe(P, o.tier)).label;

/* ---------- Outfit ideas: alternatives that suit the same warmth tier ---------- */
const MODERN_STYLES = ["street","smart","dress"];
const A_ = (label, sex, family, spec) => ({label, sex, family, ...spec});
const IDEAS = [
  // 0: very cold
  [A_("Down jacket","all","modern",{top:"sweater",outer:"puffer",legs:"jeans",shoes:"sneaker"}),
   A_("Sweater + jacket","all","modern",{top:"sweater",outer:"jacket",legs:"trousers",shoes:"shoe",scarf:true}),
   A_("Kurta + pashmina","f","trad",{top:"kurta",outer:"shawl",legs:"suruwal",shoes:"shoe",scarf:true}),
   A_("Kurta + pashmina","m","trad",{top:"kurtaM",outer:"shawl",legs:"pyjama",shoes:"shoe",scarf:true}),
   A_("Sari + pashmina","f","trad",{top:"sari",outer:"shawl",legs:"sari",shoes:"shoe"}),
   A_("Daura, coat & topi","m","trad",{top:"daura",outer:"jacket",legs:"dauraSur",shoes:"shoe",topi:true}),
   A_("Bakhu","f","trad",{top:"bakhu",legs:"none",shoes:"boot"}),
   A_("Bakhu","m","trad",{top:"bakhuM",legs:"trousers",shoes:"boot"})],
  // 1: cold
  [A_("Leather jacket","all","modern",{top:"tee",outer:"leather",legs:"jeans",shoes:"boot"}),
   A_("Hoodie + denim jacket","all","modern",{top:"hoodie",outer:"denim",legs:"jeans",shoes:"sneaker"}),
   A_("Sweater + scarf","all","modern",{top:"sweater",legs:"jeans",shoes:"shoe",scarf:true}),
   A_("Kurta + shawl","f","trad",{top:"kurta",outer:"shawl",legs:"suruwal",shoes:"shoe"}),
   A_("Kurta + coat","m","trad",{top:"kurtaM",outer:"jacket",legs:"pyjama",shoes:"shoe"}),
   A_("Daura + istakot","m","trad",{top:"daura",outer:"istakot",legs:"dauraSur",shoes:"shoe",topi:true}),
   A_("Haku patasi + shawl","f","trad",{top:"patasi",outer:"shawl",legs:"patasi",shoes:"shoe"}),
   A_("Bakhu","f","trad",{top:"bakhu",legs:"none",shoes:"boot"}),
   A_("Bakhu","m","trad",{top:"bakhuM",legs:"trousers",shoes:"boot"})],
  // 2: cool
  [A_("Sweater","all","modern",{top:"sweater",legs:"jeans",shoes:"sneaker"}),
   A_("Cardigan over tee","all","modern",{top:"tee",outer:"cardigan",legs:"jeans",shoes:"sneaker"}),
   A_("Light jacket","all","modern",{top:"tee",outer:"jacket",legs:"chinos",shoes:"sneaker"}),
   A_("Flannel shirt","all","modern",{top:"flannel",legs:"jeans",shoes:"shoe"}),
   A_("Kurta + cardigan","f","trad",{top:"kurta",outer:"cardigan",legs:"suruwal",shoes:"shoe"}),
   A_("Sari + cardigan","f","trad",{top:"sari",outer:"cardigan",legs:"sari",shoes:"shoe"}),
   A_("Kurta + istakot","m","trad",{top:"kurtaM",outer:"istakot",legs:"pyjama",shoes:"shoe"}),
   A_("Daura + istakot","m","trad",{top:"daura",outer:"istakot",legs:"dauraSur",shoes:"shoe",topi:true})],
  // 3: mild
  [A_("Long sleeves","all","modern",{top:"sleeves",legs:"jeans",shoes:"sneaker"}),
   A_("Flannel shirt","all","modern",{top:"flannel",legs:"chinos",shoes:"shoe"}),
   A_("Light cardigan","all","modern",{top:"tee",outer:"cardigan",legs:"chinos",shoes:"shoe"}),
   A_("Kurti with jeans","f","modern",{top:"kurti",legs:"jeans",shoes:"sneaker"}),
   A_("Kurta-suruwal","f","trad",{top:"kurta",legs:"suruwal",shoes:"shoe"}),
   A_("Kurta-pyjama","m","trad",{top:"kurtaM",legs:"pyjama",shoes:"shoe"}),
   A_("Sari","f","trad",{top:"sari",legs:"sari",shoes:"shoe"}),
   A_("Daura-suruwal","m","trad",{top:"daura",legs:"dauraSur",shoes:"shoe",topi:true}),
   A_("Haku patasi","f","trad",{top:"patasi",legs:"patasi",shoes:"shoe"})],
  // 4: warm
  [A_("T-shirt","all","modern",{top:"tee",legs:"jeans",shoes:"sneaker"}),
   A_("Polo shirt","all","modern",{top:"polo",legs:"chinos",shoes:"shoe"}),
   A_("Short-sleeve shirt","all","modern",{top:"shirt",short:true,legs:"chinos",shoes:"shoe"}),
   A_("Dress","f","modern",{top:"dress",short:true,legs:"bare",shoes:"sandal"}),
   A_("Kurti with jeans","f","modern",{top:"kurti",short:true,legs:"jeans",shoes:"sandal"}),
   A_("Cotton kurta","f","trad",{top:"kurta",short:true,legs:"suruwal",shoes:"sandal"}),
   A_("Cotton kurta","m","trad",{top:"kurtaM",legs:"pyjama",shoes:"sandal"}),
   A_("Sari","f","trad",{top:"sari",legs:"sari",shoes:"sandal"}),
   A_("Gunyu-cholo","f","trad",{top:"cholo",rolled:true,legs:"gunyu",shoes:"sandal"}),
   A_("Daura-suruwal","m","trad",{top:"daura",legs:"dauraSur",shoes:"sandal",topi:true})],
  // 5: hot
  [A_("Linen shirt","all","modern",{top:"linen",rolled:true,legs:"chinos",shoes:"sandal",light:true}),
   A_("Tee & shorts","m","modern",{top:"tee",legs:"shorts",shoes:"sneaker",light:true}),
   A_("Tee & shorts","n","modern",{top:"tee",legs:"shorts",shoes:"sneaker",light:true}),
   A_("Tee & skirt","f","modern",{top:"tee",legs:"skirt",shoes:"sandal",light:true}),
   A_("Sundress","f","modern",{top:"sundress",short:true,legs:"bare",shoes:"sandal",light:true}),
   A_("Light cotton kurta","f","trad",{top:"kurta",short:true,legs:"suruwal",shoes:"sandal",light:true}),
   A_("Light cotton kurta","m","trad",{top:"kurtaM",legs:"pyjama",shoes:"sandal",light:true}),
   A_("Light cotton sari","f","trad",{top:"sari",legs:"sari",shoes:"sandal",light:true}),
   A_("Light daura","m","trad",{top:"daura",legs:"dauraSur",shoes:"sandal",topi:true,light:true})]
];
/* Up to 3 ideas: mostly from the person's own family of styles, plus one crossover. */
function ideasFor(o, P){
  const sex = P.sex === "f" ? "f" : P.sex === "m" ? "m" : "n";
  const fam = MODERN_STYLES.includes(P.style) ? "modern" : "trad";
  const primary = labelOf(o, P);
  const ok = x => x.sex === "all" || x.sex === sex || (sex === "n" && x.sex === "m" && !["daura","bakhuM"].includes(x.top));
  const pool = IDEAS[o.tier].filter(x => ok(x) && x.label !== primary);
  const seen = new Set();
  const uniq = pool.filter(x => !seen.has(x.label) && seen.add(x.label));
  const own = uniq.filter(x => x.family === fam).slice(0, 2);
  const cross = uniq.filter(x => x.family !== fam).slice(0, 1);
  return own.concat(cross);
}
/* ===================== Language: English / Roman Nepali (Kathmandu mix) ===================== */
const I18N = {
  en: {
    langBtn:"नेपाली", sub:"K Laune, Kathmandu", today:"Today", tomorrow:"Tomorrow", morning:"Morning", afternoon:"Afternoon", allDay:"All day",
    head2:(a,b) => `${a} in the morning, ${b} by afternoon.`, head1:a => `${a} works all day.`,
    ideas:t => t === "All day" ? "Other ideas for today" : `Other ideas for the ${t.toLowerCase()}`,
    feels:(t,ride) => `feels ${t}°${ride ? " on the ride" : ""}`,
    mask:"Wear a mask", maskSensitive:"Mask if sensitive", uvVeryHigh:"Cap & sunscreen", uvShade:"Shade or sunscreen", uvHigh:"Sunglasses or cap",
    windy:"Windy", muggy:"Muggy: breathable clothes", evening:x => `Out late? ${x} after 7 pm`,
    umbrella:"Umbrella", raincoat:"Raincoat", noRain:"No rain gear", rainPct:p => `${p}% rain`, gloves:"Gloves", layer:"Spare layer", water:"Water",
    about:"About this forecast",
    how:"Temperatures and rain come from a model trained on Kathmandu Airport records that corrects several global weather models. Mornings use the temperature around 8 am, afternoons the day's high; \"feels like\" includes humidity and wind, and riders get extra wind chill. UV, wind and air quality (AQI) come directly from Open-Meteo and CAMS.",
    conf:(m,lo,hi) => `Afternoon high most likely ${m}°; 8 in 10 days like this land between ${lo}° and ${hi}°.`,
    confFar:"Days further out are rougher estimates, usually within about 2° of the actual high.",
    updated:t => `Updated ${t}; a new forecast arrives every evening.`, data:"Weather data:",
    dark:"Switch to dark", light:"Switch to light", sample:"Sample data", me:"My avatar", chooseDay:"Choose a day",
    next:"Next", back:"Back", done:"Done", skip:"Skip",
    q:{age:"Who's heading out?", sex:"Dress them as", skin:"Pick a skin tone", hair:"Hair", style:"What do they usually wear?", ride:"How do they get around?", cold:"In the cold, they…"},
    facial:"Facial hair", suggested:" (suggested)",
    opt:{kid:"Kid", young:"Teen or young adult", adult:"Adult", elder:"Elder", m:"Man", f:"Woman", n:"Rather not say",
      walk:"Walk or bus", ride:"Scooter or bike", rideSub:"advice adds wind chill", cold0:"Feel it easily", cold1:"About average", cold2:"Rarely feel it",
      none:"None", mustache:"Mustache", beard:"Beard"},
    hair:{}, styleName:{}, styleSub:{}, label:{}
  },
  ne: {
    langBtn:"EN", sub:"काठमाडौं", today:"आज", tomorrow:"भोलि", morning:"बिहान", afternoon:"दिउँसो", allDay:"दिनभरि",
    head2:(a,b) => `बिहान ${a}, दिउँसो ${b}।`, head1:a => `दिनभरि ${a} ठिक हुन्छ।`,
    ideas:t => t === "दिनभरि" ? "आजका लागि अरू idea" : `${t}का लागि अरू idea`,
    feels:(t,ride) => `${ride ? "scooter मा " : ""}${t}° जस्तो लाग्छ`,
    mask:"Mask लगाउने", maskSensitive:"संवेदनशील भए mask", uvVeryHigh:"Cap र sunscreen", uvShade:"छाया वा sunscreen", uvHigh:"Sunglasses वा cap",
    windy:"हावा चल्छ", muggy:"उकुसमुकुस: हल्का कपडा", evening:x => `बेलुका ढिलो? ७ बजेपछि ${x}`,
    umbrella:"छाता बोक्ने", raincoat:"Raincoat लगाउने", noRain:"छाता चाहिँदैन", rainPct:p => `${p}% पानी`, gloves:"Gloves लगाउने", layer:"Extra layer बोक्ने", water:"पानी बोक्ने",
    about:"यो forecast बारे",
    how:"Temperature र पानी Kathmandu Airport को record मा train गरिएको model ले global weather model हरू सच्याएर दिन्छ। बिहानका लागि ८ बजेतिरको temperature, दिउँसोका लागि दिनको सबैभन्दा बढी; \"जस्तो लाग्छ\" मा humidity र हावा पनि जोडिन्छ, scooter चलाउनेलाई हावाको चिसो थप। UV, हावा र AQI सिधै Open-Meteo र CAMS बाट।",
    conf:(m,lo,hi) => `दिउँसो धेरैजसो ${m}°; यस्ता 10 दिनमा 8 दिन ${lo}° देखि ${hi}° बीच हुन्छ।`,
    confFar:"टाढाका दिनको forecast अलि अन्दाजी हो, actual भन्दा 2° जति फरक पर्न सक्छ।",
    updated:t => `Update: ${t}; हरेक साँझ नयाँ forecast आउँछ।`, data:"Weather data:",
    dark:"Dark mode", light:"Light mode", sample:"Sample data", me:"मेरो avatar", chooseDay:"दिन छान्नुहोस्",
    next:"अर्को", back:"पछाडि", done:"भयो", skip:"पछि गर्छु",
    q:{age:"कसका लागि?", sex:"केटा कि केटी?", skin:"छालाको रङ", hair:"कपाल", style:"प्रायः के लगाउनुहुन्छ?", ride:"कसरी आउजाउ गर्नुहुन्छ?", cold:"जाडो कत्तिको लाग्छ?"},
    facial:"जुँगा / दाह्री", suggested:" (सुझाव)",
    opt:{kid:"Kid", young:"Teen / youth", adult:"Adult", elder:"Senior", m:"केटा / पुरुष", f:"केटी / महिला", n:"भन्न चाहन्न",
      walk:"हिँडेर / bus मा", ride:"Scooter / bike मा", rideSub:"हावाको चिसो पनि हिसाब गर्छ", cold0:"छिट्टै जाडो लाग्छ", cold1:"Normal", cold2:"खासै जाडो लाग्दैन",
      none:"छैन", mustache:"जुँगा", beard:"दाह्री"},
    hair:{short:"छोटो", buzz:"Buzz cut", curly:"Curly", tied:"बाँधेको", long:"लामो", bun:"जुरो", braid:"चुल्ठी", ponytail:"Ponytail", bob:"Bob cut"},
    styleName:{street:"Streetwear", smart:"Smart casual", dress:"Dress", kurta:"कुर्ता", daura:"दौरा-सुरुवाल", sari:"साडी"},
    styleSub:{street:"T-shirt, hoodie, jeans", smart:"Shirt, blazer, pant", dress:"Dress, cardigan", kurta:"कुर्ता-सुरुवाल / पाइजामा", daura:"ढाका टोपीसँग", sari:"जाडोमा shawl सँग"},
    label:{
      "Puffer over hoodie":"Hoodie माथि puffer", "Bomber jacket":"Bomber jacket", "Hoodie":"Hoodie", "Denim jacket":"Denim jacket",
      "Graphic tee":"Graphic T-shirt", "Tee & shorts":"T-shirt र half pant", "Tee & skirt":"T-shirt र skirt",
      "Overcoat":"Overcoat", "Shirt & blazer":"Shirt र blazer", "Sweater over shirt":"Shirt माथि sweater", "Shirt":"Shirt",
      "Shirt, sleeves rolled":"Shirt, बाहुला मोडेर", "Polo shirt":"Polo T-shirt", "Long coat":"लामो coat", "Blazer":"Blazer", "Sweater":"Sweater",
      "Blouse":"Blouse", "Short-sleeve blouse":"छोटो बाहुलाको blouse", "Light blouse":"हल्का blouse",
      "Dress, coat & tights":"Dress, coat र stocking", "Dress & cardigan":"Dress र cardigan", "Long-sleeve dress":"लामो बाहुलाको dress", "Dress":"Dress", "Sundress":"हल्का sundress",
      "Puffer over kurta":"कुर्ता माथि puffer", "Kurta + coat":"कुर्ता र coat", "Kurta + istakot":"कुर्ता र इस्टकोट", "Kurta-pyjama":"कुर्ता-पाइजामा",
      "Cotton kurta":"सुती कुर्ता", "Kurta + shawl":"कुर्ता र सल", "Kurta + cardigan":"कुर्ता र cardigan", "Kurta-suruwal":"कुर्ता-सुरुवाल", "Light cotton kurta":"हल्का सुती कुर्ता",
      "Puffer over daura":"दौरा माथि puffer", "Daura + coat":"दौरा र coat", "Daura + istakot":"दौरा र इस्टकोट", "Daura-suruwal":"दौरा-सुरुवाल", "Light daura":"हल्का दौरा",
      "Puffer over sari":"साडी माथि puffer", "Sari + shawl":"साडी र सल", "Sari + cardigan":"साडी र cardigan", "Sari":"साडी", "Light cotton sari":"हल्का सुती साडी",
      "Down jacket":"Down jacket", "Sweater + jacket":"Sweater र jacket", "Kurta + pashmina":"कुर्ता र पश्मिना", "Sari + pashmina":"साडी र पश्मिना",
      "Daura, coat & topi":"दौरा, coat र ढाका टोपी", "Bakhu":"बख्खु", "Leather jacket":"Leather jacket", "Hoodie + denim jacket":"Hoodie र denim jacket",
      "Sweater + scarf":"Sweater र muffler", "Haku patasi + shawl":"हाकु पटासी र सल", "Haku patasi":"हाकु पटासी", "Cardigan over tee":"T-shirt माथि cardigan",
      "Light jacket":"हल्का jacket", "Flannel shirt":"Flannel shirt", "Long sleeves":"लामो बाहुला", "Light cardigan":"हल्का cardigan", "Kurti with jeans":"कुर्ती र jeans",
      "T-shirt":"T-shirt", "Short-sleeve shirt":"छोटो बाहुलाको shirt", "Gunyu-cholo":"गुन्यू-चोलो", "Linen shirt":"Linen shirt"
    },

    weekdays:["आइतबार","सोमबार","मंगलबार","बुधबार","बिहिबार","शुक्रबार","शनिबार"],
    months:["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
  }
};
let LANG = "en";
try { LANG = localStorage.getItem("klaune-lang") || ((navigator.language || "").startsWith("ne") ? "ne" : "en"); } catch(e) {}
const T = () => I18N[LANG];
const tLabel = s => (LANG === "ne" && T().label[s]) || s;
/* ===================== Forecast data ===================== */
const SAMPLE = {
  issued_for:"2026-10-12", generated_at:"2026-10-12T18:20:00+05:45", sample:true,
  days:[
    {date:"2026-10-13", horizon:1, model:"MOS v2", max_temp_c:26.8, min_temp_c:12.6, max_range_c:[25.2,28.5], min_range_c:[11.4,13.9], rain_prob:0.18, umbrella:false, layer:false,
     extras:{uv_max:8.6, gust_max:20, rh_mean:58, feels_max_adj:0.8, feels_min_adj:-1, morning_frac:0.15, evening_frac:0.35, aqi_max:128}},
    {date:"2026-10-14", horizon:2, model:"Champion v2 / Rain v1", max_temp_c:22.1, min_temp_c:14.4, rain_prob:0.74, umbrella:true, layer:false,
     extras:{uv_max:4, gust_max:38, rh_mean:86, feels_max_adj:-0.5, feels_min_adj:-0.5, morning_frac:0.2, evening_frac:0.45, aqi_max:55}},
    {date:"2026-10-15", horizon:3, model:"Champion v2 / Rain v1", max_temp_c:31.0, min_temp_c:18.6, rain_prob:0.08, umbrella:false, layer:false}
  ]
};
const r = Math.round;
const lc = s => /^T-shirt/.test(s) ? s : s[0].toLowerCase() + s.slice(1);

/* ===================== Profile ===================== */
const DEFAULT = {age:"adult", sex:"m", skin:2, hair:"short", facial:"none", style:"smart", ride:false, cold:1};
function normalise(p){
  p = {...DEFAULT, ...p};
  if (p.style === "everyday") p.style = defaultStyle(p.age, p.sex);            // profiles saved by older versions
  if (!stylesFor(p.sex).includes(p.style)) p.style = defaultStyle(p.age, p.sex);
  if (!HAIRS[p.sex].some(([v]) => v === p.hair)) p.hair = HAIRS[p.sex][0][0];
  if (p.sex !== "m" || p.age === "kid") p.facial = "none";
  return p;
}
let P = {...DEFAULT}, hasProfile = false;
try { const s = JSON.parse(localStorage.getItem("klaune-profile")); if (s) { P = normalise(s); hasProfile = true; } } catch(e) {}
const saveProfile = () => { try { localStorage.setItem("klaune-profile", JSON.stringify(P)); } catch(e) {} };

/* ===================== Main page ===================== */
const ICON = {
  umbrella:'<svg viewBox="0 0 24 24"><path d="M3 12a9 9 0 0 1 18 0H3z"/><path d="M12 12v7a2 2 0 0 1-4 0"/></svg>',
  coat:'<svg viewBox="0 0 24 24"><path d="M8 3h8l4 5-2 2v11H6V10L4 8l4-5z"/><path d="M12 3v18"/></svg>',
  dry:'<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M5.6 18.4L7 17M17 7l1.4-1.4"/></svg>',
  layer:'<svg viewBox="0 0 24 24"><path d="M12 4l9 5-9 5-9-5 9-5z"/><path d="M3 14l9 5 9-5"/></svg>',
  water:'<svg viewBox="0 0 24 24"><path d="M12 3c3 5 6 8 6 11a6 6 0 0 1-12 0c0-3 3-6 6-11z"/></svg>',
  mask:'<svg viewBox="0 0 24 24"><path d="M4 9c3-1 5-2 8-2s5 1 8 2v4c-2 4-5 5-8 5s-6-1-8-5V9z"/><path d="M4 10H2M20 10h2M8 11h8M8 14h8"/></svg>',
  wind:'<svg viewBox="0 0 24 24"><path d="M3 9h12a3 3 0 1 0-3-3M3 15h16a3 3 0 1 1-3 3M3 12h8"/></svg>',
  moon:'<svg viewBox="0 0 24 24"><path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/></svg>',
  gloves:'<svg viewBox="0 0 24 24"><path d="M7 21v-6L5 11V6a1.5 1.5 0 0 1 3 0v4V4a1.5 1.5 0 0 1 3 0v6V4.5a1.5 1.5 0 0 1 3 0V11V7a1.5 1.5 0 0 1 3 0v7l-2 7H7z"/></svg>'
};
const lab = o => tLabel(labelOf(o, P));
function headline(L){
  return L.length === 1 ? T().head1(LANG === "ne" ? lc(lab(L[0])) : lab(L[0]))
                        : T().head2(LANG === "ne" ? lc(lab(L[0])) : lab(L[0]), lc(lab(L[1])));
}
function chips(d, L){
  const pct = r(d.rain_prob * 100), out = [], t = T(), x = d.extras || {};
  const rp = `<span class="sub">${t.rainPct(pct)}</span>`;
  if (d.umbrella) out.push(P.ride ? `<span class="chip wet">${ICON.coat}${t.raincoat} ${rp}</span>`
                                   : `<span class="chip wet">${ICON.umbrella}${t.umbrella} ${rp}</span>`);
  else out.push(`<span class="chip">${ICON.dry}${t.noRain} ${rp}</span>`);
  if ((x.aqi_max ?? 0) >= 151) out.push(`<span class="chip warn">${ICON.mask}${t.mask} <span class="sub">AQI ${x.aqi_max}</span></span>`);
  else if ((x.aqi_max ?? 0) >= 101) out.push(`<span class="chip">${ICON.mask}${t.maskSensitive} <span class="sub">AQI ${x.aqi_max}</span></span>`);
  if (!d.umbrella && (x.uv_max ?? 0) >= 8) out.push(`<span class="chip">${ICON.dry}${MODERN_STYLES.includes(P.style) ? t.uvVeryHigh : t.uvShade} <span class="sub">UV ${r(x.uv_max)}</span></span>`);
  else if (!d.umbrella && (x.uv_max ?? 0) >= 6) out.push(`<span class="chip">${ICON.dry}${t.uvHigh} <span class="sub">UV ${r(x.uv_max)}</span></span>`);
  if ((x.gust_max ?? 0) >= 35) out.push(`<span class="chip">${ICON.wind}${t.windy} <span class="sub">${r(x.gust_max)} km/h</span></span>`);
  if ((x.rh_mean ?? 0) >= 75 && d.max_temp_c >= 26) out.push(`<span class="chip">${ICON.water}${t.muggy}</span>`);
  if (L.evening) out.push(`<span class="chip">${ICON.moon}${t.evening(lc(tLabel(wardrobe(P, L.evening.tier).label)))}</span>`);
  if (L.some(o => o.gloves)) out.push(`<span class="chip">${ICON.gloves}${t.gloves}</span>`);
  if (d.layer && d.max_temp_c < 25) out.push(`<span class="chip">${ICON.layer}${t.layer}</span>`);
  if (L.some(o => o.bottle)) out.push(`<span class="chip">${ICON.water}${t.water}</span>`);
  return out.join("");
}
function dayName(iso){
  const d = new Date(iso + "T12:00:00"), n = new Date(), t = T();
  const diff = Math.round((d - new Date(n.getFullYear(), n.getMonth(), n.getDate(), 12)) / 864e5);
  const wd = LANG === "ne" ? t.weekdays[d.getDay()] : d.toLocaleDateString("en-GB",{weekday:"short"});
  const date = LANG === "ne" ? `${wd}, ${d.getDate()} ${t.months[d.getMonth()]}` : d.toLocaleDateString("en-GB",{weekday:"short", day:"numeric", month:"short"});
  const short = LANG === "ne" ? wd.replace(/बार$/, "") : wd;
  return diff === 0 ? {s:t.today, l:`${t.today}, ${date}`} : diff === 1 ? {s:t.tomorrow, l:`${t.tomorrow}, ${date}`} : {s:short, l:date};
}
const mood = d => d.rain_prob >= .6 ? "rain" : ["cold","cold","mild","mild","warm","hot"][tierOf(d.max_temp_c)];

let FC, SEL = 0, FOCUS = null;
const OVR = {};                       // chosen alternative outfits, per "day:look"
function render(i){
  if (i !== SEL) FOCUS = null;
  SEL = i;
  const d = FC.days[i], L = looks(d, P);
  L.forEach((o, j) => { const k = `${i}:${j}:${o.tier}`; if (OVR[k]) o.outfit = OVR[k]; });
  if (FOCUS === null || FOCUS >= L.length) FOCUS = L.length - 1;
  document.body.dataset.mood = mood(d);
  document.getElementById("when").textContent = dayName(d.date).l;
  document.getElementById("headline").textContent = headline(L);
  const stage = document.getElementById("stage");
  stage.className = "stage" + (L.length > 1 ? " two" : "");
  stage.innerHTML = L.map((o, j) => {
    const temp = o.tRange ? `${r(o.tRange[0])}–${r(o.tRange[1])}°` : `${r(o.t)}°`;
    const feel = !o.tRange && Math.abs(o.feel - o.t) >= 1 ? `<div class="feel">${T().feels(r(o.feel), P.ride)}</div>` : "";
    const time = {Morning:T().morning, Afternoon:T().afternoon, "All day":T().allDay}[o.time];
    return (j ? '<span class="arrow" aria-hidden="true">→</span>' : "") +
      `<button class="look enter" type="button" data-j="${j}" aria-pressed="${j === FOCUS}">${avatar(o, P)}<div class="time">${time}</div><div class="temp">${temp}</div>${feel}<div class="wear">${lab(o)}</div></button>`;
  }).join("");
  drawIdeas(L);
  document.getElementById("chips").innerHTML = chips(d, L);
  document.querySelectorAll(".day").forEach((b, j) => b.setAttribute("aria-pressed", j === i));
  document.getElementById("conf").textContent = d.max_range_c
    ? T().conf(r(d.max_temp_c), r(d.max_range_c[0]), r(d.max_range_c[1])) : T().confFar;
  meButton(); staticText();
}
function drawIdeas(L){
  const o = L[FOCUS], base = {...o, outfit:undefined};
  const own = {...wardrobe(P, o.tier), family: MODERN_STYLES.includes(P.style) ? "modern" : "trad"};
  const opts = [own, ...ideasFor(base, P)];
  const current = labelOf(o, P);
  const time = {Morning:T().morning, Afternoon:T().afternoon, "All day":T().allDay}[o.time];
  document.getElementById("ideas").innerHTML = `<p>${T().ideas(time)}</p><div class="idea-row">` + opts.map((x, n) =>
    `<button class="idea" type="button" data-n="${n}" aria-pressed="${x.label === current}">${avatar({...base, outfit:x, umbrella:null, rain:false, helmet:false, bottle:false}, P)}<span>${tLabel(x.label)}</span></button>`).join("") + `</div>`;
  document.getElementById("ideas").onclick = e => {
    const b = e.target.closest(".idea"); if (!b) return;
    const x = opts[+b.dataset.n], k = `${SEL}:${FOCUS}:${o.tier}`;
    if (+b.dataset.n === 0) delete OVR[k]; else OVR[k] = x;
    render(SEL);
  };
}
function meButton(){
  // tiny head-and-shoulders crop of the user's avatar
  const svg = avatar({tier:3, time:"You"}, {...P, age: P.age === "kid" ? "adult" : P.age}).replace(/viewBox="[^"]+"/, 'viewBox="30 6 60 60"');
  document.getElementById("me").innerHTML = svg + T().me;
}
function drawDays(){
  document.getElementById("days").style.gridTemplateColumns = `repeat(${FC.days.length}, 1fr)`;
  document.getElementById("days").innerHTML = FC.days.map((d, j) =>
    `<button class="day" type="button" aria-pressed="${j === SEL}" data-i="${j}"><span class="d">${dayName(d.date).s}</span>` +
    `<span class="t">${r(d.max_temp_c)}°</span>${d.umbrella ? '<span class="u" aria-hidden="true">☂</span>' : ""}</button>`).join("");
}
function staticText(){
  const t = T(), g = FC && FC.generated_at ? new Date(FC.generated_at) : null;
  document.documentElement.lang = LANG === "ne" ? "ne" : "en";
  document.getElementById("t-sub").textContent = t.sub;
  document.getElementById("lang").textContent = t.langBtn;
  document.getElementById("t-about").textContent = t.about;
  document.getElementById("t-how").textContent = t.how;
  document.getElementById("t-data").textContent = t.data;
  document.getElementById("days").setAttribute("aria-label", t.chooseDay);
  document.getElementById("badge").innerHTML = FC && FC.sample ? `<span class="demo">${t.sample}</span>` : "";
  document.getElementById("updated").textContent = g ? t.updated(g.toLocaleString("en-GB",{weekday:"short", hour:"2-digit", minute:"2-digit"})) : "";
  document.getElementById("theme").textContent = document.documentElement.dataset.theme === "dark" ? t.light : t.dark;
  document.getElementById("skip").textContent = t.skip;
  document.getElementById("back").textContent = t.back;
}
function start(fc){
  FC = fc;
  document.getElementById("stage").addEventListener("click", e => {
    const b = e.target.closest(".look"); if (!b) return;
    FOCUS = +b.dataset.j; render(SEL);
  });
  // Hide days already past; in the daytime open on today, from 5 pm on tomorrow.
  const todayISO = new Date().toLocaleDateString("en-CA");
  const upcoming = FC.days.filter(d => d.date >= todayISO);
  if (upcoming.length) FC.days = upcoming;
  const evening = new Date().getHours() >= 17;
  let first = FC.days.findIndex(d => evening ? d.date > todayISO : d.date >= todayISO);
  if (first < 0) first = 0;
  SEL = first; drawDays();
  document.getElementById("days").addEventListener("click", e => { const b = e.target.closest(".day"); if (b) render(+b.dataset.i); });
  render(first);
  if (!hasProfile) openSheet();
}

/* ===================== Make-your-avatar questionnaire ===================== */
const O = k => T().opt[k];
const STEPS = [
  {q:"age", groups: () => [{key:"age", opts:["kid","young","adult","elder"].map(k => [k, O(k)])}]},
  {q:"sex", groups: () => [{key:"sex", opts:["m","f","n"].map(k => [k, O(k)])}]},
  {q:"skin", swatches:true},
  {q:"hair", groups: d => [{key:"hair", opts:HAIRS[d.sex].map(([v, en]) => [v, T().hair[v] || en])}].concat(d.sex === "m" && d.age !== "kid"
      ? [{key:"facial", label:T().facial, opts:["none","mustache","beard"].map(k => [k, O(k)])}] : [])},
  {q:"style", groups: d => [{key:"style", opts:stylesFor(d.sex).map(k => [k, T().styleName[k] || STYLE_NAMES[k],
      (T().styleSub[k] || STYLE_SUB[k]) + (k === defaultStyle(d.age, d.sex) ? T().suggested : "")])}]},
  {q:"ride", groups: () => [{key:"ride", opts:[[false, O("walk")], [true, O("ride"), O("rideSub")]]}]},
  {q:"cold", groups: () => [{key:"cold", opts:[[0, O("cold0")], [1, O("cold1")], [2, O("cold2")]]}]}
];
let step = 0, draft = {...P}, stylePicked = false;
const sheet = document.getElementById("sheet");
function openSheet(){ draft = {...P}; step = 0; stylePicked = false; sheet.classList.add("open"); drawStep(); document.getElementById("next").focus(); }
function closeSheet(save){
  if (save) { P = normalise(draft); hasProfile = true; saveProfile(); }
  else if (!hasProfile) { hasProfile = true; saveProfile(); }
  sheet.classList.remove("open"); render(SEL); document.getElementById("me").focus();
}
function drawStep(){
  const S = STEPS[step], d = FC ? FC.days[SEL] : SAMPLE.days[0];
  draft = normalise(draft);
  document.getElementById("dots").innerHTML = STEPS.map((_, i) => `<i class="${i <= step ? "on" : ""}"></i>`).join("");
  const L = looks(d, draft);
  document.getElementById("preview").innerHTML = avatar(L[L.length - 1], draft);
  document.getElementById("q").textContent = T().q[S.q];
  const A = document.getElementById("answers");
  A.innerHTML = S.swatches
    ? `<div class="swatches">${SKINS.map((c, i) => `<button class="sw" type="button" data-k="skin" data-v="${i}" style="background:${c}" aria-label="Skin tone ${i + 1}" aria-pressed="${draft.skin === i}"></button>`).join("")}</div>`
    : S.groups(draft).map(g => (g.label ? `<p class="glabel">${g.label}</p>` : "") +
        `<div class="opts">${g.opts.map(([v, t, sub]) => `<button class="opt" type="button" data-k="${g.key}" data-v='${JSON.stringify(v)}' aria-pressed="${draft[g.key] === v}">${t}${sub ? `<small>${sub}</small>` : ""}</button>`).join("")}</div>`).join("");
  document.getElementById("back").style.visibility = step ? "visible" : "hidden";
  document.getElementById("next").textContent = step === STEPS.length - 1 ? T().done : T().next;
}
document.getElementById("answers").addEventListener("click", e => {
  const b = e.target.closest("[data-v]"); if (!b) return;
  const k = b.dataset.k, v = JSON.parse(b.dataset.v);
  draft[k] = v;
  if (k === "style") stylePicked = true;
  if ((k === "age" || k === "sex") && !stylePicked) draft.style = defaultStyle(draft.age, draft.sex);
  drawStep();
});
document.getElementById("next").addEventListener("click", () => step < STEPS.length - 1 ? (step++, drawStep()) : closeSheet(true));
document.getElementById("back").addEventListener("click", () => { if (step) { step--; drawStep(); } });
document.getElementById("skip").addEventListener("click", () => closeSheet(false));
document.getElementById("me").addEventListener("click", openSheet);
document.addEventListener("keydown", e => { if (e.key === "Escape" && sheet.classList.contains("open")) closeSheet(false); });

/* ===================== Boot + theme ===================== */
fetch("forecast.json", {cache:"no-store"}).then(x => x.ok ? x.json() : Promise.reject()).then(start).catch(() => start(SAMPLE));
const root = document.documentElement, tb = document.getElementById("theme");
function setTheme(t){ root.dataset.theme = t; tb.textContent = t === "dark" ? T().light : T().dark;
  try{ localStorage.setItem("klaune-theme", t); }catch(e){} }
let saved = null; try{ saved = localStorage.getItem("klaune-theme"); }catch(e){}
setTheme(saved || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
tb.addEventListener("click", () => setTheme(root.dataset.theme === "dark" ? "light" : "dark"));

document.getElementById("lang").addEventListener("click", () => {
  LANG = LANG === "en" ? "ne" : "en";
  try { localStorage.setItem("klaune-lang", LANG); } catch(e) {}
  if (FC) { drawDays(); render(SEL); }
  if (sheet.classList.contains("open")) drawStep();
});

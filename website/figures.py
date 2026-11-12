"""Draw the testbed network for the site: python website/figures.py (writes website/_static/testbed.html)."""
from pathlib import Path
from types import SimpleNamespace

from agmsim.montreal import NODE, NODES, ROADS, build_montreal

W, H, PAD = 680, 560, 50
STYLE = {"autoroute": ("#378ADD", 3), "bridge": ("#D85A30", 3.5), "small_bridge": ("#D85A30", 2.5), "arterial": ("#888780", 1.3)}
SHIFT = {"mcgill": (-7, -5, "end"), "downtown": (-7, 14, "end"), "old_port": (8, 14, "start"), "berri": (8, -4, "start"),
         "plateau": (-7, -6, "end"), "centre_sud": (8, -6, "start"), "cdn": (0, 17, "middle"), "verdun": (0, 18, "middle"),
         "turcot": (-8, 14, "end"), "st_lambert": (9, 4, "start"), "longueuil": (9, 4, "start"), "villeray": (-8, -6, "end"),
         "rosemont": (8, 4, "start"), "hochelaga": (9, 4, "start"), "st_michel": (8, -6, "start"), "brossard": (0, 18, "middle"),
         "dorval": (0, 18, "middle"), "st_laurent": (7, 18, "start"), "mercier": (9, 4, "start"), "ahuntsic": (-8, 14, "end"),
         "decarie": (-8, -6, "end")}


def main() -> None:
    net = build_montreal(SimpleNamespace())
    xs, ys = [x for x, _ in net.nodes.values()], [y for _, y in net.nodes.values()]
    scale = min((W - 2 * PAD) / (max(xs) - min(xs)), (H - 2 * PAD - 24) / (max(ys) - min(ys)))
    left = (W - scale * (max(xs) - min(xs))) / 2

    def at(node):
        x, y = net.nodes[node]
        return round(left + (x - min(xs)) * scale), round(H - PAD - (y - min(ys)) * scale)

    out = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The simplified Montreal '
           f'network: {len(NODES)} nodes and {len(ROADS)} roads"><title>The simplified Montreal network</title>']
    for kind in ("arterial", "autoroute", "small_bridge", "bridge"):
        colour, width = STYLE[kind]
        path = "".join("M{} {}L{} {}".format(*at(NODE[a]), *at(NODE[b])) for a, b, k, _ in ROADS if k == kind)
        out.append(f'<path d="{path}" stroke="{colour}" stroke-width="{width}" stroke-linecap="round" fill="none"/>')
    for node, (short, name, *_rest) in enumerate(NODES):
        x, y = at(node)
        dx, dy, anchor = SHIFT.get(short, (0, -9, "middle"))
        out.append(f'<circle cx="{x}" cy="{y}" r="3.5" stroke-width="1.2"/>'
                   f'<text x="{x + dx}" y="{y + dy}" text-anchor="{anchor}" font-size="12">{name}</text>')
    out.append('<g transform="translate(50,18)" font-size="12"><path d="M0 0L22 0" stroke="#378ADD" stroke-width="3"/>'
               '<text x="28" y="4">expressway</text><path d="M110 0L132 0" stroke="#D85A30" stroke-width="3.5"/>'
               '<text x="138" y="4">bridge or tunnel</text><path d="M250 0L272 0" stroke="#888780" stroke-width="1.3"/>'
               '<text x="278" y="4">arterial</text></g></svg>')
    Path(__file__).parent.joinpath("_static", "testbed.html").write_text('<div class="testbed-figure">' + "".join(out) + "</div>\n")


if __name__ == "__main__":
    main()

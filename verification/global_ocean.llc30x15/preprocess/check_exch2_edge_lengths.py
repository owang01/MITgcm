#!/usr/bin/env python3
"""
Hand-port of the edge-length consistency check in
pkg/exch2/w2_set_f2f_index.F (lines ~97-118), run against the LLC30
dimsFacets/facetEdgeLink values destined for
verification/global_ocean.llc30x15/input/data.exch2.

Why this exists (Task 3 of docs/plans/20261103-llc30-lightweight-verification-
config.md): before writing SIZE.h/data.exch2, independently verify that the
scaled-down dimsFacets is self-consistent with the (resolution-independent,
copied-unchanged) facetEdgeLink table. An earlier draft of this plan got the
facet 4/5 orientation backwards (30,90 instead of 90,30) and this exact check
is what would have (and in review, did) catch it.

Fortran semantics being reproduced (1-indexed in the source; this script
keeps 1-indexed facet numbers but 0-indexed Python lists for dims/link):

  facet_link(i,j) = XX.d  encodes: edge i of facet j connects to edge d of
  facet XX, where i,d in {1,2,3,4} <=> {North,South,East,West}.

  lo = 2*(j-1) + (i+1)/2     -- integer division, i.e. Fortran INT((i+1)/2)
       facet_dims(lo)
  This indexes into facet_dims, the flattened dimsFacets(1:2*nFacets) array
  (facet_dims(2*j-1)=x-dim of facet j, facet_dims(2*j)=y-dim of facet j).
  For i=1 (N) or i=2 (S): (i+1)/2 = 1 (integer div) -> lo = 2*(j-1)+1 = x-dim.
  For i=3 (E) or i=4 (W): (i+1)/2 = 2            -> lo = 2*(j-1)+2 = y-dim.
  i.e. N/S edge length = facet's x-dim; E/W edge length = facet's y-dim.

  Then for the linked facet jj / edge ii, the same lo-formula (with jj,ii)
  gives ll, and the check asserts lo == ll for every (i,j) with a connection
  (facet_link(i,j) != 0).
"""

# LLC30 dimsFacets(1:10), facet j's dims are (dimsFacets[2*(j-1)], dimsFacets[2*(j-1)+1])
# order: f1x,f1y, f2x,f2y, f3x,f3y, f4x,f4y, f5x,f5y
dimsFacets = [30, 90, 30, 90, 30, 30, 90, 30, 90, 30]

# facetEdgeLink(1:4, j): edge order (N,S,E,W) = (1,2,3,4), 1-indexed facet j
# copied byte-for-byte from the LLC90 reference (resolution-independent)
facetEdgeLink = {
    1: ['3.4', '0.', '2.4', '5.1'],
    2: ['3.2', '0.', '4.2', '1.3'],
    3: ['5.4', '2.1', '4.4', '1.1'],
    4: ['5.2', '2.3', '0.', '3.3'],
    5: ['1.4', '4.1', '0.', '3.1'],
}

edge_name = {1: 'N', 2: 'S', 3: 'E', 4: 'W'}
nFacets = 5


def facet_dims(lo_1indexed):
    """facet_dims(lo) with lo 1-indexed into the flat dimsFacets array."""
    return dimsFacets[lo_1indexed - 1]


def edge_len(i, j):
    """Length of edge i (1=N,2=S,3=E,4=W) of facet j (1-indexed)."""
    lo = 2 * (j - 1) + (i + 1) // 2
    return facet_dims(lo)


def parse_link(token):
    """Parse a facetEdgeLink token like '3.4' -> (jj=3, ii=4). '0.' -> (0, 0)."""
    token = token.strip()
    if token in ('0.', '0'):
        return 0, 0
    whole, frac = token.split('.')
    jj = int(whole)
    ii = int(frac)
    return jj, ii


def main():
    errors = []
    checked = 0
    for j in range(1, nFacets + 1):
        for i in range(1, 5):
            tok = facetEdgeLink[j][i - 1]
            jj, ii = parse_link(tok)
            if jj == 0:
                # disconnected edge (only facet 3's W edge / facet 4,5's E edge
                # in the 5-facet form) -- nothing to check
                continue
            lo = edge_len(i, j)
            ll = edge_len(ii, jj)
            checked += 1
            status = "OK" if lo == ll else "MISMATCH"
            print(f"facet {j} {edge_name[i]}-edge (len={lo:3d})  <->  "
                  f"facet {jj} {edge_name[ii]}-edge (len={ll:3d})  [{status}]")
            if lo != ll:
                errors.append((j, i, jj, ii, lo, ll))

    print()
    print(f"Checked {checked} connected edges across {nFacets} facets.")
    if errors:
        print(f"FAIL: {len(errors)} mismatched edge(s):")
        for (j, i, jj, ii, lo, ll) in errors:
            print(f"  facet {j} {edge_name[i]} (len={lo}) != "
                  f"facet {jj} {edge_name[ii]} (len={ll})")
        raise SystemExit(1)
    else:
        print("PASS: all connected edge lengths are mutually consistent.")


if __name__ == '__main__':
    main()

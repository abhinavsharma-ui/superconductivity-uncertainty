"""CC's half of the split tol=1e-4 rebuild: db positions 674-805.

Two points of care.

The short-circuit in build_physics_dataset -- Tc_ME := 0 whenever
Tc_AD < SOLVER_FLOOR_K -- is DELIBERATELY ABSENT here. It is the defect this
rebuild exists to test, so this script solves every material including those
with Tc_AD = 0, which is also why this half is the expensive one: 63.6 s per
material in that band against 0.09 s at the hot end.

Materials are selected by KEY, not by position. The split was communicated as a
position range, but positions are an index into an ordering neither machine has
verified the other reproduces -- key 656, 701, 716, 739, 750, 786, 805 are
absent from the sequence, so position and key drift apart by a variable offset.
Selecting on the key removes the assumption.

Positions 674-676 (keys 686, 687, 688) are computed by both machines. They are
the merge test: two of the three sit in the near-zero-Tc band on purpose,
because that is where a platform difference in the bisection would surface and
it is the band the committed build cannot see. If they disagree beyond solver
tolerance the halves are separate datasets, not two parts of one.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

from build_physics_dataset import (_be_polite, RAW, MU_STAR_AD, MU_STAR_ME,
                                   MU_STAR_AD_ALT, MU_STAR_ME_ALT, CUTOFF_FACTOR,
                                   SOLVER_FLOOR_K, MAX_MATSUBARA, SOLVER_TOL)
from eliashberg import a2f_moments, allen_dynes_tc, eliashberg_tc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "rebuild_cc.csv")

KEYS = """686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 702 703 704
705 706 707 708 709 710 711 712 713 714 715 717 718 719 720 721 722 723 724 725
726 727 728 729 730 731 732 733 734 735 736 737 738 740 741 742 743 744 745 746
747 748 749 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767
768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 787 788
789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 806 807 808 809
810 811 812 813 814 815 816 817 818 819 820 821 822 823""".split()
OVERLAP = {"686", "687", "688"}


def _one(item):
    _be_polite()
    key, formula, source, w_list, a_list = item
    t0 = time.time()
    w = np.asarray(w_list, float)
    a = np.asarray(a_list, float)
    m = a2f_moments(w, a)
    lam, w_log, w_2, w_max = m["lambda_"], m["w_log"], m["w_2"], m["w_max"]

    tc_ad = allen_dynes_tc(lam, w_log, w_2, MU_STAR_AD, corrections=True)
    tc_ad13 = allen_dynes_tc(lam, w_log, w_2, MU_STAR_AD_ALT, corrections=True)
    # NO short-circuit on tc_ad: that bound is what is under test.
    solve = lambda mu, guess: eliashberg_tc(
        w, a, mu_star=mu, cutoff_factor=CUTOFF_FACTOR,
        t_guess=guess if guess > SOLVER_FLOOR_K else None,
        t_floor=SOLVER_FLOOR_K, max_matsubara=MAX_MATSUBARA, tol=SOLVER_TOL)
    tc_me = solve(MU_STAR_ME, tc_ad)
    tc_me13 = solve(MU_STAR_ME_ALT, tc_ad13)
    return dict(key=str(key), formula=formula, source_name=source, lam=lam,
                w_log=w_log, w_2=w_2, w_ratio=w_2 / w_log, w_max=w_max,
                Tc_AD=tc_ad, Tc_AD13=tc_ad13, Tc_ME=tc_me, Tc_ME13=tc_me13,
                secs=time.time() - t0)


def main(workers):
    _be_polite()
    with open(RAW) as fh:
        db = json.load(fh)
    have = set(db["lambda"].keys())
    missing = [k for k in KEYS if k not in have]
    assert not missing, f"keys absent from the database: {missing}"
    print(f"{len(KEYS)} keys, all present; {len(OVERLAP)} of them are the overlap")

    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT, dtype={"key": str}).key)
        print(f"resuming: {len(done)} already on disk")

    work = [(k, db["comp"][k], db.get("source_name", {}).get(k, ""),
             db["Freq_meV"][k], db["a2F"][k]) for k in KEYS if k not in done]
    if not work:
        print("nothing to do")
        return summarise()

    t0, buf, n = time.time(), [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for rec in ex.map(_one, work, chunksize=1):
            buf.append(rec)
            n += 1
            if len(buf) >= 5 or n == len(work):
                pd.DataFrame(buf).to_csv(OUT, mode="a", index=False,
                                         header=not os.path.exists(OUT))
                buf = []
            if n % 10 == 0 or n == len(work):
                el = time.time() - t0
                print(f"  {n}/{len(work)}  {el/60:.1f} min  "
                      f"eta {el/n*(len(work)-n)/60:.0f} min", flush=True)
    summarise()


def summarise():
    d = pd.read_csv(OUT, dtype={"key": str})
    print(f"\n=== {len(d)} rows, {d.secs.sum()/60:.1f} min of solver time ===")
    print("\nTHE OVERLAP ROWS -- these are the merge test:")
    o = d[d.key.isin(OVERLAP)]
    print(o[["key", "formula", "Tc_AD", "Tc_ME", "Tc_ME13", "secs"]]
          .to_string(index=False))
    bad = d[(d.Tc_AD < SOLVER_FLOOR_K) & (d.Tc_ME > 0.05)]
    print(f"\ndefect check -- Tc_AD < {SOLVER_FLOOR_K} but Tc_ME > 0.05 K: "
          f"{len(bad)} hits")
    if len(bad):
        print(bad[["key", "formula", "lam", "Tc_AD", "Tc_ME"]].to_string(index=False))
    n_lo = (d.Tc_AD < SOLVER_FLOOR_K).sum()
    print(f"  (of {n_lo} rows in this half that the committed build skips entirely)")


if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a.add_argument("--summarise", action="store_true")
    ns = a.parse_args()
    summarise() if ns.summarise else main(ns.workers)

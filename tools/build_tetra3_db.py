"""
Generate a tetra3 star-pattern database sized to a camera's field of view.

tetra3's generate_database() reads a raw Hipparcos catalogue file (hip_main.dat) that
must live in the tetra3 package directory. Skyfield downloads that same file (to the
project directory) the first time the star catalogue is used, so this helper copies it
into place if needed, then builds and saves the .npz database into tetra3's data dir.

Usage (run in the conda 'track' env):
    python tools/build_tetra3_db.py --max-fov 12 --min-fov 4 --mag 7 --name db_fov12_mag7

The database is then available to PlateSolver via load_database='<name>'. Building takes
a couple of minutes; call build_database() from a worker thread in the UI with a
status_callback so the app stays responsive.
"""

import argparse
import os
import shutil


def ensure_catalog(catalog='hip_main', status_callback=None):
    """Make sure the named catalogue (hip_main / tyc_main) is in the tetra3 dir.

    tetra3 reads `<catalog>.dat` from its own package directory. We copy from the
    project directory (where skyfield/CDS downloads land) or, for Hipparcos, fetch it.
    """
    import tetra3
    t3dir = os.path.dirname(tetra3.__file__)
    fname = catalog + '.dat'
    dst = os.path.join(t3dir, fname)
    if os.path.exists(dst):
        return dst

    src = os.path.join(os.getcwd(), fname)
    if os.path.exists(src):
        if status_callback:
            status_callback(f"Copying {fname} to tetra3 dir...")
        shutil.copyfile(src, dst)
        return dst

    if catalog == 'hip_main':
        from skyfield.api import load
        from skyfield.data import hipparcos
        if status_callback:
            status_callback("Downloading Hipparcos catalogue (one-time)...")
        with load.open(hipparcos.URL):
            pass
        src = os.path.join(os.getcwd(), os.path.basename(hipparcos.URL))
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            return dst
    raise FileNotFoundError(
        f"Could not locate {fname}. For tyc_main, download it from "
        "https://cdsarc.cds.unistra.fr/ftp/cats/I/239/tyc_main.dat into the project dir.")


# Back-compat alias.
def ensure_hip_catalog(status_callback=None):
    return ensure_catalog('hip_main', status_callback)


def build_database(max_fov, min_fov=None, star_max_magnitude=7, name="db_custom",
                   catalog='hip_main', status_callback=None, **gen_kwargs):
    """Generate and save a tetra3 database. Returns the database name.

    Extra keyword args (e.g. simplify_pattern=True, pattern_stars_per_fov,
    verification_stars_per_fov) pass straight to tetra3.generate_database -- essential
    for sub-degree FOVs, where simplify_pattern + a bounded pattern density keep the
    build from exploding into many hours / multi-GB databases.
    """
    import tetra3
    ensure_catalog(catalog, status_callback)
    if status_callback:
        status_callback(f"Generating tetra3 DB {name} ({catalog}, fov {min_fov}-{max_fov}, "
                        f"mag<={star_max_magnitude}, {gen_kwargs})...")
    t3 = tetra3.Tetra3(load_database=None)
    t3.generate_database(max_fov=max_fov, min_fov=min_fov, star_catalog=catalog,
                         star_max_magnitude=star_max_magnitude, save_as=name, **gen_kwargs)
    if status_callback:
        status_callback(f"tetra3 DB {name} ready.")
    return name


def main():
    ap = argparse.ArgumentParser(description="Build a tetra3 plate-solving database.")
    ap.add_argument('--max-fov', type=float, required=True, help="Largest FOV (deg) to index.")
    ap.add_argument('--min-fov', type=float, default=None, help="Smallest FOV (deg) to index.")
    ap.add_argument('--mag', type=float, default=7.0, help="Faintest star magnitude.")
    ap.add_argument('--name', type=str, default='db_custom', help="Database name (.npz).")
    ap.add_argument('--catalog', type=str, default='hip_main',
                    choices=['hip_main', 'tyc_main', 'bsc5'], help="Star catalogue to index.")
    args = ap.parse_args()
    name = build_database(args.max_fov, args.min_fov, args.mag, args.name,
                          catalog=args.catalog, status_callback=lambda m: print(m))
    print(f"Done: {name}")


if __name__ == '__main__':
    main()

"""
Push a generate_trials.py manifest (one row per trial) into a Google Sheet,
so you can browse/filter/sort trial outcomes without touching Python.

Only the flat, per-trial summary fields are exported (object, squeeze
intensity, pull multiple, slip label, onset step, displacement, etc.) -- the
full per-timestep sensor traces live in each trial's .npz and are far too
large/dense to be useful as spreadsheet rows. Use export_data.py for that.

Writes into an EXISTING spreadsheet you already have open (--sheet-url), into
a worksheet named "trials" -- created if missing, overwritten if present. If
you'd rather have the script create a brand-new spreadsheet instead, omit
--sheet-url and pass --title.

ONE-TIME SETUP (Google Cloud side -- this can't be automated for you):
  1. https://console.cloud.google.com/ -> create or select a project.
  2. APIs & Services > Library -> enable BOTH "Google Sheets API" and
     "Google Drive API" (Drive API is needed for gspread to look up a
     spreadsheet by URL, and to create one if --sheet-url is omitted).
  3. APIs & Services > OAuth consent screen -> External -> fill in an app
     name + your email -> add yourself under "Test users" (skip this if
     your Google Cloud project belongs to a Workspace org you administer).
  4. APIs & Services > Credentials > Create Credentials > OAuth client ID ->
     Application type "Desktop app" -> Create -> Download JSON.
  5. Save that file as:
         %APPDATA%\\gspread\\credentials.json
     (gspread's default lookup path on Windows), or pass --credentials to
     point at it wherever you saved it instead. Do NOT commit this file to
     git -- it's a real OAuth client secret.

pip install gspread google-auth google-auth-oauthlib   (already installed
in venv/ as of this script being added)

The first run opens a browser for you to sign in and approve access;
gspread then caches the resulting token (also under %APPDATA%\\gspread\\) so
you won't be prompted again on this machine. You must have edit access to
the sheet at --sheet-url under the Google account you sign in with -- if
someone else owns it, they need to have shared it with you first.

Usage:
    # write into a sheet you already created and are looking at
    python export_to_sheets.py \\
        --manifest data/raw_trials/manifest_balanced.json \\
        --sheet-url "https://docs.google.com/spreadsheets/d/XXXXXXX/edit"

    # or let the script create a brand-new one
    python export_to_sheets.py \\
        --manifest data/raw_trials/manifest_balanced.json \\
        --title "PI-TCN Slip Trials"
"""
import argparse
import json

import gspread


# Only flat, scalar fields -- object_spec/object_actual are nested dicts
# (full per-trial specs, for exact replay) and don't belong in a spreadsheet
# row. Read them straight from the manifest JSON if you need that detail.
MANIFEST_COLUMNS = [
    "trial_id",
    "object_name",
    "object_kind",
    "slip_label",
    "onset_step",
    "max_displacement",
    "effective_threshold",
    "squeeze_intensity",
    "pull_multiple",
    "pull_force_max",
    "grasp_force_at_settle",
    "obj_mass",
    "object_size_jitter",
    "object_friction_sliding",
    "object_friction_torsional",
    "object_friction_rolling",
    "pad_friction_mean",
    "pad_friction_min",
    "object_yaw_deg",
    "object_roll_deg",
    "object_pitch_deg",
    "sensor_noise_std",
    "ramp_steps",
    "ramp_duration_s",
    "approach_velocity_rad_s",
    "ever_contacted",
    "n_fingers_contacted",
    "first_contact_step",
    "data_path",
]


def manifest_to_rows(manifest):
    rows = [MANIFEST_COLUMNS]
    for m in manifest:
        rows.append([m.get(col, "") for col in MANIFEST_COLUMNS])
    return rows


WORKSHEET_NAME = "trials"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="Path to manifest_balanced.json or manifest_full.json "
                         "from generate_trials.py")
    ap.add_argument(
        "--sheet-url",
        default=None,
        # FIX: this used to default to a specific real, pre-existing
        # spreadsheet URL -- running the script exactly as its own usage
        # example shows (--manifest only) would silently clear and overwrite
        # that sheet's "trials" worksheet. No safe default exists for "which
        # real spreadsheet to overwrite", so there isn't one now: omitting
        # --sheet-url creates a brand-new spreadsheet instead (harmless,
        # reversible), and overwriting an existing one requires explicitly
        # naming it.
        help="URL (or bare key) of an EXISTING Google Sheet to write into. "
             "Its 'trials' worksheet is created if missing, or cleared and "
             "overwritten if present. Omit this (the default) to instead "
             "create a brand-new spreadsheet (see --title).")
    ap.add_argument("--title", default="PI-TCN Slip Trials",
                    help="Name for a newly-created spreadsheet. Ignored if "
                         "--sheet-url is given.")
    ap.add_argument("--credentials", default=None,
                    help="Path to the OAuth client JSON from Google Cloud "
                         "Console. Defaults to gspread's own lookup path "
                         "(%%APPDATA%%\\gspread\\credentials.json on Windows) "
                         "if omitted.")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    if not manifest:
        raise ValueError(f"{args.manifest} has no trials -- nothing to export.")

    gc = (gspread.oauth(credentials_filename=args.credentials)
          if args.credentials else gspread.oauth())

    if args.sheet_url:
        sh = gc.open_by_url(args.sheet_url)
        try:
            ws = sh.worksheet(WORKSHEET_NAME)
            ws.clear()
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=WORKSHEET_NAME, rows=1, cols=1)
    else:
        sh = gc.create(args.title)
        ws = sh.sheet1
        ws.update_title(WORKSHEET_NAME)

    rows = manifest_to_rows(manifest)
    ws.resize(rows=len(rows), cols=len(MANIFEST_COLUMNS))

    # Written in chunks rather than one giant values.update call -- at
    # thousands of rows a single request is more likely to hit a Sheets API
    # request-size/timeout limit and fail outright, losing the whole export,
    # than to go through. A chunk failing partway still leaves everything
    # before it in the sheet.
    chunk_size = 2000
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start:start + chunk_size]
        ws.update(chunk, f"A{start + 1}")
        print(f"  wrote rows {start + 1}-{start + len(chunk)} of {len(rows)}")

    ws.freeze(rows=1)

    print(f"Wrote {len(manifest)} trials to: {sh.url}")


if __name__ == "__main__":
    main()

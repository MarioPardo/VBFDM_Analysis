import uproot
import os

def check_root_files(directory):
    failed_files = []
    total_files = 0

    for root, _, files in os.walk(directory):
        for fname in files:
            print("Name:", fname)
            if fname.endswith(".root"):
                fpath = os.path.join(root, fname)
                total_files += 1
                try:
                    with uproot.open(fpath) as f:
                        _ = f.keys()  # Just trying to list keys is enough to test readability
                except Exception as e:
                    print(f"❌ Failed: {fpath} — {type(e).__name__}: {e}")
                    failed_files.append(fpath)
                else:
                    print(f"✅ OK: {fpath}")

    print(f"\nChecked {total_files} files.")
    if failed_files:
        print(f"{len(failed_files)} files failed integrity check:")
        for f in failed_files:
            print(" -", f)
    else:
        print("All files passed.")

# Example usage
check_root_files("/home/phenoprojects/MC_Samples/background_data/Wlnu_jets")

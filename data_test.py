import pickle

# Load both datasets
path_0 = "imaging-jan26/wrapped_snn_network-poly.pkl"
path_90 = "imaging-jan26/wrapped_snn_network-poly_psi90.pkl"

with open(path_0, "rb") as f0, open(path_90, "rb") as f90:
    data_0 = pickle.load(f0, encoding="latin1")
    data_90 = pickle.load(f90, encoding="latin1")

labels_0 = data_0["labels"]
labels_90 = data_90["labels"]

print(f"0° dataset size: {len(labels_0)}")
print(f"90° dataset size: {len(labels_90)}")

# Check if lengths match
if len(labels_0) != len(labels_90):
    print("Dataset lengths do not match.")
else:
    print("Dataset lengths match. :) :)")

    # Check label-by-label match
    mismatches = []
    for i, (l0, l90) in enumerate(zip(labels_0, labels_90)):
        if l0 != l90:
            mismatches.append((i, l0, l90))

    if not mismatches:
        print("All labels match perfectly!")
    else:
        print(f"Found {len(mismatches)} mismatches.")
        print("First few mismatches:", mismatches[:5])

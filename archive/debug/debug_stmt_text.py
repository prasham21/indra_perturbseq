import pandas as pd

# === CONFIG ===
ORIGINAL_FILE = "/Users/prashammarfatia/Downloads/indra_top100_with_directional_consistency_1.csv"
PERMUTED_FILE = "/Users/prashammarfatia/Downloads/indra_top100_PERMUTED_negative_control.csv"

# === LOAD DATA ===
df_original = pd.read_csv(ORIGINAL_FILE)
df_permuted = pd.read_csv(PERMUTED_FILE)

print("="*70)
print("📊 HOP DISTRIBUTION COMPARISON")
print("="*70)

# === ORIGINAL DATA HOP DISTRIBUTION ===
print("\n✅ ORIGINAL (REAL) PAIRS:")
print("-" * 70)
original_hops = df_original['hop_number'].value_counts().sort_index()
print(original_hops)
print(f"\nTotal: {len(df_original)} pairs")

for hop in ['1hop', '2hop', '3hop', '4hop']:
    count = original_hops.get(hop, 0)
    pct = (count / len(df_original)) * 100
    print(f"  {hop}: {count} ({pct:.1f}%)")

# === PERMUTED DATA HOP DISTRIBUTION ===
print("\n🔀 PERMUTED (FAKE) PAIRS:")
print("-" * 70)
permuted_hops = df_permuted['hop_number'].value_counts().sort_index()
print(permuted_hops)
print(f"\nTotal: {len(df_permuted)} pairs")

for hop in ['1hop', '2hop', '3hop', '4hop']:
    count = permuted_hops.get(hop, 0)
    pct = (count / len(df_permuted)) * 100
    print(f"  {hop}: {count} ({pct:.1f}%)")

# === COMPARISON ===
print("\n" + "="*70)
print("📈 COMPARISON")
print("="*70)

print(f"\n{'Hop Level':<15} {'Original':<15} {'Permuted':<15} {'Difference'}")
print("-" * 70)

for hop in ['1hop', '2hop', '3hop', '4hop']:
    orig_count = original_hops.get(hop, 0)
    perm_count = permuted_hops.get(hop, 0)
    diff = orig_count - perm_count
    print(f"{hop:<15} {orig_count:<15} {perm_count:<15} {diff:+d}")

# Calculate average hop number (treating 1hop=1, 2hop=2, etc.)
def calculate_avg_hops(df):
    hop_map = {'1hop': 1, '2hop': 2, '3hop': 3, '4hop': 4}
    df['hop_numeric'] = df['hop_number'].map(hop_map)
    return df['hop_numeric'].mean()

avg_original = calculate_avg_hops(df_original.copy())
avg_permuted = calculate_avg_hops(df_permuted.copy())

print("\n" + "="*70)
print("📊 AVERAGE HOP DISTANCE")
print("="*70)
print(f"Original (Real) pairs:    {avg_original:.2f} hops")
print(f"Permuted (Fake) pairs:    {avg_permuted:.2f} hops")
print(f"Difference:               {avg_original - avg_permuted:+.2f} hops")

if avg_original < avg_permuted:
    print("\n✅ GOOD: Real pairs have FEWER hops (more direct connections)")
elif avg_original > avg_permuted:
    print("\n⚠️ CONCERNING: Real pairs have MORE hops than random pairs")
else:
    print("\n⚠️ CONCERNING: Real and fake pairs have similar hop distances")

print("\n" + "="*70)
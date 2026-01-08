import numpy as np
from scipy import stats

print("="*80)
print("📊 RANK-SUM TEST: Real vs Permuted")
print("="*80)

# === YOUR NUMBERS ===
# Real data
real_1hop = 256
real_2hop = 26769
real_3hop = 47234
real_4hop = 6797  # Not explained

# Permuted data
perm_1hop = 213
perm_2hop = 11337
perm_3hop = 26465
perm_4hop = 27485  # Not explained

print("\n📊 Input data:")
print(f"\nReal:")
print(f"   1-hop: {real_1hop:,}")
print(f"   2-hop: {real_2hop:,}")
print(f"   3-hop: {real_3hop:,}")
print(f"   Not explained: {real_4hop:,}")
print(f"   Total: {real_1hop + real_2hop + real_3hop + real_4hop:,}")

print(f"\nPermuted:")
print(f"   1-hop: {perm_1hop:,}")
print(f"   2-hop: {perm_2hop:,}")
print(f"   3-hop: {perm_3hop:,}")
print(f"   Not explained: {perm_4hop:,}")
print(f"   Total: {perm_1hop + perm_2hop + perm_3hop + perm_4hop:,}")

# === CREATE RANK ARRAYS (like MATLAB zeros + values) ===
# data_original=[zeros(1,256)+1 zeros(1,26769)+2 zeros(1,47234)+3 zeros(1,6797)+4];
data_original = np.concatenate([
    np.ones(real_1hop) * 1,
    np.ones(real_2hop) * 2,
    np.ones(real_3hop) * 3,
    np.ones(real_4hop) * 4
])

# data_permuted=[zeros(1,213)+1 zeros(1,11337)+2 zeros(1,26465)+3 zeros(1,27485)+4];
data_permuted = np.concatenate([
    np.ones(perm_1hop) * 1,
    np.ones(perm_2hop) * 2,
    np.ones(perm_3hop) * 3,
    np.ones(perm_4hop) * 4
])

print(f"\n✅ Rank arrays created:")
print(f"   Real array size: {len(data_original):,}")
print(f"   Permuted array size: {len(data_permuted):,}")

# === DESCRIPTIVE STATISTICS ===
print("\n" + "="*80)
print("📈 DESCRIPTIVE STATISTICS")
print("="*80)

print(f"\n{'Metric':<30} {'Real':<15} {'Permuted':<15}")
print("-"*60)
print(f"{'Mean rank':<30} {np.mean(data_original):<15.4f} {np.mean(data_permuted):<15.4f}")
print(f"{'Median rank':<30} {np.median(data_original):<15.1f} {np.median(data_permuted):<15.1f}")
print(f"{'Std deviation':<30} {np.std(data_original):<15.4f} {np.std(data_permuted):<15.4f}")

# === MANN-WHITNEY U TEST (equivalent to MATLAB ranksum) ===
print("\n" + "="*80)
print("🧪 MANN-WHITNEY U TEST (ranksum)")
print("="*80)

# Test if real has LOWER ranks (alternative='less')
statistic, p_value = stats.mannwhitneyu(data_original, data_permuted, alternative='less')

print(f"\nNull Hypothesis: Real and permuted have the same rank distribution")
print(f"Alternative Hypothesis: Real pairs have LOWER ranks (explained at fewer hops)")
print(f"\nTest Results:")
print(f"   U-statistic: {statistic:,.0f}")
print(f"   P-value: {p_value:.4e}")

# Calculate effect size (rank-biserial correlation)
n1 = len(data_original)
n2 = len(data_permuted)
rank_biserial = 1 - (2*statistic) / (n1 * n2)
print(f"   Effect size (rank-biserial): {rank_biserial:.4f}")

# === INTERPRETATION ===
print("\n" + "="*80)
print("🎯 INTERPRETATION")
print("="*80)

if p_value < 0.001:
    print(f"\n✅✅✅ HIGHLY SIGNIFICANT (p = {p_value:.4e})")
    print(f"   Real pairs are explained at SIGNIFICANTLY LOWER hops!")
elif p_value < 0.05:
    print(f"\n✅ SIGNIFICANT (p = {p_value:.4f})")
    print(f"   Real pairs are explained at significantly LOWER hops!")
else:
    print(f"\n⚠️ NOT SIGNIFICANT (p = {p_value:.4f})")
    print(f"   No significant difference in rank distribution")

mean_diff = np.mean(data_original) - np.mean(data_permuted)
print(f"\nMean rank difference: {mean_diff:+.4f}")
if mean_diff < 0:
    print(f"   Real pairs have LOWER mean rank (better) by {abs(mean_diff):.4f}")
else:
    print(f"   Permuted pairs have lower mean rank")

# === DISTRIBUTION COMPARISON ===
print("\n" + "="*80)
print("📊 DISTRIBUTION BREAKDOWN")
print("="*80)

real_total = real_1hop + real_2hop + real_3hop + real_4hop
perm_total = perm_1hop + perm_2hop + perm_3hop + perm_4hop

print(f"\n{'Rank':<20} {'Real':<15} {'Real %':<15} {'Permuted':<15} {'Permuted %':<15}")
print("-"*80)
print(f"{'1 (1-hop)':<20} {real_1hop:<15,} {real_1hop/real_total*100:<15.2f}% {perm_1hop:<15,} {perm_1hop/perm_total*100:<15.2f}%")
print(f"{'2 (2-hop)':<20} {real_2hop:<15,} {real_2hop/real_total*100:<15.2f}% {perm_2hop:<15,} {perm_2hop/perm_total*100:<15.2f}%")
print(f"{'3 (3-hop)':<20} {real_3hop:<15,} {real_3hop/real_total*100:<15.2f}% {perm_3hop:<15,} {perm_3hop/perm_total*100:<15.2f}%")
print(f"{'4 (Not explained)':<20} {real_4hop:<15,} {real_4hop/real_total*100:<15.2f}% {perm_4hop:<15,} {perm_4hop/perm_total*100:<15.2f}%")
print(f"{'Total':<20} {real_total:<15,} {100.0:<15.2f}% {perm_total:<15,} {100.0:<15.2f}%")

print("\n" + "="*80)
print("✅ Analysis complete!")
print("="*80)
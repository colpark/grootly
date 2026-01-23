#!/usr/bin/env python3
"""
Plot joint trajectories from trossen_client logs until error.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data extracted from logs (8 actions before error)
# Joint indices: 0-5 = arm joints, 6 = gripper

# Previous positions (what robot was at before each action)
prev_left = np.array([
    [-1.9073777e-04, 1.0465782e+00, 5.2357519e-01, 6.2733656e-01, -1.9073777e-04, -1.9073777e-04, 8.2589686e-06],
    [0.04980926, 1.0965781, 0.47357517, 0.57733655, -0.05019074, 0.04980926, 0.02000826],
    [0.09980926, 1.1465781, 0.4467913, 0.52733654, -0.10019074, 0.09487714, 0.04000826],
    [0.14980926, 1.196578, 0.48040175, 0.47733653, -0.15019074, 0.09256259, 0.06000826],
    [0.19980925, 1.2340416, 0.48224467, 0.4273365, -0.20019074, 0.0964526, 0.08000825],
    [0.24980925, 1.2352663, 0.48283738, 0.3773365, -0.25019073, 0.09839533, 0.10000825],
    [0.29980925, 1.2594409, 0.4791586, 0.3273365, -0.30019075, 0.13999367, 0.12000825],
    [0.34980926, 1.2418704, 0.46161297, 0.27733648, -0.35019076, 0.15518601, 0.14000824],
])

prev_right = np.array([
    [-1.9073777e-04, 1.0469596e+00, 5.2357519e-01, 6.2619209e-01, -1.9073777e-04, -1.9073777e-04, 4.9211085e-06],
    [4.9809262e-02, 1.0853615e+00, 4.7357517e-01, 5.8059508e-01, 7.5922413e-03, -1.9073777e-04, 4.9211085e-06],
    [0.09980926, 1.0657269, 0.42357516, 0.5788504, 0.00858658, -0.00225638, 0.00611744],
    [1.4980926e-01, 1.0872680e+00, 3.8522428e-01, 5.8408576e-01, 8.0345543e-03, -7.4575725e-04, 6.6525075e-03],
    [0.19980925, 1.0661724, 0.3696299, 0.59281033, 0.00902209, -0.00386153, 0.00514002],
    [0.24980925, 1.0607032, 0.36260104, 0.58408564, 0.00941555, -0.00107424, 0.00418047],
    [0.29980925, 1.013824, 0.3589616, 0.5887654, 0.01042267, -0.00303445, 0.00160007],
    [3.4980926e-01, 1.0263264e+00, 3.6805066e-01, 5.8508778e-01, 9.6434327e-03, -1.7403476e-03, 1.0259056e-03],
])

# Model outputs (what policy predicted)
model_left = np.array([
    [0.49659324, 1.1837537, 0.43555194, -0.45000035, -0.73346275, 0.13214104, 1.3737478],
    [0.49551886, 1.2095498, 0.4467913, -0.44525948, -0.698616, 0.09487714, 1.3737478],
    [0.4556964, 1.2169116, 0.48040175, -0.48212922, -0.68594396, 0.09256259, 1.414134],
    [0.446389, 1.2340416, 0.48224467, -0.49757004, -0.68911237, 0.0964526, 1.4057612],
    [0.4401356, 1.2352663, 0.48283738, -0.48003706, -0.6796087, 0.09839533, 1.4204466],
    [0.47208494, 1.2594409, 0.4791586, -0.46702445, -0.67327297, 0.13999367, 1.4268922],
    [0.4385747, 1.2418704, 0.46161297, -0.47359708, -0.6796087, 0.15518601, 1.4272633],
    [0.4687254, 1.2454951, 0.4737211, -0.47114766, -0.67960817, 0.17294325, 1.4263159],
])

model_right = np.array([
    [8.5065323e-01, 1.0853615e+00, 4.1486824e-01, 5.8059508e-01, 7.5922413e-03, -1.9073777e-04, 4.9211085e-06],
    [0.8232451, 1.0657269, 0.3956871, 0.5788504, 0.00858658, -0.00225638, 0.00611744],
    [8.3188266e-01, 1.0872680e+00, 3.8522428e-01, 5.8408576e-01, 8.0345543e-03, -7.4575719e-04, 6.6525075e-03],
    [0.8676408, 1.0661724, 0.3696299, 0.59281033, 0.00902209, -0.00386153, 0.00514002],
    [0.86469775, 1.0607032, 0.36260104, 0.58408564, 0.00941555, -0.00107424, 0.00418047],
    [0.8762777, 1.013824, 0.3589616, 0.5887654, 0.01042267, -0.00303445, 0.00160007],
    [8.8437432e-01, 1.0263264e+00, 3.6805066e-01, 5.8508778e-01, 9.6434327e-03, -1.7403476e-03, 1.0259056e-03],
    [8.9098012e-01, 9.9507368e-01, 3.6422798e-01, 5.7892621e-01, 1.0271481e-02, -6.9903390e-04, 1.0102680e-03],
])

# Target positions (what we sent to robot after clamping)
target_left = np.array([
    [0.04980926, 1.0965781, 0.47357517, 0.57733655, -0.05019074, 0.04980926, 0.02000826],
    [0.09980926, 1.1465781, 0.4467913, 0.52733654, -0.10019074, 0.09487714, 0.04000826],
    [0.14980926, 1.196578, 0.48040175, 0.47733653, -0.15019074, 0.09256259, 0.06000826],
    [0.19980925, 1.2340416, 0.48224467, 0.4273365, -0.20019074, 0.0964526, 0.08000825],
    [0.24980925, 1.2352663, 0.48283738, 0.3773365, -0.25019073, 0.09839533, 0.10000825],
    [0.29980925, 1.2594409, 0.4791586, 0.3273365, -0.30019075, 0.13999367, 0.12000825],
    [0.34980926, 1.2418704, 0.46161297, 0.27733648, -0.35019076, 0.15518601, 0.14000824],
    [0.39980927, 1.2454951, 0.4737211, 0.22733648, -0.40019077, 0.17294325, 0.16000824],
])

target_right = np.array([
    [4.9809262e-02, 1.0853615e+00, 4.7357517e-01, 5.8059508e-01, 7.5922413e-03, -1.9073777e-04, 4.9211085e-06],
    [0.09980926, 1.0657269, 0.42357516, 0.5788504, 0.00858658, -0.00225638, 0.00611744],
    [1.4980926e-01, 1.0872680e+00, 3.8522428e-01, 5.8408576e-01, 8.0345543e-03, -7.4575725e-04, 6.6525075e-03],
    [0.19980925, 1.0661724, 0.3696299, 0.59281033, 0.00902209, -0.00386153, 0.00514002],
    [0.24980925, 1.0607032, 0.36260104, 0.58408564, 0.00941555, -0.00107424, 0.00418047],
    [0.29980925, 1.013824, 0.3589616, 0.5887654, 0.01042267, -0.00303445, 0.00160007],
    [3.4980926e-01, 1.0263264e+00, 3.6805066e-01, 5.8508778e-01, 9.6434327e-03, -1.7403476e-03, 1.0259056e-03],
    [3.9980927e-01, 9.9507368e-01, 3.6422798e-01, 5.7892621e-01, 1.0271481e-02, -6.9903396e-04, 1.0102680e-03],
])

# Time steps
steps = np.arange(1, 9)

# Joint names
joint_names = ['J0 (waist)', 'J1 (shoulder)', 'J2 (elbow)', 'J3 (forearm)', 'J4 (wrist1)', 'J5 (wrist2)', 'J6 (gripper)']

# Create figure with subplots
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
fig.suptitle('Joint Trajectories Until Motor Error\n(Motor ID 7 / Gripper exceeded 12.5 rad limit)', fontsize=14, fontweight='bold')

# Plot 1: Left arm - Model vs Target (all joints)
ax = axes[0, 0]
for j in range(7):
    ax.plot(steps, model_left[:, j], 'o--', label=f'{joint_names[j]} (model)', alpha=0.7)
    ax.plot(steps, target_left[:, j], 's-', label=f'{joint_names[j]} (target)', alpha=0.7)
ax.set_xlabel('Action Step')
ax.set_ylabel('Position (rad)')
ax.set_title('LEFT ARM: Model Output vs Clamped Target')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
ax.grid(True, alpha=0.3)
ax.axhline(y=12.5, color='r', linestyle='--', label='Motor limit')
ax.axhline(y=-12.5, color='r', linestyle='--')

# Plot 2: Right arm - Model vs Target (all joints)
ax = axes[0, 1]
for j in range(7):
    ax.plot(steps, model_right[:, j], 'o--', label=f'{joint_names[j]} (model)', alpha=0.7)
    ax.plot(steps, target_right[:, j], 's-', label=f'{joint_names[j]} (target)', alpha=0.7)
ax.set_xlabel('Action Step')
ax.set_ylabel('Position (rad)')
ax.set_title('RIGHT ARM: Model Output vs Clamped Target')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
ax.grid(True, alpha=0.3)

# Plot 3: Left gripper detailed (the problematic joint)
ax = axes[1, 0]
ax.plot(steps, model_left[:, 6], 'ro-', linewidth=2, markersize=8, label='Model output')
ax.plot(steps, target_left[:, 6], 'bs-', linewidth=2, markersize=8, label='Clamped target')
ax.plot(steps, prev_left[:, 6], 'g^-', linewidth=2, markersize=8, label='Prev position')
ax.axhline(y=12.5, color='r', linestyle='--', linewidth=2, label='Motor limit (12.5)')
ax.axhline(y=12.5/78, color='orange', linestyle=':', linewidth=2, label='~0.16 (12.5/78 scaling)')
ax.set_xlabel('Action Step')
ax.set_ylabel('Position (rad)')
ax.set_title('LEFT GRIPPER (Motor ID 7) - ERROR SOURCE\nModel outputs ~1.4 rad, motor receives ~12.5 rad (78x scaling?)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.5, 2.0)

# Plot 4: Right gripper
ax = axes[1, 1]
ax.plot(steps, model_right[:, 6], 'ro-', linewidth=2, markersize=8, label='Model output')
ax.plot(steps, target_right[:, 6], 'bs-', linewidth=2, markersize=8, label='Clamped target')
ax.plot(steps, prev_right[:, 6], 'g^-', linewidth=2, markersize=8, label='Prev position')
ax.set_xlabel('Action Step')
ax.set_ylabel('Position (rad)')
ax.set_title('RIGHT GRIPPER')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Left arm joint 0 (also showing issues - clamped heavily)
ax = axes[2, 0]
ax.plot(steps, model_left[:, 0], 'ro-', linewidth=2, markersize=8, label='Model output (~0.45)')
ax.plot(steps, target_left[:, 0], 'bs-', linewidth=2, markersize=8, label='Clamped target')
ax.plot(steps, prev_left[:, 0], 'g^-', linewidth=2, markersize=8, label='Prev position')
ax.set_xlabel('Action Step')
ax.set_ylabel('Position (rad)')
ax.set_title('LEFT J0 (waist) - Heavy clamping applied')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Left arm joint 3 (negative values - sign flip issue?)
ax = axes[2, 1]
ax.plot(steps, model_left[:, 3], 'ro-', linewidth=2, markersize=8, label='Model output (negative!)')
ax.plot(steps, target_left[:, 3], 'bs-', linewidth=2, markersize=8, label='Clamped target')
ax.plot(steps, prev_left[:, 3], 'g^-', linewidth=2, markersize=8, label='Prev position')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('Action Step')
ax.set_ylabel('Position (rad)')
ax.set_title('LEFT J3 (forearm) - Model outputs NEGATIVE, target POSITIVE')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 7: Delta analysis - showing clamping effect
ax = axes[3, 0]
delta_left = target_left - prev_left
for j in range(7):
    ax.plot(steps, delta_left[:, j], 'o-', label=joint_names[j])
ax.axhline(y=0.05, color='r', linestyle='--', label='Max delta (0.05)')
ax.axhline(y=-0.05, color='r', linestyle='--')
ax.axhline(y=0.02, color='orange', linestyle=':', label='Max gripper delta (0.02)')
ax.axhline(y=-0.02, color='orange', linestyle=':')
ax.set_xlabel('Action Step')
ax.set_ylabel('Delta (rad)')
ax.set_title('LEFT ARM: Position Changes Per Step (should be within limits)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 8: Key insight - gripper scaling problem
ax = axes[3, 1]
# If motor received 12.512637 and we sent ~0.16, scaling is ~78x
sent_gripper = target_left[:, 6]
received_estimate = sent_gripper * 78  # Estimated what motor actually sees
ax.plot(steps, sent_gripper, 'bs-', linewidth=2, markersize=8, label='What we sent (clamped)')
ax.plot(steps, received_estimate, 'ro-', linewidth=2, markersize=8, label='Estimated motor value (78x)')
ax.axhline(y=12.5, color='r', linestyle='--', linewidth=2, label='Motor limit (12.5)')
ax.axhline(y=-12.5, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Action Step')
ax.set_ylabel('Position (rad)')
ax.set_title('GRIPPER SCALING PROBLEM\nSent 0.16 rad, motor sees 12.5 rad (78x internal scaling)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('deployment/joint_trajectories_error.png', dpi=150, bbox_inches='tight')
print("Saved: deployment/joint_trajectories_error.png")

# Also create a summary text
print("\n" + "="*60)
print("KEY FINDINGS FROM TRAJECTORY ANALYSIS")
print("="*60)
print(f"\n1. LEFT GRIPPER (Motor ID 7) - THE ERROR SOURCE:")
print(f"   - Model outputs: ~1.37-1.43 rad consistently")
print(f"   - After clamping: 0.02 -> 0.16 rad (incrementing by 0.02/step)")
print(f"   - Motor received: 12.512637 rad (OUT OF BOUNDS)")
print(f"   - Implied scaling: 12.51 / 0.16 = ~78x")

print(f"\n2. LEFT ARM JOINTS 0, 3, 4 - HEAVY CLAMPING:")
print(f"   - J0: Model wants ~0.45, clamped to ~0.05->0.40")
print(f"   - J3: Model outputs NEGATIVE (-0.45 to -0.50)")
print(f"        but target is POSITIVE (0.22 to 0.63)")
print(f"   - J4: Model outputs NEGATIVE (-0.67 to -0.73)")
print(f"        but target is NEGATIVE (-0.05 to -0.40)")

print(f"\n3. RIGHT ARM:")
print(f"   - J0: Model wants ~0.85-0.89, heavily clamped to 0.05->0.40")
print(f"   - Other joints: relatively stable")

print(f"\n4. ROOT CAUSE:")
print(f"   - Gripper has internal 78x position scaling in LeRobot driver")
print(f"   - MAX_GRIPPER_DELTA=0.02 * 78 = 1.56 rad/step in motor units")
print(f"   - After 8 steps: 0.16 * 78 = 12.48 rad -> exceeds 12.5 limit")

print(f"\n5. RECOMMENDED FIX:")
print(f"   - MAX_GRIPPER_DELTA should be 0.02/78 = 0.000256 rad")
print(f"   - OR apply inverse scaling to gripper before sending")
print(f"   - OR find where the 78x scaling is applied and compensate")
print("="*60)

plt.show()

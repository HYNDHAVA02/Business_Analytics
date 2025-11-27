import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Generate conceptual survival curves for demonstration
# This simulates the effect of Absences on dropout risk over time

weeks = np.linspace(0, 40, 100)  # 40 weeks (school year)

# Create survival functions for different absence levels
# Higher absences = faster decline in survival probability
absences_levels = [0, 10, 20, 30]
colors = ['green', 'blue', 'orange', 'red']

plt.figure(figsize=(10, 6))

for absences, color in zip(absences_levels, colors):
    # Exponential decay based on absences (higher absences = steeper decline)
    # S(t) = exp(-lambda * t) where lambda increases with absences
    lambda_val = 0.001 + (absences * 0.003)  # Hazard rate
    survival_prob = np.exp(-lambda_val * weeks)
    
    plt.plot(weeks, survival_prob, label=f'Absences = {absences}', 
             color=color, linewidth=2)

plt.title('Survival Curves by Absences (Risk of Dropout)', fontsize=14, fontweight='bold')
plt.xlabel('Weeks Enrolled', fontsize=12)
plt.ylabel('Survival Probability (Retention Rate)', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, 40)
plt.ylim(0, 1)

# Add annotation for key insight
plt.annotate('High absences accelerate dropout risk', 
             xy=(20, 0.3), xytext=(25, 0.5),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10, color='red')

plt.tight_layout()
plt.savefig('strategic_analytics/reports/survival_curves.png', dpi=300, bbox_inches='tight')
print("✓ Survival curves plot saved successfully to strategic_analytics/reports/survival_curves.png")

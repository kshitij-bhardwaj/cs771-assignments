import matplotlib.pyplot as plt
from validation import values
# Assuming you have results stored in lists/arrays for each parameter
penalty = ['l1', 'l2'] 
my_fit_times = []
acc0_losses = []
acc1_losses = []
for i in range (len(penalty)):
    t_train, loss0, loss1 = values(penalty[i])
    my_fit_times.append(t_train)
    acc0_losses.append(loss0)
    acc1_losses.append(loss1)
# Create the figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
# Plot my_fit time
ax1.bar(penalty, my_fit_times, color=['tab:blue', 'tab:green'])
ax1.set_xlabel("Penalty")
ax1.set_ylabel("my_fit Time (seconds)")
ax1.set_title("my_fit Time Comparison")
for c in ax1.containers:
    d = c.datavalues.astype(float)
    for i in range (len(d)) :
        d[i] = round(d[i], 2)
    labels = d.astype(str)
    ax1.bar_label(c, label_type='edge', labels=labels)

# Plot 1 - accuracy for Response0
ax2.bar(penalty, acc0_losses, color=['tab:blue', 'tab:green'])
ax2.set_xlabel("Penalty")
ax2.set_ylabel("1 - Accuracy (Response0)")
ax2.set_title("Response0 Classification Error")
for c in ax2.containers:
    d = c.datavalues.astype(float)
    for i in range (len(d)) :
        d[i] = round(d[i], 5)
    labels = d.astype(str)
    ax2.bar_label(c, label_type='edge', labels=labels)

# Plot 1 - accuracy for Response1
ax3.bar(penalty, acc1_losses, color=['tab:blue', 'tab:green'])
ax3.set_xlabel("Penalty")
ax3.set_ylabel("1 - Accuracy (Response1)")
ax3.set_title("Response1 Classification Error")
for c in ax3.containers:
    d = c.datavalues.astype(float)
    for i in range (len(d)) :
        d[i] = round(d[i], 4)
    labels = d.astype(str)
    ax3.bar_label(c, label_type='edge', labels=labels)


# Customize the plot (optional)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from validation import values
# Assuming you have results stored in lists/arrays for each parameter
loss_functions = ["hinge", "squared_hinge"]  # List of loss functions used
my_fit_times = []
acc0_losses = []
acc1_losses = []
for loss_function in loss_functions:
    t_train, loss0, loss1 = values(loss_function)
    my_fit_times.append(t_train)
    acc0_losses.append(loss0)
    acc1_losses.append(loss1)
# Create the figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
print(my_fit_times, acc0_losses, acc1_losses)
# Plot my_fit time
ax1.bar(loss_functions, my_fit_times, color=['tab:blue', 'tab:green'])
ax1.set_xlabel("Loss Function")
ax1.set_ylabel("my_fit Time (seconds)")
ax1.set_title("my_fit Time Comparison")
for c in ax1.containers:
    d = c.datavalues.astype(float)
    print(d.dtype)
    for i in range (len(d)) :
        d[i] = round(d[i], 2)
    labels = d.astype(str)
    ax1.bar_label(c, label_type='edge', labels=labels)

# Plot 1 - accuracy for Response0
ax2.bar(loss_functions, acc0_losses, color=['tab:blue', 'tab:green'])
ax2.set_xlabel("Loss Function")
ax2.set_ylabel("1 - Accuracy (Response0)")
ax2.set_title("Response0 Classification Error")
for c in ax2.containers:
    d = c.datavalues.astype(float)
    print(d.dtype)
    for i in range (len(d)) :
        d[i] = round(d[i], 5)
    labels = d.astype(str)
    ax2.bar_label(c, label_type='edge', labels=labels)

# Plot 1 - accuracy for Response1
ax3.bar(loss_functions, acc1_losses, color=['tab:blue', 'tab:green'])
ax3.set_xlabel("Loss Function")
ax3.set_ylabel("1 - Accuracy (Response1)")
ax3.set_title("Response1 Classification Error")
for c in ax3.containers:
    d = c.datavalues.astype(float)
    print(d.dtype)
    for i in range (len(d)) :
        d[i] = round(d[i], 4)
    labels = d.astype(str)
    ax3.bar_label(c, label_type='edge', labels=labels)

# Customize the plot (optional)
plt.tight_layout()
plt.show()

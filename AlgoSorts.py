"""
Performance Benchmarking - full runnable script

Features:
- TkAgg backend forced for matplotlib
- Threaded benchmark to keep GUI responsive
- Per-algorithm exception handling (RecursionError, MemoryError)
- Hybrid Quick Sort used when dataset is Reverse and size > 1000
- Simple logging, results table, and chart embedded in Tkinter
"""

import matplotlib  # to draw the bar charts
matplotlib.use('TkAgg')  # Tells Matplotlib to work inside a Tkinter window
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext #to create the windows, buttons and tables
import random
import time # to measure seconds
import tracemalloc #to measure memory usage
import threading # To run the sort without freezing the window
import datetime
import traceback
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# DATASET GENERATION

def generate_dataset(size, dataset_type):
    size = int(size)
    if dataset_type == "Random":
        return [random.randint(1, size) for _ in range(size)] # Creates a list of random numbers between 1 and the size
    if dataset_type == "Partially Sorted":
        data = sorted([random.randint(1, size) for _ in range(size)]) # First creates a perfectly sorted list
        for _ in range(max(1, size // 10)):   # takes 10% of the numbers to mess up the order
            i = random.randint(0, size - 1)  # picks random two addresses (indices)
            j = random.randint(0, size - 1)
            data[i], data[j] = data[j], data[i]  # and the swap is made, 10% times of the dataset size. 
        return data
    if dataset_type == "Reverse":
        return list(range(size, 0, -1)) #range(start, stop, step)
    return []


# SORTING ALGORITHMS

def heap_sort(arr):
    data = arr.copy() #It creates a duplicate of the list so that the original data isn't messed up while we sort it.
    n = len(data)
    def heapify_iterative(n, i): #this function is to look at a parent and its two children and make sure the largest of the three is at the top.
        largest = i  # Assume the current spot (i) is the biggest
        while True:
            l = 2 * i + 1  # The left child's position
            r = 2 * i + 2  #The right child's position
            if l < n and data[l] > data[largest]:  #If the left child is bigger than the parent
                largest = l
            if r < n and data[r] > data[largest]: # If the right child is bigger than the current largest
                largest = r
            if largest != i:  # If the largest ISN'T the parent, 
                data[i], data[largest] = data[largest], data[i] # swap them
                i = largest #Move down to the new spot and repeat
            else:
                break
    for i in range(n // 2 - 1, -1, -1): #This loop starts from the middle of the list and works backward.
        heapify_iterative(n, i)
        #It calls heapify_iterative on every parent node to turn the random list into a Max Heap. After this loop finishes, the absolute largest number is guaranteed to be at data[0] (the very top of the pyramid).
    # Now that we have the largest number at the top, we "harvest" it.
    for i in range(n - 1, 0, -1):
        data[i], data[0] = data[0], data[i] # # Swap the top (largest) with the last item in the list.
        heapify_iterative(i, 0) # "Shrink" the heap by 1 and reorganize the new top
    return data
# We have coded two versions of Quick sort to handle dataset size bigger than 1000 without crushing due to recursion
def quick_sort(arr):
    if len(arr) <= 1: # A list with 0 or 1 item is already sorted
        return arr
    pivot = arr[-1] #picks the very last number as the "judge."
    left = [x for x in arr[:-1] if x <= pivot] # gathers every number smaller than or equal to the judge.
    right = [x for x in arr[:-1] if x > pivot] #gathers every number bigger than the judge.
    return quick_sort(left) + [pivot] + quick_sort(right) # It calls itself to sort the left pile, then the right pile, and glues them together with the judge in the middle.

# Hybrid quick sort: median-of-three pivot, insertion for small partitions, iterative stack
def hybrid_quick_sort(arr):
    data = arr.copy()
    def insertion_sort(a, lo, hi):
        for i in range(lo + 1, hi + 1): # Picks a card
            key = a[i]
            j = i - 1
            # Moves the card left until it's in the right spot
            while j >= lo and a[j] > key:
                a[j + 1] = a[j]
                j -= 1
            a[j + 1] = key

    def median_of_three(a, i, j, k):
        # Looks at the first, middle, and last number
        vals = [(a[i], i), (a[j], j), (a[k], k)]
        # Sorts them and picks the one in the middle
        vals.sort(key=lambda x: x[0])
        return vals[1][1] # Returns the location of the "middle-est" number
    # to avoid recursion, we use Stack
    stack = [(0, len(data) - 1)] #Starts with one task: sorts the whole thing
    while stack:
        lo, hi = stack.pop() # Takes the last task added to the list
        if lo >= hi: # If the section is empty, skips it
            continue
        if hi - lo + 1 <= 32: # Is this section 32 items or smaller?
            insertion_sort(data, lo, hi) # hand this "small pile" over to Insertion Sort.
            continue # Task done, move to next item in stack
        mid = (lo + hi) // 2 # Picks a smart pivot and moves it to the end (hi)
        pivot_index = median_of_three(data, lo, mid, hi)
        pivot = data[pivot_index]
        data[pivot_index], data[hi] = data[hi], data[pivot_index]
        i = lo
        for j in range(lo, hi):
            if data[j] <= pivot: # If number is smaller than pivot...
                data[i], data[j] = data[j], data[i] # swaps it to the left side
                i += 1
        data[i], data[hi] = data[hi], data[i] # Finally, puts the pivot in its permanent home at index 'i'
        # After splitting the list into a "Smaller" side and a "Larger" side, we add these new areas to our "To-Do" stack.
        left_size = i - 1 - lo
        right_size = hi - (i + 1) + 1
        # Optimization: Add the BIGGER side to the stack first
        # This keeps the stack as small as possible in memory
        if left_size > right_size:
            if lo < i - 1:
                stack.append((lo, i - 1)) # To-do: Sort left side later
            if i + 1 < hi:
                stack.append((i + 1, hi)) # To-do: Sort right side later
            if i + 1 < hi:
                stack.append((i + 1, hi))
            if lo < i - 1:
                stack.append((lo, i - 1))
    return data # Once the stack is empty, every piece is sorted

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2 # Splits the list into two halves.
    # 1. Call merge_sort on the left half (arr[:mid])
    # 2. Call merge_sort on the right half (arr[mid:])
    # 3. Use the 'merge' function below to zip them back together.
    return merge(merge_sort(arr[:mid]), merge_sort(arr[mid:]))

def merge(left, right): #this is where the actual sorting happens
    result = [] # This will be our new, perfectly sorted list
    i = j = 0 # i tracks the Left pile, j tracks the Right pile

    # WHILE BOTH PILES HAVE CARDS:
        # Look at the top card of both piles. 
        # Pick the smaller one and move it to our 'result' list.
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1 # Move to the next card in the Left pile
        else:
            result.append(right[j]); j += 1 # Move to the next card in the Right pile
    
    # Once one pile is empty, there might be cards left in the other.
    # Since those piles were already sorted, we just dump them at the end.
    result.extend(left[i:]); result.extend(right[j:])
    return result

def shell_sort(arr): 
    data = arr.copy()
    gap = len(data) // 2 # we compare numbers that are far apart
    while gap > 0:
        for i in range(gap, len(data)):
            temp = data[i] # Saves the number we want to move
            j = i

            # Checks numbers behind it, but jumps back by 'gap' distance
            while j >= gap and data[j - gap] > temp:
                data[j] = data[j - gap] # Shifts the larger number forward
                j -= gap # Moves our pointer back by the gap distance
            data[j] = temp # Puts 'temp' in its new spot
        gap //= 2
    return data

def radix_sort(arr):
    data = arr.copy()
    if not data:
        return data # Safety check(if the list is empty, just return it)
    max_num = max(data) # Find the biggest number in the list.
    exp = 1 # This is our 'digit tracker' (Exponent).
    while max_num // exp > 0: #keeps running as long as there are still digits left to sort.
        counting_sort(data, exp) #This calls a "helper" function. It sorts the entire list, but only looking at the digit at the current exp position.
        exp *= 10 # Moves to the next digit to the left.
    return data

def counting_sort(arr, exp):
    n = len(arr)
    output = [0] * n # A temporary list to hold the numbers in their new order
    count = [0] * 10 # A small list to count how many times each digit (0-9) appears
    for i in range(n): #count the digits
        index = arr[i] // exp
        count[index % 10] += 1 #This extracts the specific digit we care about.
    for i in range(1, 10):
        count[i] += count[i - 1] #It adds the previous count to the current one to find the starting position for each group.
    for i in range(n - 1, -1, -1): #goes through the list backwards, so numbers that have the same digit don't swap places
        index = arr[i] // exp 
        output[count[index % 10] - 1] = arr[i] #we look at the number’s digit, find its correct position from our count math, and place it into the output list
        count[index % 10] -= 1 # We move the "next available slot" for that digit back by one, so the next time we see that same digit, it goes in the spot right before this one.
    for i in range(n): #we copy all the numbers back into the original arr so the next pass of Radix Sort can use them
        arr[i] = output[i]

ALGORITHMS = { # Decoupling
    "Heap": heap_sort,
    "Quick Sort": quick_sort,
    "Merge": merge_sort,
    "Shell": shell_sort,
    "Radix": radix_sort
}


# GUI SETUP (Building the Application Window)

window = tk.Tk()
window.title("Performance Benchmarking")
window.geometry("980x780")

tk.Label(window, text="Performance Benchmarking", font=("Arial", 18, "bold")).pack(pady=8) # Main Header

# Input frame (Contains Size, Data Type, and the Start Button)
input_frame = tk.LabelFrame(window, text="Configuration & Controls", padx=10, pady=10)
input_frame.pack(fill="x", padx=12)

# Array Size Entry (Allows users to define 'n' (dataset scale))
tk.Label(input_frame, text="Array Size:").grid(row=0, column=0, sticky="w")
size_entry = tk.Entry(input_frame, width=12)
size_entry.insert(0, "5000")
size_entry.grid(row=0, column=1, padx=6)

# Data Pattern Dropdown: Select between Random, Sorted, or Reversed data

tk.Label(input_frame, text="Data Pattern:").grid(row=0, column=2, sticky="w")
dataset_var = tk.StringVar(value="Random")
ttk.Combobox(
    input_frame,
    textvariable=dataset_var,
    values=["Random", "Partially Sorted", "Reverse"],
    state="readonly",
    width=18
).grid(row=0, column=3, padx=6)

# The Main Trigger: This button starts the threaded analysis
run_button = tk.Button(input_frame, text="RUN ANALYSIS", font=("Arial", 11, "bold"), bg="#4CAF50", fg="white")
run_button.grid(row=0, column=4, padx=12)

# Dynamic Status Label: Updates to show "Running..." or "Idle"
status_label = tk.Label(input_frame, text="Status: Idle", anchor="w")
status_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

# Algorithm Selector (Dynamically creates checkboxes based on our ALGORITHMS dictionary)
algo_frame = tk.LabelFrame(window, text="Select Algorithms", padx=10, pady=10)
algo_frame.pack(fill="x", padx=12, pady=6)

algo_vars = {}
for col, name in enumerate(ALGORITHMS):
    var = tk.BooleanVar(value=True)
    algo_vars[name] = var
    tk.Checkbutton(algo_frame, text=name, variable=var).grid(row=0, column=col, padx=8)

# Results Treeview (A spreadsheet-style table to show Time and Memory stats)
table_frame = tk.LabelFrame(window, text="Benchmark Results", padx=10, pady=10)
table_frame.pack(fill="x", padx=12)

tree = ttk.Treeview(table_frame, columns=("Algorithm", "Time (s)", "Memory (KB)"), show="headings", height=6)
for col in ("Algorithm", "Time (s)", "Memory (KB)"):
    tree.heading(col, text=col)
    tree.column(col, anchor="center", width=200)
tree.pack(fill="x")

# Bottom: log and chart
bottom_frame = tk.Frame(window)
bottom_frame.pack(fill="both", expand=True, padx=12, pady=8)

log_frame = tk.LabelFrame(bottom_frame, text="Log", padx=6, pady=6)
log_frame.pack(side="left", fill="both", expand=False, padx=(0,8), pady=2)

log_text = scrolledtext.ScrolledText(log_frame, width=48, height=18, state="disabled", wrap="word")
log_text.pack(fill="both", expand=True)

chart_frame = tk.LabelFrame(bottom_frame, text="Execution Time vs Peak Memory", padx=6, pady=6)
chart_frame.pack(side="left", fill="both", expand=True, pady=2)

chart_canvas_container = tk.Frame(chart_frame)
chart_canvas_container.pack(fill="both", expand=True)


# BENCHMARK LOGIC (Performance Tracking & Threading)

# Helper: Appends messages to the ScrolledText box with a timestamp
def log(message):
    timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
    log_text.configure(state="normal")
    log_text.insert("end", f"{timestamp} {message}\n")
    log_text.see("end")
    log_text.configure(state="disabled")

def safe_int(value, default=1000, max_allowed=100000):
    try:
        v = int(value)
        if v < 1:
            return default
        return min(v, max_allowed)
    except Exception:
        return default
    
# Worker Thread: Runs the sorting logic so the GUI window doesn't freeze
def run_benchmark_thread():
    run_button.config(state="disabled") # 1. Lockdown UI (Prevent user from clicking things while test is running)
    for cb in algo_frame.winfo_children():
        cb.configure(state="disabled")
    status_label.config(text="Status: Running benchmark...")

    selected = [name for name, v in algo_vars.items() if v.get()] # 2. Gather Inputs
    if not selected:
        messagebox.showwarning("Error", "Select at least one algorithm.")
        status_label.config(text="Status: Idle")
        run_button.config(state="normal")
        for cb in algo_frame.winfo_children():
            cb.configure(state="normal")
        return

    size = safe_int(size_entry.get(), default=5000)
    pattern = dataset_var.get()

    times = []
    memories = []
    labels = []

    for name in selected: # 3. Main Test Loop (Iterate through each checked algorithm)
        # Apply special condition: use hybrid_quick_sort for Reverse and size > 1000
        if name == "Quick Sort" and pattern == "Reverse" and size > 1000:
            func = hybrid_quick_sort
            used_name = "Hybrid Quick Sort"
        else:
            func = ALGORITHMS[name]
            used_name = name

        data = generate_dataset(size, pattern)
        log(f"NEW DATASET for {used_name}: sample {data[:8]} ...")
        tracemalloc.start() #Start Performance Trackers
        start = time.perf_counter()
        try:
            func(data) # Execute the sort
        except RecursionError:
            log(f"RecursionError in {used_name}; falling back to built-in sorted()")
            _ = sorted(data)
        except MemoryError:
            log(f"MemoryError in {used_name}; falling back to built-in sorted()")
            _ = sorted(data)
        except Exception as e:
            log(f"Exception in {used_name}: {e}")
            traceback.print_exc()
            _ = sorted(data)
        end = time.perf_counter()
        try:
            _, peak = tracemalloc.get_traced_memory()
        except Exception:
            peak = 0
        tracemalloc.stop()

        t = end - start # 4. Data Conversion
        m_kb = peak / 1024.0

        times.append(t)
        memories.append(m_kb)
        labels.append(used_name)

        log(f"COMPLETED {used_name}: {t:.4f}s, peak {m_kb:.2f} KB")
        window.after(0, lambda r=(used_name, t, m_kb): tree.insert("", "end", values=(r[0], f"{r[1]:.4f}", f"{r[2]:.2f}"))) # Update table safely using .after (communicating back to Main Thread)

    def draw_chart(): # Creates the bar chart comparing Time vs Memory
        for w in chart_canvas_container.winfo_children():
            w.destroy() # Clear previous chart

        x = np.arange(len(labels))
        width = 0.35
        fig = plt.Figure(figsize=(8, 4))
        ax1 = fig.add_subplot(111) # Primary Axis: Time
        ax2 = ax1.twinx() # Secondary Axis: Memory (Overlayed)

        color_time = "#1f77b4"
        color_mem = "#d62728"

        # Plotting the two data series side-by-side
        ax1.bar(x - width / 2, times, width, label="Time (s)", color=color_time)
        ax2.bar(x + width / 2, memories, width, label="Memory (KB)", color=color_mem, alpha=0.7)

        ax1.set_ylabel("Time (seconds)")
        ax2.set_ylabel("Memory (KB)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=20)
        ax1.set_title(f"Comparative Benchmark (Pattern: {pattern} [n={size}])")

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_time, label="Time (s)"),
                           Patch(facecolor=color_mem, label="Memory (KB)")]
        ax1.legend(handles=legend_elements, loc="upper left")

        fig.tight_layout()

        # Embed the Matplotlib figure into the Tkinter frame
        canvas_fig = FigureCanvasTkAgg(fig, master=chart_canvas_container)
        canvas_fig.get_tk_widget().pack(fill="both", expand=True)

    window.after(0, draw_chart) # 5. Finalize: Draw chart and unlock UI

    status_label.config(text="Status: Benchmark session finished")
    run_button.config(state="normal")
    for cb in algo_frame.winfo_children():
        cb.configure(state="normal")

def on_run_clicked():
    for item in tree.get_children():
        tree.delete(item)
    log_text.configure(state="normal")
    log_text.delete("1.0", "end")
    log_text.configure(state="disabled")
    threading.Thread(target=run_benchmark_thread, daemon=True).start()

run_button.config(command=on_run_clicked)

# Start GUI with error handling so tracebacks are visible
def start_gui():
    try:
        window.mainloop()
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        try:
            messagebox.showerror("Unhandled Error", f"An error occurred:\n{tb.splitlines()[-1]}")
        except Exception:
            pass

if __name__ == "__main__":
    start_gui()

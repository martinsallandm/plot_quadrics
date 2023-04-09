# Plot quadrics

Python code for plotting quadrics using the marching cubes algorithm


## instalation

there is no need for installation other than install the requirements

```$ pip install -r requirements.txt```

### but...

- Change the path to the local folder on the ipynb files
- Change the path to the c lib in the plot_quad.py
- Compile the c files in plot_quad

To do that, enter the folder plot_quad and execute

```$ gcc -fPIC --shared compute.c -o compute.so ```


## usage:


plot_quadrics(E, lim, N, color)

```
E - list containing quadrics matrices
lim - plot limits
N - resolution of the plot
color - list containing the colors for each quadric
```

## example:

See ```test1.ipynb``` for a simple usage

See ```test2.ipynb``` for plotting several quadrics at the same time


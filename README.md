# mandelbrot-opencl
Mandelbrot set viewing application with high frame-rate to enable real-time exploration

## Dependencies
* OpenCL
* A compute device compatible with OpenCL (run clinfo to check)

## Build
```
$ make
```

## Run
```
$ ./mandel
```

## Performance
Since this application is meant to allow you to explore the Mandelbrot set in real time, some performance measurements are needed.

Currently, the default application reaches an average of 58 fps on my laptop, which has an integrated "Intel(R) HD Graphics Kabylake Desktop GT1.5" GPU, running OpenCL 2.0 Beignet 1.3.

Performance depends on which part of the Mandelbrot set is in view, and how many iterations are performed for each pixel on the screen. By default, almost the entire set is in view, and up to 1024 iterations are performed.

## Areas for improvement
The biggest potential improvement that I have found is in the method I am using to display the Mandelbrot set to the screen. Once a buffer of divergence iterations counts for each pixel is returned from the OpenCL kernel, I'm using the CPU to loop through each of these values and convert them into colors. This could certainly be sped up significanly with a second OpenCL kernel.

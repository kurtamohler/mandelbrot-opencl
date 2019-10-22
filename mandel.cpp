
#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include "util.hpp"
#include "err_code.h"

#include <vector>
#include <iostream>
#include <vector>
#include <math.h>

#include <SFML/Graphics.hpp>

#ifndef DEVICE
#define CL_DEVICE_TYPE_DEFAULT
#endif


#define TOL (0.001)     // tolerance used in float compare

#define PI 3.14159265

// #define ROWS 51
// #define COLS 101


using namespace std;


void test_various_loops() {

}

int main() {
    unsigned int x_size = 1920/2;
    unsigned int y_size = 1080/2;

    // unsigned int x_size = 101;
    // unsigned int y_size = 51;

    unsigned int total_pixels = x_size * y_size;

    vector<float> h_mandel_frame(total_pixels, 0);
    cl::Buffer d_mandel_frame;

    sf::RenderWindow window(
        sf::VideoMode(x_size, y_size),
        "OpenCL Mandelbrot"
    );

    float sum_fps = 0;

    int total_frames = 0;

    try {
        cl::Context context(DEVICE);
        cl::Program program(
            context,
            util::loadProgram("mandel.cl"),
            true
        );

        cl::CommandQueue queue(context);


        size_t max_workgroup_size = 512;
        // dev->getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_workgroup_size);

        // Create the kernel functor
        auto mandel = cl::make_kernel<
            cl::Buffer,
            unsigned int,
            unsigned int,
            float,
            float,
            float,
            float,
            unsigned int
        >(program, "mandel");

        d_mandel_frame = cl::Buffer(
            context,
            begin(h_mandel_frame),
            end(h_mandel_frame),
            true
        );

        unsigned int max_loops = 1024;
        unsigned int max_samples = 10;


        // Fully zoomed out to show all of the points not in the set
        // float x_min = -2.7;
        // float x_max = 0.3;
        // float y_min = -1.98   - 0.005;
        // float y_max = 0.02    + 0.005;



        float x_min = -2;
        float x_max = 1;
        float y_min = -1;
        float y_max = 1;


        // Close to worst case performance if in range where no point is
        // in the mandelbrot set
        // float x_min = -0;
        // float x_max = 0.1;
        // float y_min = -0.01;
        // float y_max = 0.01;

        mandel(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(total_pixels)
                // ,16
            ),
            d_mandel_frame,
            x_size,
            y_size,
            x_min, x_max,  // x range
            y_min, y_max,  // y range
            max_loops
        );

        util::Timer timer;

        while (window.isOpen()) {
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed)
                    window.close();
            }


            cl::copy(
                queue,
                d_mandel_frame,
                begin(h_mandel_frame),
                end(h_mandel_frame)
            );

            // queue.finish();

            mandel(
                cl::EnqueueArgs(
                    queue,
                    cl::NDRange(total_pixels)
                    ,16
                ),
                d_mandel_frame,
                x_size,
                y_size,
                x_min, x_max,  // x range
                y_min, y_max,  // y range
                max_loops
            );

            window.clear();

            float max_loops_recip = 8.0f / ((float) max_loops);

            sf::VertexArray points;
            for (int y = 0; y < y_size; y++) {

                for (int x = 0; x < x_size; x++) {


                    float point_val = h_mandel_frame[x + y * x_size];

                    float r_val = 0;
                    float b_val = 0;

                    if (point_val > 0) {
                        r_val = 127.0f + 127.0f * sin(point_val * 2 * PI * max_loops_recip);
                        b_val = ((float)max_loops) - r_val;
                    }

                    points.append(sf::Vertex(
                        sf::Vector2f(x,y),
                        sf::Color(
                            r_val,
                            0,
                            b_val,
                            255
                        )
                    ));
                }
            }
            window.draw(points);
            window.display();





            // FPS calculation
            double runtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
            timer.reset();

            double fps = 1.0 / runtime;
            // cout << fps << " fps" << endl;

            if (total_frames > 10) {
                sum_fps += fps;
            }
            total_frames++;



            // if (x_min < 0.299983f) {
            //     x_min = x_max - 0.99f * (x_max - x_min);
            //     y_min = y_max - 0.99f * (y_max - y_min);
            // } else {
            //     break;
            // }

            // if (total_frames >= 50) {
            //     break;
            // }
        }

        double avg_fps = sum_fps / ((double) (total_frames-10));
        cout << "Average fps: " << avg_fps << endl;
        double ns_per_pixel = 1000000000.0 / (x_size * y_size * sum_fps);
        cout << "ns per pixel: " << ns_per_pixel << endl;


    } catch (cl::Error err) {
        cout << "Exception" << endl;

        cerr << "ERROR: " << err.what()
            << "(" << err_code(err.err()) << ")"
            << endl;
    }
}
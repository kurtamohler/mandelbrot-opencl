
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

using namespace std;

class MandelbrotCL {
public:
    MandelbrotCL(
        unsigned int xSize,
        unsigned int ySize
    ) :
        xSize(xSize),
        ySize(ySize),
        totalPixels(xSize * ySize),
        divergeMandelIters_host(totalPixels, 0),
        window(
            sf::VideoMode(xSize, ySize),
            "OpenCL Mandelbrot"
        ),
        clContext(DEVICE),
        clProgram(
            clContext,
            util::loadProgram("mandel.cl"),
            true
        ),
        clQueue(clContext),
        mandelbrot_kernel(clProgram, "mandel"),
        mandelbrot_color_kernel(clProgram, "mandel_color"),
        view(window.getDefaultView())
    {
        divergeMandelIters_device = new cl::Buffer(
            clContext,
            begin(divergeMandelIters_host),
            end(divergeMandelIters_host),
            true
        );

        InitPoints();

        pixelColors_device = new cl::Buffer(
            clContext,
            begin(pixelColors_host),
            end(pixelColors_host),
            true
        );

        // Get the first frame started right off the bat
        mandelbrot_kernel(
            cl::EnqueueArgs(
                clQueue,
                cl::NDRange(totalPixels)
                // ,16
            ),
            *divergeMandelIters_device,
            xSize,
            ySize,
            xMin, xMax,  // x range
            yMin, yMax,  // y range
            maxMandelIters
        );

        mandelbrot_color_kernel(
            cl::EnqueueArgs(
                clQueue,
                cl::NDRange(totalPixels)
            ),
            *divergeMandelIters_device,
            *pixelColors_device,
            xSize,
            ySize,
            0,
            maxMandelIters
        );

    }


    bool WindowOpen() {
        return window.isOpen();
    }

    void InitPoints() {
        points.resize(totalPixels);
        pixelColors_host.resize(totalPixels);
        // pixelColors_host = new sf::Color[totalPixels];

        int point_ind = 0;

        for (int y = 0; y < ySize; y++) {
            for (int x = 0; x < xSize; x++) {
                points[point_ind] = sf::Vertex(
                    sf::Vector2f(x,y),
                    pixelColors_host[point_ind]
                );
                point_ind++;
            }
        }
    }

    void GenerateFrame() {
        static unsigned int color_add = 0;
        color_add++;
        PollEvents();

        cl::copy(
            clQueue,
            *pixelColors_device,
            begin(pixelColors_host),
            end(pixelColors_host)
        );


        mandelbrot_kernel(
            cl::EnqueueArgs(
                clQueue,
                cl::NDRange(totalPixels)
                // ,16
            ),
            *divergeMandelIters_device,
            xSize,
            ySize,
            xMin, xMax,  // x range
            yMin, yMax,  // y range
            maxMandelIters
        );

        mandelbrot_color_kernel(
            cl::EnqueueArgs(
                clQueue,
                cl::NDRange(totalPixels)
            ),
            *divergeMandelIters_device,
            *pixelColors_device,
            xSize,
            ySize,
            color_add,
            maxMandelIters
        );

        window.clear();
        int point_ind = 0;

        // for (int point_ind = 0; point_ind < totalPixels; point_ind++) {
        for (int y = 0; y < ySize; y++) {
            for (int x = 0; x < xSize; x++) {
                points[point_ind].color = pixelColors_host[point_ind];
                point_ind++;
            }
        }

        window.draw(points);
        window.display();
    }


private:

    bool PollEvents() {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();

            } else if (event.type == sf::Event::MouseWheelScrolled) {
                ApplyScrollZoom(event.mouseWheelScroll.delta);

            } else if (event.type == sf::Event::KeyPressed) {
                ApplyKeyPan(event.key);

            // } else if (event.type == sf::Event::Resized) {
            //     Resize(event.size.width, event.size.height);
            }
        }
    }

    /*
    void Resize(unsigned int xNew, unsigned int yNew) {
        xSize = xNew;
        ySize = yNew;
        totalPixels = xNew * yNew;
        points.resize(totalPixels);

        divergeMandelIters_host.resize(totalPixels);

        delete divergeMandelIters_device;

        divergeMandelIters_device = new cl::Buffer(
            clContext,
            begin(divergeMandelIters_host),
            end(divergeMandelIters_host),
            true
        );


        view.setSize({
            static_cast<float>(xNew),
            static_cast<float>(yNew)
        });
        window.setView(view);
 

        cout << xNew << ", " << yNew << endl;

    }
    */

    void ApplyScrollZoom(float delta) {
        float xDelta = (delta/10.0f) * (xMax - xMin);
        float yDelta = (delta/10.0f) * (yMax - yMin);

        xMin += xDelta;
        xMax -= xDelta;

        yMin += yDelta;
        yMax -= yDelta;

        cout << (xMax - xMin) << endl;
    }

    void ApplyKeyPan(sf::Event::KeyEvent key) {
        float factor = 0.03f;
        float xDelta = factor * (xMax - xMin);
        float yDelta = factor * (yMax - yMin);

        if (key.code == sf::Keyboard::A) {
            xMin -= xDelta;
            xMax -= xDelta;

        } else if (key.code == sf::Keyboard::D) {
            xMin += xDelta;
            xMax += xDelta;

        } else if (key.code == sf::Keyboard::W) {
            yMin -= yDelta;
            yMax -= yDelta;

        } else if (key.code == sf::Keyboard::S) {
            yMin += yDelta;
            yMax += yDelta;

        }

    }

    unsigned int xSize;
    unsigned int ySize;
    unsigned int totalPixels;

    // The number of mandelbrot iterations it takes for each pixel to diverge.
    // Zero indicates that the pixel did not diverge in the allotted time.
    vector<float> divergeMandelIters_host;
    cl::Buffer* divergeMandelIters_device;
    vector<sf::Color> pixelColors_host;
    cl::Buffer* pixelColors_device;

    cl::Context clContext;
    cl::Program clProgram;
    cl::CommandQueue clQueue;

    cl::make_kernel<
        cl::Buffer,
        unsigned int,
        unsigned int,
        float,
        float,
        float,
        float,
        unsigned int
    > mandelbrot_kernel;

    cl::make_kernel<
        cl::Buffer,
        cl::Buffer,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
    > mandelbrot_color_kernel;

    sf::RenderWindow window;
    sf::VertexArray points;
    sf::View view;

    util::Timer frameTimer;

    int totalFrames = 0;

    // unsigned int maxMandelIters = 1024;
    unsigned int maxMandelIters = 2048;
    // unsigned int maxMandelIters = 128;

    float xMin = -2 * (1920.0f/1080.0f);
    float xMax = 2 * (1920.0f/1080.0f);
    float yMin = -2;
    float yMax = 2;
};


int main() {
    unsigned int xSize = 2*1920/3;
    unsigned int ySize = 2*1080/3;

    try {
        MandelbrotCL mandelbrot(xSize, ySize);

        unsigned int numFrames = 0;
        util::Timer timer;

        while (mandelbrot.WindowOpen()) {
            mandelbrot.GenerateFrame();

            numFrames++;
        }

        float runtimeSeconds = static_cast<float>(timer.getTimeMilliseconds()) / 1000.0;
        float avgFps = ((float)numFrames) / runtimeSeconds;

        cout << "FPS: " << avgFps << endl;

    } catch (cl::Error err) {
        cout << "Exception" << endl;

        cerr << "ERROR: " << err.what()
            << "(" << err_code(err.err()) << ")"
            << endl;
    }
}

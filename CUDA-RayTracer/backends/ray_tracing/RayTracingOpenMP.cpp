#include "RayTracingOpenMP.h"
#include "../../scene/Color.h"
#include "../../scene/Vector.h"
#include "../../scene/Point.h"
#include <algorithm>
#include "Light.h"
#include "KdTree.h"

const int N = (int)1e6 + 5;
const float eps = 1e-6;
const int BYTES_PER_PIXEL = 3;



struct Resolution
{
	int width, height;

	Resolution()
	{
	}

	Resolution(int width, int height)
	{
		this->width = width;
		this->height = height;
	}
};

struct Random
{
};

struct Camera
{
	float width, height;
	Resolution resolution;
	Point focus_point = Point(0, 0, -1);
	int num_of_samples = 1024;
	std::unique_ptr<Color[]> active_pixel_sensor;
	Random random;

	Camera(float width, float height, Resolution resolution, int num_of_samples) :
		width(width),
		height(height),
		resolution(resolution),
		num_of_samples(num_of_samples),
		active_pixel_sensor(std::make_unique<Color[]>(resolution.width * resolution.height))
	{
		for (int i = 0; i < resolution.height; ++i)
		{
			for (int j = 0; j < resolution.width; ++j)
			{
				getActivePixelSensor(i, j) = Color(0, 0, 0);
			}
		}
	}

	Color& getActivePixelSensor(int x, int y) const
	{
		return active_pixel_sensor[resolution.width * x + y];
	}

	Vector get_random_vector(int row, int column) const
	{
		// not implemented
		return Vector(Point(0, 0, 0), 0, 0, 0);
	}

	Vector get_primary_vector(int row, int column) const
	{
		// row from [0,..., resolution.height-1]  column from [0, resolution.width-1]
		float pixel_width = width / resolution.width;
		float pixel_height = height / resolution.height;
		float x_cord = (-width / 2) + pixel_width * (0.5 + column);
		float y_cord = (-height / 2) + pixel_height * (0.5 + row);
		return Vector(focus_point, Point(x_cord, y_cord, 0));
	}

	void update(int row, int column, Color color)
	{
		getActivePixelSensor(row, column) += color;
	}

	Color get_pixel_color(int x, int y) const
	{
		return getActivePixelSensor(x, y) / (float)num_of_samples;
	}
};


Image RayTracingOpenMP::render()
{
	KdTree kdTree(scene.get());
	Light light(Point(0, 0, -1), Color(255, 255, 255), Color(255, 255, 255));
	kdTree.registerLight(light);
	Resolution resolution = Resolution(width, height);
	Camera camera(2, 2, resolution, 1);
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			Vector vector = camera.get_primary_vector(i, j);
			Color color = kdTree.trace(vector, 20);
			camera.update(i, j, color);
		}
	}
	// return Image
	data = new byte[width * height * BYTES_PER_PIXEL];
	for (int i = 0; i < resolution.height; ++i)
	{
		for (int j = 0; j < resolution.width; ++j)
		{
			Color color = camera.get_pixel_color(resolution.height - i - 1, j);
			int y = height - 1 - i;
			int x = j;
			data[(width * y + x) * BYTES_PER_PIXEL] = std::min(color.red, (float)255);
			data[(width * y + x) * BYTES_PER_PIXEL + 1] = std::min(color.green, (float)255);
			data[(width * y + x) * BYTES_PER_PIXEL + 2] = std::min(color.blue, (float)255);
		}
	}
	return Image(resolution.width, resolution.height, data);
}

RayTracingOpenMP::~RayTracingOpenMP()
{
	delete[] data;
}

RayTracingOpenMP::RayTracingOpenMP()
{
}

void RayTracingOpenMP::setResolution(unsigned width, unsigned height)
{
	this->width = width;
	this->height = height;
}

void RayTracingOpenMP::setSoftShadows(bool var)
{
	// not implemented
}

#include "ContoursOperations.h"
#include <stack>
#include "PerlinNoise.hpp"
#include "RandomGenerator.h"

cv::Mat ContoursOperations::generateIsolines(const GenerationParams& params)
{
	const siv::PerlinNoise::seed_type seed = RandomGenerator::instance().getRandomInt(INT_MAX);
	const siv::PerlinNoise perlin{ seed };

	cv::Mat grad;
	cv::Mat n(params.width, params.height, CV_64FC1);

	double xMul = params.Xmul; // default: 0.005
	double yMul = params.Ymul; // default: 0.005
	int mul = params.mul; // default: 20

#pragma omp parallel for
	for (int j = 0; j < n.rows; ++j)
	{
		for (int i = 0; i < n.cols; ++i)
		{
			double noise = perlin.noise2D_01(i * xMul, j * yMul) * mul;
			int nnn = (int)ceil(noise);
			noise = noise - floor(noise);
			n.at<double>(j, i) = noise;
		}
	}

	cv::Mat grad_x, grad_y;
	cv::Mat abs_grad_x, abs_grad_y;
	cv::Sobel(n, grad_x, CV_64FC1, 1, 0);
	cv::Sobel(n, grad_y, CV_64FC1, 0, 1);
	cv::convertScaleAbs(grad_x, abs_grad_x);
	cv::convertScaleAbs(grad_y, abs_grad_y);
	cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	cv::Mat normalized;
	n.convertTo(normalized, CV_8UC1, 255, 0);
	grad.forEach<uchar>([](uchar& u, const int* pos)
		{
			if (u == 1)
				u = 0;
			else if (u > 1)
			{
				u = 255;
			}
		});
	cv::Mat gradInv = cv::Scalar(255) - grad;

	return gradInv;
}

void ContoursOperations::findContours(const cv::Mat& img, std::vector<Contour>& contours)
{
	int width = img.cols;
	int height = img.rows;

	cv::Mat mat = img.clone();
	for (int m = 0; m < height; ++m)
	{
		for (int n = 0; n < width; ++n)
		{
			if (mat.at<uchar>(m, n) == 255)
			{
				Contour c;
				extractContour(n, m, mat, c.points);
				contours.push_back(std::move(c));
			}
		}
	}

	for (size_t i = 0; i < contours.size(); ++i)
	{
		Contour& c = contours[i];
		c.index = i;
		c.value = i + 1;

		bool isClosed = true;

		if (cv::norm(contours[i].points.front() - contours[i].points.back()) > 3)
		{
			isClosed = false;
		}

		c.isClosed = isClosed;

		c.boundingRect = cv::boundingRect(contours[i].points);
	}
}

void ContoursOperations::extractContour(int x_start, int y_start, cv::Mat& img, std::vector<cv::Point>& contour)
{
	int width = img.cols;
	int height = img.rows;

	int x = x_start;
	int y = y_start;

	auto isContour = [&](int x, int y) -> bool
		{
			if (x < 0 || x >= width || y < 0 || y >= height)
			{
				return false;
			}
			return img.at<uchar>(y, x) == 255;
		};

	std::stack<cv::Point> stack;

	stack.push(cv::Point(x, y));

	Direction direction = Direction::NONE;

	cv::Point prev_point = cv::Point(x, y);

	bool reverse = false;

	while (!stack.empty())
	{
		cv::Point p = stack.top();
		stack.pop();

		if (img.at<uchar>(p.y, p.x) == 255)
		{
			contour.push_back(p);
			img.at<uchar>(p.y, p.x) = 0;
		}

		direction = getDirection(prev_point, p);

		std::vector<cv::Point> neighbours = getOrder(p, direction);

		for (const auto& n : neighbours)
		{
			if (isContour(n.x, n.y))
			{
				stack.push(n);
				break;
			}
		}

		prev_point = p;

		if (stack.empty())
		{
			if (!reverse)
			{
				reverse = true;

				cv::Point point(x, y);

				stack.push(point);
				std::reverse(contour.begin(), contour.end());
				prev_point = point;
			}
		}
	}
}

Direction ContoursOperations::getDirection(cv::Point prev, cv::Point next)
{
	Direction direction = Direction::NONE;
	cv::Point dir = next - prev;
	if (dir.x != 0 && dir.y != 0)
	{
		if (dir.x > 0 && dir.y > 0)
		{
			direction = Direction::BOTTOM_RIGHT;
		}
		else if (dir.x > 0 && dir.y < 0)
		{
			direction = Direction::TOP_RIGHT;
		}
		else if (dir.x < 0 && dir.y > 0)
		{
			direction = Direction::BOTTOM_LEFT;
		}
		else if (dir.x < 0 && dir.y < 0)
		{
			direction = Direction::TOP_LEFT;
		}
	}
	else if (dir.x != 0)
	{
		if (dir.x > 0)
		{
			direction = Direction::RIGHT;
		}
		else
		{
			direction = Direction::LEFT;
		}
	}
	else if (dir.y != 0)
	{
		if (dir.y > 0)
		{
			direction = Direction::DOWN;
		}
		else
		{
			direction = Direction::TOP;
		}
	}
	return direction;
}

std::vector<cv::Point> ContoursOperations::getOrder(cv::Point pt, Direction direction)
{
	switch (direction)
	{
	case Direction::TOP:
		return { cv::Point(pt.x, pt.y - 1), cv::Point(pt.x + 1, pt.y - 1), cv::Point(pt.x - 1, pt.y - 1), cv::Point(pt.x + 1, pt.y), cv::Point(pt.x - 1, pt.y), cv::Point(pt.x + 1, pt.y + 1), cv::Point(pt.x - 1, pt.y + 1) };
	case Direction::TOP_RIGHT:
		return { cv::Point(pt.x + 1, pt.y - 1), cv::Point(pt.x, pt.y - 1), cv::Point(pt.x + 1, pt.y), cv::Point(pt.x - 1, pt.y - 1), cv::Point(pt.x + 1, pt.y + 1), cv::Point(pt.x, pt.y + 1), cv::Point(pt.x - 1, pt.y) };
	case Direction::RIGHT:
		return { cv::Point(pt.x + 1, pt.y), cv::Point(pt.x + 1, pt.y - 1), cv::Point(pt.x + 1, pt.y + 1), cv::Point(pt.x, pt.y - 1), cv::Point(pt.x, pt.y + 1), cv::Point(pt.x - 1, pt.y - 1), cv::Point(pt.x - 1, pt.y + 1) };
	case Direction::BOTTOM_RIGHT:
		return { cv::Point(pt.x + 1, pt.y + 1), cv::Point(pt.x + 1, pt.y), cv::Point(pt.x, pt.y + 1), cv::Point(pt.x + 1, pt.y - 1), cv::Point(pt.x - 1, pt.y + 1), cv::Point(pt.x - 1, pt.y), cv::Point(pt.x, pt.y - 1) };
	case Direction::DOWN:
		return { cv::Point(pt.x, pt.y + 1), cv::Point(pt.x + 1, pt.y + 1), cv::Point(pt.x - 1, pt.y + 1), cv::Point(pt.x + 1, pt.y), cv::Point(pt.x - 1, pt.y), cv::Point(pt.x + 1, pt.y - 1), cv::Point(pt.x - 1, pt.y - 1) };
	case Direction::BOTTOM_LEFT:
		return { cv::Point(pt.x - 1, pt.y + 1), cv::Point(pt.x, pt.y + 1), cv::Point(pt.x - 1, pt.y), cv::Point(pt.x + 1, pt.y + 1), cv::Point(pt.x - 1, pt.y - 1), cv::Point(pt.x, pt.y - 1), cv::Point(pt.x + 1, pt.y) };
	case Direction::LEFT:
		return { cv::Point(pt.x - 1, pt.y), cv::Point(pt.x - 1, pt.y + 1), cv::Point(pt.x - 1, pt.y - 1), cv::Point(pt.x, pt.y + 1), cv::Point(pt.x, pt.y - 1), cv::Point(pt.x + 1, pt.y + 1), cv::Point(pt.x + 1, pt.y - 1) };
	case Direction::TOP_LEFT:
		return { cv::Point(pt.x - 1, pt.y - 1), cv::Point(pt.x, pt.y - 1), cv::Point(pt.x - 1, pt.y), cv::Point(pt.x + 1, pt.y - 1), cv::Point(pt.x - 1, pt.y + 1), cv::Point(pt.x, pt.y + 1), cv::Point(pt.x + 1, pt.y) };
	default:
		return { cv::Point(pt.x, pt.y - 1), cv::Point(pt.x + 1, pt.y - 1), cv::Point(pt.x - 1, pt.y - 1), cv::Point(pt.x + 1, pt.y), cv::Point(pt.x - 1, pt.y), cv::Point(pt.x + 1, pt.y + 1), cv::Point(pt.x - 1, pt.y + 1), cv::Point(pt.x, pt.y + 1) };
	}
}

void ContoursOperations::findDepth(cv::Mat& img, std::vector<Contour>& contours)
{
	int width = img.cols;
	int height = img.rows;

	for (int k = 0; k < contours.size(); ++k)
	{
		Contour& c = contours[k];
		std::set<int> outers_left;
		std::set<int> outers_right;

		int y = c.points[0].y;
		uchar prev = 0;
		for (int x = 0; x < width; ++x)
		{
			uchar val = img.at<uchar>(y, x);

			if (val == prev)
			{
				continue;
			}

			if (val != 0 && val != c.value)
			{
				if (x < c.points[0].x)
				{
					if (outers_left.find(val) == outers_left.end())
					{
						outers_left.insert(val);
					}
					else
					{
						outers_left.erase(val);
					}
				}
				else
				{
					if (outers_right.find(val) == outers_right.end())
					{
						outers_right.insert(val);
					}
					else
					{
						outers_right.erase(val);
					}
				}
				prev = val;
			}

			std::set<int> outers;
			std::set_union(outers_left.begin(), outers_left.end(), outers_right.begin(), outers_right.end(), std::inserter(outers, outers.begin()));

			for (auto iter = outers.begin(); iter != outers.end();)
			{
				int id = *iter;
				Contour& outer = contours[id - 1];
				if ((outer.boundingRect & c.boundingRect) != c.boundingRect)
				{
					iter = outers.erase(iter);
				}
				else
				{
					++iter;
				}
			}

			c.depth = outers.size();
		}
	}
}

void ContoursOperations::fillContours(cv::Mat& contoursMat, const std::vector<Contour>& contours, cv::Mat& drawing)
{
	int max_depth = 0;

	for (auto& c : contours)
	{
		if (c.depth > max_depth)
		{
			max_depth = c.depth;
		}
	}

	ColorScaler scaler(-1, max_depth, cv::Scalar(18, 185, 27), cv::Scalar(20, 20, 185));

	int width = contoursMat.cols;
	int height = contoursMat.rows;

	for (int k = 0; k < contours.size(); ++k)
	{
		const Contour& c = contours[k];
		cv::Point seed_point(-1, -1);

		// try point polygon test to find seed point
		for (auto& pt : c.points)
		{
			for (int m = -1; m <= 1; ++m)
			{
				for (int n = -1; n <= 1; ++n)
				{
					if (m == 0 && n == 0)
					{
						continue;
					}
					cv::Point p(pt.x + m, pt.y + n);
					if (p.x < 0 || p.x >= width || p.y < 0 || p.y >= height)
					{
						continue;
					}
					if (contoursMat.at<uchar>(p) == 0)
					{
						if (cv::pointPolygonTest(c.points, p, false) > 0)
						{
							seed_point = p;
							goto floodfill;
						}
					}
				}
			}
		}

	floodfill:
		if (seed_point.x != -1)
		{
			cv::Scalar color = scaler.getColor(c.depth);
			cv::floodFill(drawing, seed_point, color);
		}
	}

	// fill the holes
	cv::Scalar hole_color = scaler.getColor(-1);
	for (int i = 0; i < drawing.rows; ++i)
	{
		for (int j = 0; j < drawing.cols; ++j)
		{
			if (drawing.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 0, 0))
			{
				cv::floodFill(drawing, cv::Point(j, i), hole_color);
			}
		}
	}
}
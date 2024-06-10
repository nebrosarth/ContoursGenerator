#pragma once
#include <qpoint.h>
#include <random>
#include <map>
#include <qcolor.h>

class RandomGenerator
{
public:
	static RandomGenerator& instance();
	QPoint getRandomPoint(int maxWidth, int maxHeight);
	QColor getRandomColor();
	int getRandomInt(int max);

	RandomGenerator(const RandomGenerator&) = delete;
	void operator=(const RandomGenerator&) = delete;
private:
	RandomGenerator();

	std::random_device dev;
	std::mt19937 rng;
	std::uniform_real_distribution<> distribution;
};


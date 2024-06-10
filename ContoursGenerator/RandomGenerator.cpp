#include "RandomGenerator.h"

RandomGenerator& RandomGenerator::instance()
{
	static RandomGenerator generator;
	return generator;
}

QPoint RandomGenerator::getRandomPoint(int maxWidth, int maxHeight)
{
	return { getRandomInt(maxWidth), getRandomInt(maxHeight) };
}

QColor RandomGenerator::getRandomColor()
{
	return QColor( getRandomInt(255), getRandomInt(255), getRandomInt(255));
}

int RandomGenerator::getRandomInt(int max)
{
	return distribution(rng) * max;
}


RandomGenerator::RandomGenerator() :
	rng(dev()), distribution(0.0, 1.0)
{
}

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

int RandomGenerator::getRandomInt(int max)
{
	auto iter = distributions.try_emplace(max, std::uniform_int_distribution<std::mt19937::result_type>(0, max));
	return iter.first->second(rng);
}


RandomGenerator::RandomGenerator() :
	rng(dev())
{
}

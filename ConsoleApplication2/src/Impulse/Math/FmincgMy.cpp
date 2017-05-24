#include "stdafx.h"
#include <functional>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <ctime>

#include "FmincgMy.h"
#include "../Network/NetworkTrainer.h"
#include "../Math/Vector.h"

// number of extrapolation runs, set to a higher value for smaller ravine landscapes
#define EXT 3.0
// a bunch of constants for line searches
#define RHO 0.01
// RHO and SIG are the constants in the Wolfe-Powell conditions
#define SIG 0.5
// don't reevaluate within 0.1 of the limit of the current bracket
#define INT 0.1
// max 20 function evaluations per line search
#define MAX 20
// maximum allowed slope ratio
#define RATIO 100.0


std::vector<double> FmincgMy::minimize(std::function<CostResult(std::vector<double>)> costFunction, std::vector<double> theta, int length, bool verbose) {
	// we start by setting up all memory that we will need in terms of vectors,
	// while calculating we will just fill this memory (overloaded << uses memcpy)

	// input will be the pointer to our current active parameter set
	Eigen::VectorXd input = Eigen::Map<Eigen::VectorXd>(theta.data(), theta.size());
	Eigen::VectorXd X0(input);
	// search directions
	Eigen::VectorXd s(theta.size());
	// gradients
	Eigen::VectorXd df0(theta.size());
	Eigen::VectorXd df1(theta.size());
	Eigen::VectorXd df2(theta.size());

	// define some integers for bookkeeping and then start
	int M = 0;
	int i = 0; // zero the run length counter
	int red = 1; // starting point
	int ls_failed = 0; // no previous line search has failed
	CostResult evaluateCost = costFunction(Impulse::Math::Vector::eigenToVector(input));
	double f1 = evaluateCost.error;
	df1 << Impulse::Math::Vector::vectorToEigen(evaluateCost.gradient);
	i = i + (length < 0 ? 1 : 0);
	// search direction is steepest
	s << (df1 * -1.0);

	double d1 = (s * -1.0).dot(s); // this is the slope
	double z1 = red / (1.0 - d1); // initial step is red/(|s|+1)

	while (i < abs(length)) {// while not finished
		clock_t begin = clock();
		i = i + (length > 0 ? 1 : 0);// count iterations?!
									 // make a copy of current values
		X0 << input;
		double f0 = f1;
		df0 << df1;
		// begin line search
		// fill our new line searched parameters
		input << input + (s * z1);
		CostResult evaluateCost2 = costFunction(Impulse::Math::Vector::eigenToVector(input));
		double f2 = evaluateCost2.error;
		df2 << Impulse::Math::Vector::vectorToEigen(evaluateCost2.gradient);
		i = i + (length < 0 ? 1 : 0); // count epochs
		double d2 = df2.dot(s);
		// initialize point 3 equal to point 1
		double f3 = f1;
		double d3 = d1;
		double z3 = -z1;
		if (length > 0) {
			M = MAX;
		}
		else {
			M = std::min(MAX, -length - i);
		}
		// initialize quanteties
		int success = 0;
		double limit = -1;

		while (true) {
			while (((f2 > f1 + z1 * RHO * d1) | (d2 > -SIG * d1)) && (M > 0)) {
				// tighten the bracket
				limit = z1;
				double z2 = 0.0;
				double A = 0.0;
				double B = 0.0;
				if (f2 > f1) {
					// quadratic fit
					z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3);
				}
				else {
					// cubic fit
					A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);
					B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
					// numerical error possible - ok!
					z2 = (sqrt(B * B - A * d2 * z3 * z3) - B) / A;
				}
				if (_isnan(z2) || !_finite(z2)) {
					// if we had a numerical problem then bisect
					z2 = z3 / 2.0;
				}
				// don't accept too close to limits
				z2 = std::max(std::min(z2, INT * z3), (1 - INT) * z3);
				// update the step
				z1 = z1 + z2;
				input << input + (s * z2);
				CostResult evaluateCost3 = costFunction(Impulse::Math::Vector::eigenToVector(input));
				f2 = evaluateCost3.error;
				df2 << Impulse::Math::Vector::vectorToEigen(evaluateCost3.gradient);
				M = M - 1;
				i = i + (length < 0 ? 1 : 0); // count epochs
				d2 = df2.dot(s);
				// z3 is now relative to the location of z2
				z3 = z3 - z2;
			}

			if (f2 > f1 + z1 * RHO * d1 || d2 > -SIG * d1) {
				break; // this is a failure
			}
			else if (d2 > SIG * d1) {
				success = 1;
				break; // success
			}
			else if (M == 0) {
				break; // failure
			}
			// make cubic extrapolation
			double A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);
			double B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
			double z2 = -d2 * z3 * z3 / (B + sqrt(B * B - A * d2 * z3 * z3));
			// num prob or wrong sign?
			if (_isnan(z2) || !_finite(z2) || z2 < 0)
				// if we have no upper limit
				if (limit < -0.5) {
					// the extrapolate the maximum amount
					z2 = z1 * (EXT - 1);
				}
				else {
					// otherwise bisect
					z2 = (limit - z1) / 2;
				}
			else if ((limit > -0.5) && (z2 + z1 > limit)) {
				// extraplation beyond max?
				z2 = (limit - z1) / 2; // bisect
			}
			else if ((limit < -0.5) && (z2 + z1 > z1 * EXT)) {
				// extrapolationbeyond limit
				z2 = z1 * (EXT - 1.0); // set to extrapolation limit
			}
			else if (z2 < -z3 * INT) {
				z2 = -z3 * INT;
			}
			else if ((limit > -0.5) && (z2 < (limit - z1) * (1.0 - INT))) {
				// too close to the limit
				z2 = (limit - z1) * (1.0 - INT);
			}
			// set point 3 equal to point 2
			f3 = f2;
			d3 = d2;
			z3 = -z2;
			z1 = z1 + z2;
			// update current estimates
			input << input + (s * z2);
			CostResult evaluateCost3 = costFunction(Impulse::Math::Vector::eigenToVector(input));
			f2 = evaluateCost3.error;
			df2 << Impulse::Math::Vector::vectorToEigen(evaluateCost3.gradient);
			M = M - 1;
			i = i + (length < 0 ? 1 : 0); // count epochs?!
			d2 = df2.dot(s);
		}

		if (success == 1) { // if line search succeeded
			f1 = f2;
			if (verbose) {
				clock_t end = clock();
				double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
				std::cout << "Iteration: " << i << " Cost: " << f1 << " Time: " << elapsed_secs << std::endl;
				// printf("Iteration %d | Cost: %f\r", i, f1);
			}
			// Polack-Ribiere direction: s =
			// (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;
			double df2len = df2.dot(df2);
			double df12len = df1.dot(df2);
			double df1len = df1.dot(df1);
			double numerator = (df2len - df12len) / df1len;
			s << ((s * numerator) - df2);
			std::swap(df1, df2); // swap derivatives
			d2 = df1.dot(s);
			// new slope must be negative
			if (d2 > 0) {
				// otherwise use steepest direction
				s << (df1 * -1.0);
				d2 = (s * -1.0).dot(s);
			}
			// realmin in octave = 2.2251e-308
			// slope ratio but max RATIO
			double thres = d1 / (d2 - 2.2251e-308);
			z1 = z1 * std::min(RATIO, thres);
			d1 = d2;
			ls_failed = 0; // this line search did not fail
		}
		else {
			// restore data from the beginning of the iteration
			input << X0;
			f1 = f0;
			df1 << df0; // restore point from before failed line search
						  // line search failed twice in a row?
			if (ls_failed == 1 || i > abs(length)) {
				break; // or we ran out of time, so we give up
			}
			// swap derivatives
			std::swap(df1, df2);
			// try steepest
			s << (df1 * -1.0);
			d1 = (s * -1.0).dot(s);
			z1 = 1.0 / (1.0 - d1);
			ls_failed = 1; // this line search failed
		}
	}

	if (verbose) {
		// if verbose, we want to switch to a newline now
		printf("\r\n");
	}

	std::vector<double> ret;
	ret.push_back(1.0);
	return ret;
};
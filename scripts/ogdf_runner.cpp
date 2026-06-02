#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ogdf/basic/Graph.h>
#include <ogdf/basic/GraphAttributes.h>
#include <ogdf/energybased/DavidsonHarelLayout.h>
#include <ogdf/energybased/FMMMLayout.h>
#include <ogdf/energybased/GEMLayout.h>
#include <ogdf/energybased/PivotMDS.h>
#include <ogdf/energybased/StressMinimization.h>
#include <ogdf/layered/SugiyamaLayout.h>

namespace {

struct RunnerOptions {
	std::string algorithm;
	std::string inputPath;
	std::string outputPath;
	int seed = 42;
	int gemRounds = 0;
	int fmmmFixedIterations = 0;
	int stressIterations = 0;
	int numberOfPivots = 0;
	bool hasAlgorithm = false;
	bool hasSeed = false;
	bool hasGemRounds = false;
	bool hasFmmmFixedIterations = false;
	bool hasStressIterations = false;
	bool hasNumberOfPivots = false;
};

void skipWhitespace(const std::string& input, std::size_t& position) {
	while (position < input.size()
		&& std::isspace(static_cast<unsigned char>(input[position])) != 0) {
		++position;
	}
}

std::size_t findValueStart(const std::string& input, const std::string& key) {
	const std::string needle = "\"" + key + "\"";
	const std::size_t keyPosition = input.find(needle);
	if (keyPosition == std::string::npos) {
		throw std::runtime_error("missing key: " + key);
	}
	const std::size_t colonPosition = input.find(':', keyPosition + needle.size());
	if (colonPosition == std::string::npos) {
		throw std::runtime_error("missing ':' for key: " + key);
	}
	std::size_t valuePosition = colonPosition + 1;
	skipWhitespace(input, valuePosition);
	return valuePosition;
}

int parseInteger(const std::string& input, std::size_t& position) {
	skipWhitespace(input, position);
	bool negative = false;
	if (position < input.size() && input[position] == '-') {
		negative = true;
		++position;
	}
	if (position >= input.size()
		|| std::isdigit(static_cast<unsigned char>(input[position])) == 0) {
		throw std::runtime_error("expected integer");
	}

	int value = 0;
	while (position < input.size()
		&& std::isdigit(static_cast<unsigned char>(input[position])) != 0) {
		value = (value * 10) + (input[position] - '0');
		++position;
	}
	return negative ? -value : value;
}

int parseCliInteger(const std::string& input) {
	std::size_t position = 0;
	const int value = parseInteger(input, position);
	skipWhitespace(input, position);
	if (position != input.size()) {
		throw std::runtime_error("invalid integer: " + input);
	}
	return value;
}

int parseNodes(const std::string& input) {
	std::size_t position = findValueStart(input, "nodes");
	return parseInteger(input, position);
}

int parseOptionalInteger(const std::string& input, const std::string& key, const int defaultValue) {
	const std::string needle = "\"" + key + "\"";
	const std::size_t keyPosition = input.find(needle);
	if (keyPosition == std::string::npos) {
		return defaultValue;
	}
	const std::size_t colonPosition = input.find(':', keyPosition + needle.size());
	if (colonPosition == std::string::npos) {
		throw std::runtime_error("missing ':' for key: " + key);
	}
	std::size_t valuePosition = colonPosition + 1;
	return parseInteger(input, valuePosition);
}

std::string parseAlgorithm(const std::string& input) {
	std::size_t position = findValueStart(input, "algorithm");
	if (position >= input.size() || input[position] != '"') {
		throw std::runtime_error("expected string for algorithm");
	}
	++position;
	const std::size_t endPosition = input.find('"', position);
	if (endPosition == std::string::npos) {
		throw std::runtime_error("unterminated algorithm string");
	}
	return input.substr(position, endPosition - position);
}

std::vector<std::pair<int, int>> parseEdges(const std::string& input) {
	std::size_t position = findValueStart(input, "edges");
	if (position >= input.size() || input[position] != '[') {
		throw std::runtime_error("expected array for edges");
	}
	++position;

	std::vector<std::pair<int, int>> edges;
	while (true) {
		skipWhitespace(input, position);
		if (position >= input.size()) {
			throw std::runtime_error("unterminated edges array");
		}
		if (input[position] == ']') {
			++position;
			break;
		}
		if (input[position] != '[') {
			throw std::runtime_error("expected edge pair");
		}
		++position;
		int source = parseInteger(input, position);
		skipWhitespace(input, position);
		if (position >= input.size() || input[position] != ',') {
			throw std::runtime_error("expected comma inside edge pair");
		}
		++position;
		int target = parseInteger(input, position);
		skipWhitespace(input, position);
		if (position >= input.size() || input[position] != ']') {
			throw std::runtime_error("expected closing bracket for edge pair");
		}
		++position;
		edges.emplace_back(source, target);

		skipWhitespace(input, position);
		if (position >= input.size()) {
			throw std::runtime_error("unterminated edges array");
		}
		if (input[position] == ',') {
			++position;
			continue;
		}
		if (input[position] == ']') {
			++position;
			break;
		}
		throw std::runtime_error("expected ',' or ']' after edge pair");
	}

	return edges;
}

void validateEdges(
	const std::vector<std::pair<int, int>>& edges,
	const int numNodes
) {
	for (const auto& edge : edges) {
		if (edge.first < 0 || edge.second < 0
			|| edge.first >= numNodes || edge.second >= numNodes) {
			throw std::runtime_error("edge endpoint out of range");
		}
	}
}

std::string requiredStringValue(const int argc, char** argv, int& index) {
	if (index + 1 >= argc) {
		throw std::runtime_error(std::string("missing value for ") + argv[index]);
	}
	++index;
	return std::string(argv[index]);
}

int requiredIntegerValue(const int argc, char** argv, int& index) {
	return parseCliInteger(requiredStringValue(argc, argv, index));
}

void printHelp() {
	std::cout
		<< "Usage: ogdf_runner [--algorithm NAME] [--seed N] [--input PATH] [--output PATH]\n"
		<< "                   [--gem-rounds N] [--fmmm-fixed-iterations N]\n"
		<< "                   [--iterations N] [--number-of-pivots N]\n\n"
		<< "Reads JSON from stdin or --input with keys: nodes, edges, algorithm, seed,\n"
		<< "gemRounds, fmmmFixedIterations, iterations, numberOfPivots. Writes JSON\n"
		<< "positions to stdout or --output.\n"
		<< "Algorithms: gem, fmmm, stress, pivot_mds, davidson_harel, sugiyama.\n";
}

RunnerOptions parseArguments(const int argc, char** argv) {
	RunnerOptions options;
	for (int index = 1; index < argc; ++index) {
		const std::string argument = argv[index];
		if (argument == "--help" || argument == "-h") {
			printHelp();
			std::exit(0);
		}
		if (argument == "--algorithm") {
			options.algorithm = requiredStringValue(argc, argv, index);
			options.hasAlgorithm = true;
			continue;
		}
		if (argument == "--seed") {
			options.seed = requiredIntegerValue(argc, argv, index);
			options.hasSeed = true;
			continue;
		}
		if (argument == "--input") {
			options.inputPath = requiredStringValue(argc, argv, index);
			continue;
		}
		if (argument == "--output") {
			options.outputPath = requiredStringValue(argc, argv, index);
			continue;
		}
		if (argument == "--gem-rounds" || argument == "--gemRounds"
			|| argument == "--rounds") {
			options.gemRounds = requiredIntegerValue(argc, argv, index);
			options.hasGemRounds = true;
			continue;
		}
		if (argument == "--fmmm-fixed-iterations" || argument == "--fmmmFixedIterations"
			|| argument == "--fixed-iterations" || argument == "--steps") {
			options.fmmmFixedIterations = requiredIntegerValue(argc, argv, index);
			options.hasFmmmFixedIterations = true;
			continue;
		}
		if (argument == "--iterations") {
			options.stressIterations = requiredIntegerValue(argc, argv, index);
			options.hasStressIterations = true;
			continue;
		}
		if (argument == "--number-of-pivots" || argument == "--numberOfPivots"
			|| argument == "--n-pivots") {
			options.numberOfPivots = requiredIntegerValue(argc, argv, index);
			options.hasNumberOfPivots = true;
			continue;
		}
		throw std::runtime_error("unknown argument: " + argument);
	}
	return options;
}

std::string readInput(const std::string& inputPath) {
	std::string input;
	std::string line;
	if (inputPath.empty()) {
		while (std::getline(std::cin, line)) {
			input += line;
		}
		return input;
	}

	std::ifstream stream(inputPath);
	if (!stream) {
		throw std::runtime_error("could not open input file: " + inputPath);
	}
	while (std::getline(stream, line)) {
		input += line;
	}
	return input;
}

void writeOutput(const std::string& outputPath, const std::string& payload) {
	if (outputPath.empty()) {
		std::cout << payload << std::endl;
		return;
	}

	std::ofstream stream(outputPath);
	if (!stream) {
		throw std::runtime_error("could not open output file: " + outputPath);
	}
	stream << payload << std::endl;
}

void runLayout(
	const std::string& algorithm,
	ogdf::GraphAttributes& graphAttributes,
	const int gemRounds,
	const int fmmmFixedIterations,
	const int stressIterations,
	const int numberOfPivots,
	const int seed
) {
	ogdf::setSeed(seed);
	std::srand(static_cast<unsigned>(seed));
	if (algorithm == "gem") {
		ogdf::setSeed(seed);
		ogdf::GEMLayout layout;
		if (gemRounds > 0) {
			layout.numberOfRounds(gemRounds);
		}
		layout.call(graphAttributes);
		return;
	}
	if (algorithm == "fmmm") {
		ogdf::FMMMLayout layout;
		layout.randSeed(seed);
		if (fmmmFixedIterations > 0) {
			layout.stopCriterion(ogdf::FMMMOptions::StopCriterion::FixedIterations);
			layout.fixedIterations(fmmmFixedIterations);
		}
		layout.call(graphAttributes);
		return;
	}
	if (algorithm == "stress") {
		ogdf::StressMinimization layout;
		layout.hasInitialLayout(true);
		if (stressIterations > 0) {
			layout.setIterations(stressIterations);
		}
		layout.call(graphAttributes);
		return;
	}
	if (algorithm == "pivot_mds") {
		ogdf::PivotMDS layout;
		if (numberOfPivots > 0) {
			layout.setNumberOfPivots(numberOfPivots);
		}
		layout.call(graphAttributes);
		return;
	}
	if (algorithm == "davidson_harel") {
		ogdf::DavidsonHarelLayout layout;
		layout.call(graphAttributes);
		return;
	}
	if (algorithm == "sugiyama") {
		ogdf::SugiyamaLayout layout;
		layout.call(graphAttributes);
		return;
	}
	if (algorithm == "linlog") {
		throw std::runtime_error("unsupported algorithm: linlog");
	}
	throw std::runtime_error("unknown algorithm: " + algorithm);
}

} // namespace

int main(int argc, char** argv) {
	try {
		const RunnerOptions cliOptions = parseArguments(argc, argv);
		const std::string input = readInput(cliOptions.inputPath);

		const int numNodes = parseNodes(input);
		if (numNodes < 0) {
			throw std::runtime_error("nodes must be non-negative");
		}
		const std::vector<std::pair<int, int>> edges = parseEdges(input);
		validateEdges(edges, numNodes);
		const std::string algorithm = cliOptions.hasAlgorithm
			? cliOptions.algorithm
			: parseAlgorithm(input);
		const int seed = cliOptions.hasSeed
			? cliOptions.seed
			: parseOptionalInteger(input, "seed", 42);
		const int gemRounds = cliOptions.hasGemRounds
			? cliOptions.gemRounds
			: parseOptionalInteger(input, "gemRounds",
				parseOptionalInteger(input, "rounds", 0));
		const int fmmmFixedIterations = cliOptions.hasFmmmFixedIterations
			? cliOptions.fmmmFixedIterations
			: parseOptionalInteger(input, "fmmmFixedIterations",
				parseOptionalInteger(input, "fixed_iterations",
					parseOptionalInteger(input, "steps", 0)));
		const int stressIterations = cliOptions.hasStressIterations
			? cliOptions.stressIterations
			: parseOptionalInteger(input, "iterations", 0);
		const int numberOfPivots = cliOptions.hasNumberOfPivots
			? cliOptions.numberOfPivots
			: parseOptionalInteger(input, "numberOfPivots", 0);

		ogdf::Graph graph;
		ogdf::GraphAttributes graphAttributes(
			graph,
			ogdf::GraphAttributes::nodeGraphics | ogdf::GraphAttributes::edgeGraphics);

		std::vector<ogdf::node> nodes;
		nodes.reserve(static_cast<std::size_t>(numNodes));
		for (int index = 0; index < numNodes; ++index) {
			nodes.push_back(graph.newNode());
		}

		for (const auto& edge : edges) {
			graph.newEdge(nodes[static_cast<std::size_t>(edge.first)],
				nodes[static_cast<std::size_t>(edge.second)]);
		}

		// The runner-owned initial layout is part of the OGDF reference for
		// stress and gives stochastic algorithms a deterministic starting point.
		ogdf::setSeed(seed);
		std::srand(static_cast<unsigned>(seed));
		for (int index = 0; index < numNodes; ++index) {
			graphAttributes.x(nodes[static_cast<std::size_t>(index)]) =
				static_cast<double>(std::rand() % 1000) / 10.0;
			graphAttributes.y(nodes[static_cast<std::size_t>(index)]) =
				static_cast<double>(std::rand() % 1000) / 10.0;
		}

		runLayout(
			algorithm,
			graphAttributes,
			gemRounds,
			fmmmFixedIterations,
			stressIterations,
			numberOfPivots,
			seed);

		std::ostringstream output;
		output << std::setprecision(17);
		output << "{\"positions\":[";
		for (int index = 0; index < numNodes; ++index) {
			if (index > 0) {
				output << ",";
			}
			output << "[" << graphAttributes.x(nodes[static_cast<std::size_t>(index)]) << ","
				   << graphAttributes.y(nodes[static_cast<std::size_t>(index)]) << "]";
		}
		output << "]}";
		writeOutput(cliOptions.outputPath, output.str());
		return 0;
	} catch (const std::exception& exception) {
		std::cerr << exception.what() << std::endl;
		return 1;
	}
}

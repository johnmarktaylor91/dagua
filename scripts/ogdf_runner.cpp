#include <cctype>
#include <iostream>
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

int parseNodes(const std::string& input) {
	std::size_t position = findValueStart(input, "nodes");
	return parseInteger(input, position);
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

void runLayout(
	const std::string& algorithm,
	ogdf::GraphAttributes& graphAttributes
) {
	if (algorithm == "gem") {
		ogdf::GEMLayout layout;
		layout.call(graphAttributes);
		return;
	}
	if (algorithm == "fmmm") {
		ogdf::FMMMLayout layout;
		layout.call(graphAttributes);
		return;
	}
	if (algorithm == "stress") {
		ogdf::StressMinimization layout;
		layout.call(graphAttributes);
		return;
	}
	if (algorithm == "pivot_mds") {
		ogdf::PivotMDS layout;
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

int main() {
	try {
		std::string input;
		std::string line;
		while (std::getline(std::cin, line)) {
			input += line;
		}

		const int numNodes = parseNodes(input);
		if (numNodes < 0) {
			throw std::runtime_error("nodes must be non-negative");
		}
		const std::vector<std::pair<int, int>> edges = parseEdges(input);
		validateEdges(edges, numNodes);
		const std::string algorithm = parseAlgorithm(input);

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

		// Random initial positions — GEM and other force-directed algorithms
		// need non-degenerate starting positions to break symmetry.
		std::srand(static_cast<unsigned>(42));
		for (int index = 0; index < numNodes; ++index) {
			graphAttributes.x(nodes[static_cast<std::size_t>(index)]) =
				static_cast<double>(std::rand() % 1000) / 10.0;
			graphAttributes.y(nodes[static_cast<std::size_t>(index)]) =
				static_cast<double>(std::rand() % 1000) / 10.0;
		}

		runLayout(algorithm, graphAttributes);

		std::cout << "{\"positions\":[";
		for (int index = 0; index < numNodes; ++index) {
			if (index > 0) {
				std::cout << ",";
			}
			std::cout << "[" << graphAttributes.x(nodes[static_cast<std::size_t>(index)]) << ","
					  << graphAttributes.y(nodes[static_cast<std::size_t>(index)]) << "]";
		}
		std::cout << "]}" << std::endl;
		return 0;
	} catch (const std::exception& exception) {
		std::cerr << exception.what() << std::endl;
		return 1;
	}
}

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import org.gephi.graph.api.DirectedGraph;
import org.gephi.graph.api.GraphController;
import org.gephi.graph.api.GraphFactory;
import org.gephi.graph.api.GraphModel;
import org.gephi.graph.api.Node;
import org.gephi.layout.plugin.force.yifanHu.YifanHuLayout;
import org.gephi.layout.plugin.force.yifanHu.YifanHuProportional;
import org.gephi.project.api.ProjectController;
import org.gephi.project.api.Workspace;
import org.openide.util.Lookup;

public final class gephi_layout {
    private static final Gson GSON = new Gson();
    private static final long DEFAULT_SEED = 42L;
    private static final float INITIAL_POSITION_SCALE = 100.0f;
    private static final int DEFAULT_ITERATIONS = 700;
    private static final int MAX_ITERATIONS = 20000;

    private gephi_layout() {}

    private static final class InputData {
        int num_nodes;
        int[][] edges;
        Long seed;
        String algorithm;
        JsonObject params;
    }

    public static void main(String[] args) throws Exception {
        System.setProperty("java.awt.headless", "true");

        InputData input =
                GSON.fromJson(new InputStreamReader(System.in, StandardCharsets.UTF_8), InputData.class);
        if (input == null) {
            throw new IllegalArgumentException("missing input payload");
        }
        if (input.num_nodes < 0) {
            throw new IllegalArgumentException("num_nodes must be non-negative");
        }
        if (input.algorithm != null && !input.algorithm.isEmpty() && !"yifanhu".equalsIgnoreCase(input.algorithm)) {
            throw new IllegalArgumentException("unsupported algorithm: " + input.algorithm);
        }

        ProjectController projectController = Lookup.getDefault().lookup(ProjectController.class);
        GraphController graphController = Lookup.getDefault().lookup(GraphController.class);
        if (projectController == null || graphController == null) {
            throw new IllegalStateException("Gephi toolkit services are unavailable");
        }

        projectController.newProject();
        Workspace workspace = projectController.getCurrentWorkspace();
        if (workspace == null) {
            throw new IllegalStateException("Gephi did not create a workspace");
        }

        GraphModel graphModel = graphController.getGraphModel(workspace);
        DirectedGraph graph = graphModel.getDirectedGraph();
        GraphFactory factory = graphModel.factory();
        Node[] nodes = new Node[input.num_nodes];

        Random random = new Random(input.seed != null ? input.seed.longValue() : DEFAULT_SEED);
        for (int nodeIndex = 0; nodeIndex < input.num_nodes; nodeIndex++) {
            Node node = factory.newNode(String.valueOf(nodeIndex));
            node.setSize(10.0f);
            node.setPosition(
                    randomCoordinate(random),
                    randomCoordinate(random)
            );
            graph.addNode(node);
            nodes[nodeIndex] = node;
        }

        if (input.edges != null) {
            for (int[] edge : input.edges) {
                if (edge == null || edge.length < 2) {
                    continue;
                }
                int sourceIndex = edge[0];
                int targetIndex = edge[1];
                if (sourceIndex < 0 || sourceIndex >= nodes.length || targetIndex < 0 || targetIndex >= nodes.length) {
                    throw new IllegalArgumentException(
                            "edge endpoint out of range: [" + sourceIndex + ", " + targetIndex + "]"
                    );
                }
                graph.addEdge(factory.newEdge(nodes[sourceIndex], nodes[targetIndex], true));
            }
        }

        YifanHuLayout layout = new YifanHuProportional().buildLayout();
        layout.setGraphModel(graphModel);
        applyParams(layout, input.params);
        layout.initAlgo();

        int iterations = resolveIterations(input.params);
        for (int iteration = 0; iteration < iterations && layout.canAlgo(); iteration++) {
            layout.goAlgo();
        }
        layout.endAlgo();

        Map<String, float[]> positions = new LinkedHashMap<>();
        for (Node node : graph.getNodes()) {
            positions.put(String.valueOf(node.getId()), new float[] {node.x(), node.y()});
        }

        OutputStreamWriter writer = new OutputStreamWriter(System.out, StandardCharsets.UTF_8);
        GSON.toJson(positions, writer);
        writer.flush();
    }

    private static float randomCoordinate(Random random) {
        return (random.nextFloat() * 2.0f - 1.0f) * INITIAL_POSITION_SCALE;
    }

    private static int resolveIterations(JsonObject params) {
        int iterations = DEFAULT_ITERATIONS;
        if (params != null && params.has("iterations")) {
            iterations = Math.round(params.get("iterations").getAsFloat());
        }
        if (iterations < 1) {
            return 1;
        }
        return Math.min(iterations, MAX_ITERATIONS);
    }

    private static void applyParams(YifanHuLayout layout, JsonObject params) {
        if (params == null) {
            return;
        }
        applyFloat(params, "optimalDistance", layout::setOptimalDistance);
        applyFloat(params, "relativeStrength", layout::setRelativeStrength);
        applyInteger(params, "quadTreeMaxLevel", layout::setQuadTreeMaxLevel);
        applyFloat(params, "barnesHutTheta", layout::setBarnesHutTheta);
        applyFloat(params, "stepRatio", layout::setStepRatio);
        applyFloat(params, "convergenceThreshold", layout::setConvergenceThreshold);
        applyFloat(params, "initialStep", layout::setInitialStep);
        applyFloat(params, "step", layout::setStep);
        if (params.has("adaptiveCooling")) {
            layout.setAdaptiveCooling(params.get("adaptiveCooling").getAsBoolean());
        }
    }

    private static void applyFloat(JsonObject params, String key, FloatSetter setter) {
        JsonElement element = params.get(key);
        if (element != null && !element.isJsonNull()) {
            setter.apply(element.getAsFloat());
        }
    }

    private static void applyInteger(JsonObject params, String key, IntegerSetter setter) {
        JsonElement element = params.get(key);
        if (element != null && !element.isJsonNull()) {
            setter.apply(element.getAsInt());
        }
    }

    @FunctionalInterface
    private interface FloatSetter {
        void apply(Float value);
    }

    @FunctionalInterface
    private interface IntegerSetter {
        void apply(Integer value);
    }
}

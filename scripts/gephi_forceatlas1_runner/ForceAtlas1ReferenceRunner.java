import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.gephi.graph.api.Edge;
import org.gephi.graph.api.Graph;
import org.gephi.graph.api.GraphController;
import org.gephi.graph.api.GraphFactory;
import org.gephi.graph.api.GraphModel;
import org.gephi.graph.api.Node;
import org.gephi.layout.plugin.forceAtlas.ForceAtlas;
import org.gephi.layout.plugin.forceAtlas.ForceAtlasLayout;
import org.gephi.project.api.ProjectController;
import org.openide.util.Lookup;

public final class ForceAtlas1ReferenceRunner {
    private ForceAtlas1ReferenceRunner() {
    }

    public static void main(String[] args) throws Exception {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        String[] header = reader.readLine().split("\t");
        int numNodes = Integer.parseInt(header[0]);
        int numEdges = Integer.parseInt(header[1]);
        int steps = Integer.parseInt(header[2]);
        double attractionStrength = Double.parseDouble(header[3]);
        double repulsionStrength = Double.parseDouble(header[4]);
        double inertia = Double.parseDouble(header[5]);
        boolean outboundAttractionDistribution = Boolean.parseBoolean(header[6]);
        boolean adjustSizes = Boolean.parseBoolean(header[7]);
        boolean freezeBalance = Boolean.parseBoolean(header[8]);
        double freezeStrength = Double.parseDouble(header[9]);
        double freezeInertia = Double.parseDouble(header[10]);
        double gravity = Double.parseDouble(header[11]);
        double speed = Double.parseDouble(header[12]);
        double cooling = Double.parseDouble(header[13]);
        double maxDisplacement = Double.parseDouble(header[14]);

        Lookup lookup = Lookup.getDefault();
        ProjectController projectController = lookup.lookup(ProjectController.class);
        projectController.newProject();

        GraphModel graphModel = lookup.lookup(GraphController.class).getGraphModel();
        Graph graph = graphModel.getGraph();
        GraphFactory factory = graphModel.factory();
        List<Node> nodes = new ArrayList<>();
        for (int nodeIndex = 0; nodeIndex < numNodes; nodeIndex++) {
            String[] parts = reader.readLine().split("\t");
            Node node = factory.newNode(parts[0]);
            node.setX(Float.parseFloat(parts[1]));
            node.setY(Float.parseFloat(parts[2]));
            node.setSize(Float.parseFloat(parts[3]));
            node.setFixed(Boolean.parseBoolean(parts[4]));
            graph.addNode(node);
            nodes.add(node);
        }

        for (int edgeIndex = 0; edgeIndex < numEdges; edgeIndex++) {
            String[] parts = reader.readLine().split("\t");
            int source = Integer.parseInt(parts[0]);
            int target = Integer.parseInt(parts[1]);
            double weight = Double.parseDouble(parts[2]);
            Edge edge = factory.newEdge(
                "e" + edgeIndex,
                nodes.get(source),
                nodes.get(target),
                0,
                weight,
                true
            );
            graph.addEdge(edge);
        }

        ForceAtlasLayout layout = new ForceAtlas().buildLayout();
        layout.setGraphModel(graphModel);
        layout.resetPropertiesValues();
        layout.setAttractionStrength(attractionStrength);
        layout.setRepulsionStrength(repulsionStrength);
        layout.setInertia(inertia);
        layout.setOutboundAttractionDistribution(outboundAttractionDistribution);
        layout.setAdjustSizes(adjustSizes);
        layout.setFreezeBalance(freezeBalance);
        layout.setFreezeStrength(freezeStrength);
        layout.setFreezeInertia(freezeInertia);
        layout.setGravity(gravity);
        layout.setSpeed(speed);
        layout.setCooling(cooling);
        layout.setMaxDisplacement(maxDisplacement);

        layout.initAlgo();
        for (int step = 0; step < steps; step++) {
            layout.goAlgo();
        }
        layout.endAlgo();

        for (int nodeIndex = 0; nodeIndex < nodes.size(); nodeIndex++) {
            Node node = nodes.get(nodeIndex);
            System.out.printf(
                Locale.ROOT,
                "%d\t%s\t%s%n",
                nodeIndex,
                Float.toString(node.x()),
                Float.toString(node.y())
            );
        }
    }
}

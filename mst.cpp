#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iterator>
#include <algorithm>
#include <math.h>
#include <omp.h>

#define pair_t std::pair<int, int>

typedef struct munro
{
    int rank;
    double altitude;
    std::string name;
    std::string OSGridSquare;
    int OSGridSquareEasting;
    int OSGridSquareNorthing;
    int easting;
    int northing;
    std::string latitude;
    std::string longitude;
} munro_t;

std::map<int, munro_t> munroIDs;

// Data coordinates are of the form AB123123
// which is equivalent to AB1230012300 which
// gives a precision of 100m.
const int INPUT_COORDINATE_PRECISION = 100;
const int MAX = 80000;
int parent[MAX];
std::vector<std::pair<pair_t, pair_t> > GRAPH, MST; // ((weight, FORMS_CYCLE), (pu/rank, pv/rank))

// Current total network weighting.
int total = 0;

// Number of nodes (Munros).
const int N = 282;

// Number of edges in network.
// All Munros are connected -> 282 * 281.
const int E = 79242;

double distanceFromAToB(double x1, double y1, double x2, double y2);
void loadData(char* filename, std::vector<munro_t> &munros);
void computeWeights(std::vector<munro_t> &munros);
int findset(int x, int *parent);
void kruskal();
void reset();
void print(std::vector<munro_t> &munros);

int main(int argc, char** argv)
{
    // Start timing.
    double startTime = omp_get_wtime();

    // Check that the user has provided the
    // correct command-line parameters.
    if (argc < 3)
    {
        std::cout << "USAGE: dijkstra FILENAME NTHREADS NNODES" << std::endl;
        return 1;
    }

    // Get and set the number of threads to use.
    int nthreads = atoi(argv[2]);
    omp_set_num_threads(nthreads);

    // Create a vector to store the input data.
    std::vector<munro_t> munros;

    // Read data into the vector and
    // call reset.
    std::cout << "Loading data..." << std::endl;
    loadData(argv[1], munros);
    reset();

    // Compute the weights.
    std::cout << "Computing weights..." << std::endl;
    computeWeights(munros);

    // Do Kruskal's algorithm
    std::cout << "Finding MST..." << std::endl;
    kruskal();

    // Process results.
    std::cout << "Writing output..." << std::endl;
    print(munros);

    // Stop timing
    double endTime = omp_get_wtime();
    std::cout << "nthreads = " << nthreads << ", time = " << endTime - startTime << std::endl;

    return 0;
}

double distanceFromAToB(double x1, double y1, double x2, double y2)
{
    double dx, dy = 0.0;
    dx = x1 - x2;
    dy = y1 - y2;
    return sqrt(dx*dx + dy*dy);
}

void loadData(char* filename, std::vector<munro_t> &munros)
{
    // Open the filename passed to the command-line.
    std::ifstream inputFile(filename);

    // Create a string to store the current line.
    std::string currentLine;

    // Parse the input file line by line.
    while(std::getline(inputFile, currentLine))
    {
        std::stringstream stream(currentLine);
        std::string field;
        int easting, northing = 0;

        // Create a new munro structure to hold
        // the data from the current line.
        munro_t currentMunro;

        // Read in the parameters.
        std::getline(stream, field, ',');
        currentMunro.rank = atoi(field.c_str());

        std::getline(stream, field, ',');
        currentMunro.OSGridSquare = field;

        if (field == "NC")
        {
           easting = 200000;
           northing = 900000;
        }

        if (field == "NG")
        {
           easting = 100000;
           northing = 800000;
        }

        if (field == "NH")
        {
           easting = 200000;
           northing = 800000;
        }

        if (field == "NJ")
        {
           easting = 300000;
           northing = 800000;
        }

        if (field == "NM")
        {
           easting = 100000;
           northing = 700000;
        }

        if (field == "NN")
        {
           easting = 200000;
           northing = 700000;
        }

        if (field == "NO")
        {
           easting = 300000;
           northing = 700000;
        }

        std::getline(stream, field, ',');
        currentMunro.OSGridSquareEasting = atoi(field.c_str());
        currentMunro.easting = easting + INPUT_COORDINATE_PRECISION*atoi(field.c_str());

        std::getline(stream, field, ',');
        currentMunro.OSGridSquareNorthing = atoi(field.c_str());
        currentMunro.northing = northing + INPUT_COORDINATE_PRECISION*atoi(field.c_str());

        std::getline(stream, field, ',');
        currentMunro.altitude = atoi(field.c_str());

        std::getline(stream, field, ',');
        currentMunro.name = field;

        std::getline(stream, field, ',');
        currentMunro.latitude = field;

        std::getline(stream, field, ',');
        currentMunro.longitude = field;

        // Store for later.
        munros.push_back(currentMunro);

        // Munro ID = rank.
        munroIDs[currentMunro.rank] = currentMunro;
    }
}

void computeWeights(std::vector<munro_t> &munros)
{
    munro_t* munrosArray = &munros[0];

    // Loop 1 - over the munros
    // This can be optimize by sharing the work between
    // threads, as the order in which edges added to the
    // GRAPH vector is not important.

    double x1, y1, x2, y2, d = 0.0;

    #pragma omp parallel for private(x1, y1, x2, y2, d) collapse(1)
    for(int i=0; i < munros.size(); i++)
    {
        // The location of the current munro.
        x1 = munrosArray[i].easting;
        y1 = munrosArray[i].northing;

        // Loop 2 - over the munros.
        for(int j=0; j < munros.size(); j++)
        {
            // Get the location of the second munro.
            x2 = munrosArray[j].easting;
            y2 = munrosArray[j].northing;

            // Don't measure distance between a munro
            // and itself.
            if ((x1 != x2) || (y1 != y2))
            {
                // Calculate the distance between the two.
                d = distanceFromAToB(x1, y1, x2, y2);

                #pragma omp critical
                {
                    // Add to the graph.
                    GRAPH.push_back(std::pair<pair_t, pair_t>(pair_t(d, 0), pair_t(munrosArray[i].rank, munrosArray[j].rank)));
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////

// find the set (tree) which contains the node.
int findset(int x, int *parent)
{
    // If a node is not its own parent...
    if(x != parent[x])
        // Recursviely find the node's parent
        parent[x] = findset(parent[x], parent);
    return parent[x];
}

void kruskal()
{
    int i, pu, pv;
    sort(GRAPH.begin(), GRAPH.end()); // increasing weight

    // Loop 4 - over all edges
    // Each edge has two nodes, u and v.
    for(i=total=0; i<E; i++)
    {
        //Do we ALREADY know that this edge forms a cycle?
        if(GRAPH[i].first.second == 0)
        {
            // Which set is node u in?
            pu = findset(GRAPH[i].second.first, parent);

            // Which set is node v in?
            pv = findset(GRAPH[i].second.second, parent);

            // If they belong to different sets then...
            if(pu != pv)
            {
                // Add the edge's weighting to the total weighting
                // of the MST.
                total += GRAPH[i].first.first;

                // Add the edge to the minimum spanning tree (MST)
                MST.push_back(GRAPH[i]);

                // Link the nodes
                parent[pu] = parent[pv];
            }
        }
    }
}

void reset()
{
    // Loop 3 - order of access does not matter.
    #pragma omp parallel
    {
        #pragma for nowait
        for(int i=1; i<=N; i++)
        {
            parent[i] = i;
        }
    }
}

void print(std::vector<munro_t> &munros)
{
    std::ofstream outputFile("mst.kml");

    // Insert kml header including a folder for the lines.
    outputFile << "<?xml version=\"1.0\" encoding=\"UTF-8\"?><kml xmlns=\"http://earth.google.com/kml/2.2\"><Document><name>Munro MST</name>" << std::endl
               << "<Folder><name>lines</name><open>1</open>" << std::endl;
    int i, sz;
    sz = MST.size();

    // Loop 5 - over all edges in the network;
    for(i=0; i<sz; i++)
    {
        int ID1 = MST[i].second.first;
        int ID2 = MST[i].second.second;

        // Print to the console.
        //std::cout << munroIDs[ID1].name << "(" << ID1 << ") --> "
        //          << munroIDs[ID2].name << "(" << ID2 << ") = "
        //          << MST[i].first.first/1000.0 << " Km" << std::endl;

        // Write edges to kml file.
        outputFile << "<Placemark><name>" << MST[i].first.first/1000.0 << " Km"
                   << "</name><LineString><tessellate>1</tessellate><coordinates>" << std::endl
                   << munroIDs[ID1].longitude << ","
                   << munroIDs[ID1].latitude << ","
                   << "0.000000" << std::endl
                   << munroIDs[ID2].longitude << ","
                   << munroIDs[ID2].latitude << ","
                   << "0.000000" << std::endl
                   << "</coordinates></LineString></Placemark>" << std::endl;
    }


    // Create a folder in the kml to store the points.
    outputFile << "</Folder><Folder><name>points</name><open>1</open>" << std::endl;

    // Loop 6 - over all the munros and add a point for each to the kml file.
    for(i=1; i<N+1; i++)
    {
        outputFile << "<Placemark><name></name><description><![CDATA[<div dir=\"ltr\">"
                   << munroIDs[i].rank << ". " << munroIDs[i].name
                   << " (" << munroIDs[i].altitude << "m)" << std::endl
                   << "</div>]]></description><Point><coordinates>" << std::endl
                   << munroIDs[i].longitude << ","
                   << munroIDs[i].latitude << ","
                   << "0.000000</coordinates></Point></Placemark>" << std::endl;
    }

    // Insert kml footer
    outputFile << "</Folder></Document></kml>" << std::endl;

    std::cout << "Minimum cost = " << total/1000.0 << " Km" << std::endl;
}

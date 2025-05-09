#include "globals.cuh"

// screen size
__constant__ unsigned int SCREEN_WIDTH_DEVICE;
__constant__ unsigned int SCREEN_HEIGHT_DEVICE;

unsigned int SCREEN_WIDTH_HOST;
unsigned int SCREEN_HEIGHT_HOST;

using namespace std;

std::string getPathStr() {
    std::string str = __FILE__;

    std::string path = str.substr(0, str.rfind("/"));
    pathStr = path.substr(0, path.rfind("/"));

    return str;
}

Vector3 unitCubeCoords[8] = { Vector3(0,0,0), Vector3(1,0,0), Vector3(1,1,0), Vector3(0,1,0), Vector3(0,0,1), Vector3(1,0,1), Vector3(1,1,1), Vector3(0,1,1) };
Vector3 cameraPos(0, 0, -1000);
Vector3 cameraAngle(0, 0, 0);

PointLight pointLight(Vector3(0,0,0), Vector3(255, 255, 255));

Material mainMaterial(Vector3(255, 255, 255), 0, 0, 0);

std::string pathStr = getPathStr();

float PLAYER_SPEED = 1; // 1
float PLAYER_SPEED_FLYING = 0.2; // 0.2
float PLAYER_TURN_Y_SPEED = 0.1;

bool mouseControls = true;
bool doGravity = false;
bool showFps = true;
bool doOldRendering = false;
bool generateNewChunks = false;
bool showBorder = true;

int shift = 0;
int shiftX = 0;
int shiftY = 0;
int shiftZ = 0;

//BS::thread_pool threadPool(MAX_THREADS_AMOUNT);
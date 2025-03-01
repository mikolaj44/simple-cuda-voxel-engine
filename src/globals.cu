#include "globals.cuh"

using namespace std;

std::string getPathStr() {

    std::string str = __FILE__;

    std::string path = str.substr(0, str.rfind("/"));
    pathStr = path.substr(0, path.rfind("/"));

    return str;
}

Vector3::Vector3() {};

Vector3::Vector3(float x_, float y_, float z_) {

    x = x_;
    y = y_;
    z = z_;
}

void Vector3::scale(float val) {

    x *= val;
    y *= val;
    z *= val;
}

Vector3 unitCubeCoords[8] = { Vector3(0,0,0), Vector3(1,0,0), Vector3(1,1,0), Vector3(0,1,0), Vector3(0,0,1), Vector3(1,0,1), Vector3(1,1,1), Vector3(0,1,1) };
Vector3 cameraPos(0, 0, 0);
Vector3 cameraAngle(0, 0, 0);

std::string pathStr = getPathStr();

float PLAYER_SPEED = 1; // 1
float PLAYER_SPEED_FLYING = 0.2; // 0.2
float PLAYER_TURN_Y_SPEED = 0.1;

float halfHorFOV;
float halfVerFOV;

bool mouseControls = true;
bool doGravity = false;
bool showFps = false;
bool doOldRendering = true;
bool generateNewChunks = false;
bool showBorder = false;

int shift = 0;
int shiftX = 0;
int shiftY = 0;
int shiftZ = 0;

//BS::thread_pool threadPool(MAX_THREADS_AMOUNT);
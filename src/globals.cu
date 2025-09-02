#include "globals.cuh"

using namespace std;

Vector3<> cameraPos(0, 0, -1000);
Vector3<> cameraAngle(0, 0, 0);

PointLight pointLight(Vector3<>(0,0,0), Vector3<>(255, 255, 255));

Material mainMaterial(Vector3<>(255, 255, 255), 0, 0, 0);

float PLAYER_SPEED = 1; // 1
float PLAYER_SPEED_FLYING = 0.2; // 0.2
float PLAYER_TURN_Y_SPEED = 0.1;

bool mouseControls = true;
bool doGravity = false;
bool showFps = true;
bool doOldRendering = false;
bool generateNewChunks = false;
bool showBorder = true;
#include "pixel_drawing.cuh"

void plotLineLow(uchar4* pixels, int x1, int y1, int x2, int y2, int r, int g, int b, int a) {

    float dx = x2 - x1;
    float dy = y2 - y1;
    int yi = 1;

    if (dy < 0) {
        yi = -1;
        dy = -dy;
    }

    float d = (2 * dy) - dx;
    float y = y1;

    for (int x = x1; x <= x2; x++) {

        if (x < 0)
            return;
        if (y < 0)
            return;
        if (x >= SCREEN_WIDTH)
            return;
        if (y >= SCREEN_HEIGHT)
            return;

        if (x < 0)
            return;
        if (y < 0)
            return;
        if (x >= SCREEN_WIDTH)
            return;
        if (y >= SCREEN_HEIGHT)
            return;

        setPixel(pixels, x, y, r, g, b, a);

        if (d > 0) {
            y = y + yi;
            d = d + (2 * (dy - dx));
        }
        else
            d = d + 2 * dy;
    }
}

void plotLineHigh(uchar4* pixels, int x1, int y1, int x2, int y2, int r, int g, int b, int a) {

    float dx = x2 - x1;
    float dy = y2 - y1;
    int xi = 1;

    if (dx < 0) {
        xi = -1;
        dx = -dx;
    }

    float d = (2 * dx) - dy;
    float x = x1;

    for (int y = y1; y <= y2; y++) {

        if (x < 0)
            return;
        if (y < 0)
            return;
        if (x >= SCREEN_WIDTH)
            return;
        if (y >= SCREEN_HEIGHT)
            return;

        if (x < 0)
            return;
        if (y < 0)
            return;
        if (x >= SCREEN_WIDTH)
            return;
        if (y >= SCREEN_HEIGHT)
            return;

        setPixel(pixels, x, y, r, g, b, a);

        if (d > 0) {
            x = x + xi;
            d = d + (2 * (dx - dy));
        }
        else
            d = d + 2 * dx;
    }
}

vector<pair<int, int>> plotLineLowPoints(int x1, int y1, int x2, int y2) {

    float dx = x2 - x1;
    float dy = y2 - y1;
    int yi = 1;

    vector<pair<int, int>> v;

    if (dy < 0) {
        yi = -1;
        dy = -dy;
    }

    float d = (2 * dy) - dx;
    float y = y1;

    for (int x = x1; x <= x2; x++) {

        v.push_back(make_pair(x, y));

        if (d > 0) {
            y = y + yi;
            d = d + (2 * (dy - dx));
        }
        else
            d = d + 2 * dy;
    }

    return v;
}

vector<pair<int, int>> plotLineHighPoints(int x1, int y1, int x2, int y2) {

    float dx = x2 - x1;
    float dy = y2 - y1;
    int xi = 1;

    vector<pair<int, int>> v;

    if (dx < 0) {
        xi = -1;
        dx = -dx;
    }

    float d = (2 * dx) - dy;
    float x = x1;

    for (int y = y1; y <= y2; y++) {

        v.push_back(make_pair(x, y));

        if (d > 0) {
            x = x + xi;
            d = d + (2 * (dx - dy));
        }
        else
            d = d + 2 * dx;
    }

    return v;
}

void drawLine(uchar4* pixels, int x1, int y1, int x2, int y2, int r, int g, int b, int a){

    /*if (x1 < 0)
        x1 = 0;
    if (y1 < 0)
        y1 = 0;
    if (x1 >= SCREEN_WIDTH)
        x1 = SCREEN_WIDTH - 1;
    if (y1 >= SCREEN_HEIGHT)
        y1 = SCREEN_HEIGHT - 1;

    if (x2 < 0)
        x2 = 0;
    if (y2 < 0)
        y2 = 0;
    if (x2 >= SCREEN_WIDTH)
        x2 = SCREEN_WIDTH - 1;
    if (y2 >= SCREEN_HEIGHT)
        y2 = SCREEN_HEIGHT - 1;*/

    if (x1 == x2) {

        float minY = min(y1, y2);
        float maxY = max(y1, y2);

        //for (int y = minY; y <= maxY; y++)
        //    SetPixel(x1, y, r, g, b, a);
        return;
    }

    if (abs(y2 - y1) < abs(x2 - x1)) {

        if (x1 > x2)
            plotLineLow(pixels, x2, y2, x1, y1, r, g, b, a);
        else
            plotLineLow(pixels, x1, y1, x2, y2, r, g, b, a);
    }
    else {
        if (y1 > y2)
            plotLineHigh(pixels, x2, y2, x1, y1, r, g, b, a);
        else
            plotLineHigh(pixels, x1, y1, x2, y2, r, g, b, a);
    }
}

vector<pair<int, int>> LinePoints(int x1, int y1, int x2, int y2)
{

    vector<pair<int, int>> points;

    if (x1 == x2) {

        float minY = min(y1, y2);
        float maxY = max(y1, y2);

        for (int y = minY; y <= maxY; y++)
            points.push_back(make_pair(x1, y));
        return points;
    }

    vector<pair<int, int>> v;

    if (abs(y2 - y1) < abs(x2 - x1)) {

        if (x1 > x2)
            v = plotLineLowPoints(x2, y2, x1, y1);
        else
            v = plotLineLowPoints(x1, y1, x2, y2);
    }
    else {
        if (y1 > y2)
            v = plotLineHighPoints(x2, y2, x1, y1);
        else
            v = plotLineHighPoints(x1, y1, x2, y2);
    }

    points.insert(points.end(), v.begin(), v.end());

    return points;
}

float angleNormalize(float a) {

    if (a > 2 * M_PI)
        a -= 2 * M_PI;
    else if (a < 0)
        a += 2 * M_PI;

    return a;
}


float* _3d2dProjection(float x_, float y_, float z_) {

    static float coords[2];

    float alphaHor = atan2f((cameraPos.x - x_), (cameraPos.z - z_)) + M_PI;
    float alphaVer = atan2f((cameraPos.y - y_), (cameraPos.z - z_)) + M_PI;

    float leftAngleHor = cameraAngle.y + halfHorFOV;

    float rightAngleHor = cameraAngle.y - halfHorFOV;

    float leftAngleVer = cameraAngle.x + halfVerFOV;

    float rightAngleVer = cameraAngle.x - halfVerFOV;

    bool isVisible = (angleNormalize(alphaHor - rightAngleHor) < angleNormalize(leftAngleHor - rightAngleHor)); //https://stackoverflow.com/questions/66799475/how-to-elegantly-find-if-an-angle-is-between-a-range

    //isVisible = true;

    //cout << leftAngleHor * 180 / M_PI << " " << rightAngleHor * 180 / M_PI << " " << alphaHor * 180 / M_PI << endl;

    if (!isVisible) {
        coords[0] = INT_MAX;
        coords[1] = INT_MAX;

        //cout << leftAngleHor * 180 / M_PI << " " << rightAngleHor * 180 / M_PI << " " << alphaHor * 180 / M_PI << endl;
        return coords;
    }

    //if ((leftAngleVer > rightAngleVer && (alphaVer < rightAngleVer || alphaVer > leftAngleVer)) || (leftAngleVer < rightAngleVer && (alphaVer > leftAngleVer && alphaVer < rightAngleVer))) {
    //    coords[0] = INT_MAX;
    //    coords[1] = INT_MAX;

    //    //cout << leftAngle * 180 / M_PI << " " << rightAngle * 180 / M_PI << " " << alpha * 180 / M_PI << endl;
    //    return coords;
    //}

    //cout << alpha * 180 / M_PI << endl;

    float x = x_ - cameraPos.x;
    float y = y_ - cameraPos.y;
    float z = z_ - cameraPos.z;

    float dX = 0, dY = 0, dZ = 0;

    dX = cos(cameraAngle.y) * (sin(cameraAngle.z) * y + cos(cameraAngle.z) * x) - sin(cameraAngle.y) * z;

    dY = sin(cameraAngle.x) * (cos(cameraAngle.y) * z + sin(cameraAngle.y) * (sin(cameraAngle.z) * y + cos(cameraAngle.z) * x)) + cos(cameraAngle.x) * (cos(cameraAngle.z) * y - sin(cameraAngle.z) * x);

    dZ = cos(cameraAngle.x) * (cos(cameraAngle.y) * z + sin(cameraAngle.y) * (sin(cameraAngle.z) * y + cos(cameraAngle.z) * x)) - sin(cameraAngle.x) * (cos(cameraAngle.z) * y - sin(cameraAngle.z) * x);

    if (dZ == 0) {
        coords[0] = INT_MAX;
        coords[1] = INT_MAX;
    }
    else {
        coords[0] = int(dX * FOCAL_LENGTH / dZ * SCALE_V + SCREEN_WIDTH  / 2);
        coords[1] = int(dY * FOCAL_LENGTH / dZ * SCALE_V + SCREEN_HEIGHT / 2);

        //coords[0] = (FOCAL_LENGTH * x) / (z + FOCAL_LENGTH) * 100 + SCREEN_WIDTH / 2;
        //coords[1] = (FOCAL_LENGTH * y) / (z + FOCAL_LENGTH) * 100 + SCREEN_HEIGHT / 2;
    }

    //coords[0] = (FOCAL_LENGTH * x) / (z + FOCAL_LENGTH) + SCREEN_WIDTH / 2;
    //coords[1] = (FOCAL_LENGTH * y) / (z + FOCAL_LENGTH) + SCREEN_HEIGHT / 2;

    //cout << dX << " " << dY << " " << dZ << endl;

    //cout << "player position: " << cameraPos.x << " " << cameraPos.y << " " << cameraPos.z << endl;

    //cout << "real coordinates: " << x_ << " " << y_ << " " << z_ << endl;

    /*cout << "projected coordinates: " << coords[0] << " " << coords[1] << " " << endl;
    cout << "parameters: " << dX << " " << dY << " " << dZ << endl;
    cout << endl;*/

    return coords;
}
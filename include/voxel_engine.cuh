#pragma once

class VoxelEngine {
public:
    static void init(int WINDOW_WIDTH, int WINDOW_HEIGHT);
    
    static void cleanup();

    static void displayFrame();
private:
    static bool wasInitialized;
}